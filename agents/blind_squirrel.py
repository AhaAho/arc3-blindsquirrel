from __future__ import annotations

import logging
import os
import random
import sys
import time
from collections import deque
from typing import Any

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from arcengine import FrameData, GameAction, GameState
from torch.utils.data import DataLoader, Dataset
from torchvision import models as torchvision_models

from .agent import Agent

logger = logging.getLogger(__name__)

AGENT_MAX_ACTIONS = 50000
AGENT_LOOP_SLEEP = 0.1
AGENT_E = 0.5
RWEIGHT_MIN = 0.1
RWEIGHT_RANK_DISCOUNT = 0.5
RWEIGHT_NO_DISCOUNT = 0.5
MODEL_LR = 1e-4
MODEL_BATCH_SIZE = 32
MODEL_NUM_EPOCHS = 10
MODEL_SCORE_MAG = 1
MODEL_MAX_TRAIN_TIME = 15
USE_PRETRAINED_BACKBONE = os.getenv("BLINDSQUIRREL_PRETRAINED_BACKBONE", "").lower() in {
    "1",
    "true",
    "yes",
}

ACTION7_INDEX = 5
CLICK_ACTION_START = 6
ACTION_FEATURE_SIZE = 7

sys.setrecursionlimit(2000)


def _frame_score(latest_frame: FrameData) -> int:
    return latest_frame.levels_completed


def _current_grid(latest_frame: FrameData) -> list[list[int]]:
    if not latest_frame.frame:
        return []
    return latest_frame.frame[-1]


def _available_action_ids(
    available_actions: list[int] | list[int | GameAction],
) -> set[int]:
    action_ids: set[int] = set()
    for action in available_actions:
        if isinstance(action, GameAction):
            action_ids.add(int(action.value))
        else:
            action_ids.add(int(action))
    return action_ids


def _resolve_device(device: str | None) -> torch.device:
    if not device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device)
    if requested.type != "cuda":
        return requested

    if not torch.cuda.is_available():
        logger.warning(
            "Requested device %s for BlindSquirrel, but CUDA is unavailable; using CPU",
            device,
        )
        return torch.device("cpu")

    if requested.index is not None and requested.index >= torch.cuda.device_count():
        fallback_index = max(torch.cuda.device_count() - 1, 0)
        fallback = torch.device(f"cuda:{fallback_index}")
        logger.warning(
            "Requested device %s for BlindSquirrel, but only %s CUDA device(s) are visible; using %s",
            device,
            torch.cuda.device_count(),
            fallback,
        )
        return fallback

    return requested


class BlindSquirrel(Agent):
    MAX_ACTIONS: int = AGENT_MAX_ACTIONS

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        requested_device = kwargs.pop("device", None)
        super().__init__(*args, **kwargs)
        self.device = _resolve_device(requested_device)
        logger.info("BlindSquirrel %s using device %s", self.game_id, self.device)
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        self.game_counter = 0
        self.level_counter = 0
        self.graph: StateGraph | None = None
        self.current_state: State | None = None
        self.prev_state: State | None = None
        self.prev_action: int | None = None
        self._bootstrapped = False

    def _bootstrap_from_frame(self, latest_frame: FrameData) -> None:
        self.game_counter = 0
        self.level_counter = 0
        self.game_id = latest_frame.game_id or self.game_id
        self.graph = StateGraph(self.device)
        self.current_state = self.graph.get_state(latest_frame)
        self.graph.add_init_state(self.current_state)
        self.prev_state = None
        self.prev_action = None
        self._bootstrapped = True

    def process_latest_frame(self, latest_frame: FrameData) -> None:
        time.sleep(AGENT_LOOP_SLEEP)

        if latest_frame.full_reset:
            self._reset_tracking()
            if latest_frame.state in (GameState.NOT_FINISHED, GameState.WIN):
                self._bootstrap_from_frame(latest_frame)
            return

        if latest_frame.state is GameState.NOT_PLAYED:
            self._reset_tracking()
            return

        if latest_frame.state is GameState.GAME_OVER:
            return

        if latest_frame.state not in (GameState.NOT_FINISHED, GameState.WIN):
            raise ValueError(f"Unexpected game state: {latest_frame.state}")

        if not self._bootstrapped:
            self._bootstrap_from_frame(latest_frame)
            return

        if self.graph is None:
            raise RuntimeError("BlindSquirrel state graph was not initialized")

        self.current_state = self.graph.get_state(latest_frame)
        if self.prev_state is None or self.prev_action is None:
            return

        self.game_counter += 1
        if self.current_state.score > self.prev_state.score:
            self.level_counter = 0
        else:
            self.level_counter += 1

        self.graph.update(self.prev_state, self.prev_action, self.current_state)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        self.process_latest_frame(latest_frame)
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        if latest_frame.full_reset and not self._bootstrapped:
            self._reset_tracking()

        if not self._bootstrapped:
            self._bootstrap_from_frame(latest_frame)

        if self.current_state is None:
            raise RuntimeError("BlindSquirrel current state was not initialized")

        if (
            AGENT_E < random.random()
            and latest_frame.levels_completed > 0
            and len(self.current_state.future_states) > 0
        ):
            action = self.get_model_action()
        else:
            action = self.get_rweights_action()

        action_obj = self.current_state.get_action_obj(action)
        self.prev_state = self.current_state
        self.prev_action = action
        return action_obj

    def get_model_action(self) -> int:
        if self.current_state is None or self.graph is None:
            raise RuntimeError("BlindSquirrel current state graph is unavailable")

        model = self.graph.action_model
        if model is None:
            return self.get_rweights_action()

        game_id = self.current_state.game_id
        score = self.current_state.score
        device = next(model.parameters()).device
        model.eval()
        model_values: dict[int, float] = {}

        with torch.no_grad():
            for action, rweight in self.current_state.action_rweights.items():
                if rweight == 0:
                    continue
                x_s = torch.as_tensor(
                    self.current_state.frame, dtype=torch.long
                ).unsqueeze(0)
                x_a = self.current_state.get_action_tensor(action).unsqueeze(0)
                value = model(x_s.to(device), x_a.to(device)).item()
                if rweight is None:
                    value *= self._rweight_calc(game_id, score, action)
                model_values[action] = value

        if not model_values:
            logger.warning("No weighted model actions available for %s", self.game_id)
            return self.current_state.get_fallback_action()

        return max(model_values, key=model_values.get)

    def get_rweights_action(self) -> int:
        if self.current_state is None:
            raise RuntimeError("BlindSquirrel current state was not initialized")

        game_id = self.current_state.game_id
        score = self.current_state.score
        actions: list[int] = []
        weights: list[float] = []

        for action, rweight in self.current_state.action_rweights.items():
            if rweight == 0:
                continue
            weight = (
                self._rweight_calc(game_id, score, action)
                if rweight is None
                else 1.0
            )
            actions.append(action)
            weights.append(weight)

        if not actions:
            logger.warning("No weighted actions available for %s", self.game_id)
            return self.current_state.get_fallback_action()

        return random.choices(actions, weights=weights, k=1)[0]

    def _rweight_calc(self, game_id: str, score: int, action: int) -> float:
        if self.graph is None:
            raise RuntimeError("BlindSquirrel graph is unavailable")

        no, yes = self.graph.action_counter.get((game_id, score, action), [0, 0])
        if yes > 0:
            return max(RWEIGHT_MIN, yes / (no + yes))
        if action < CLICK_ACTION_START:
            return max(RWEIGHT_MIN, RWEIGHT_NO_DISCOUNT**no)
        rank = action - CLICK_ACTION_START
        return max(
            RWEIGHT_MIN,
            (RWEIGHT_RANK_DISCOUNT**rank) * (RWEIGHT_NO_DISCOUNT**no),
        )

    @staticmethod
    def _move_batch_to_device(
        batch: dict[str, torch.Tensor], device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {name: tensor.to(device) for name, tensor in batch.items()}


class State:
    def __init__(self, latest_frame: FrameData) -> None:
        if latest_frame.state not in (GameState.NOT_FINISHED, GameState.WIN):
            raise ValueError(f"Unsupported state for BlindSquirrel.State: {latest_frame.state}")

        self.latest_frame = latest_frame
        self.game_id = latest_frame.game_id
        self.score = _frame_score(latest_frame)

        if latest_frame.state is GameState.NOT_FINISHED and _current_grid(latest_frame):
            self.frame = tuple(tuple(inner) for inner in _current_grid(latest_frame))
            self.object_data = self.get_object_data()
        else:
            self.frame = "WIN"
            self.object_data = []

        self.future_states: dict[int, State] = {}
        self.prior_states: list[tuple[State, int]] = []
        self.available_action_ids = _available_action_ids(latest_frame.available_actions)
        self.num_actions = len(self.object_data) + CLICK_ACTION_START
        self.action_rweights = {i: None for i in range(self.num_actions)}
        self._apply_available_action_mask()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return (self.game_id, self.score, self.frame) == (
            other.game_id,
            other.score,
            other.frame,
        )

    def __hash__(self) -> int:
        return hash((self.game_id, self.score, self.frame))

    def _apply_available_action_mask(self) -> None:
        if int(GameAction.ACTION1.value) not in self.available_action_ids:
            self.action_rweights[0] = 0
        if int(GameAction.ACTION2.value) not in self.available_action_ids:
            self.action_rweights[1] = 0
        if int(GameAction.ACTION3.value) not in self.available_action_ids:
            self.action_rweights[2] = 0
        if int(GameAction.ACTION4.value) not in self.available_action_ids:
            self.action_rweights[3] = 0
        if int(GameAction.ACTION5.value) not in self.available_action_ids:
            self.action_rweights[4] = 0
        if int(GameAction.ACTION7.value) not in self.available_action_ids:
            self.action_rweights[ACTION7_INDEX] = 0
        if int(GameAction.ACTION6.value) not in self.available_action_ids:
            for idx in range(CLICK_ACTION_START, self.num_actions):
                self.action_rweights[idx] = 0

    def get_object_data(self) -> list[dict[str, Any]]:
        grid = np.array(self.frame)
        object_data: list[dict[str, Any]] = []
        orig_idx = 0
        for colour in range(16):
            raw_labeled, _ = scipy.ndimage.label(grid == colour)
            slices = scipy.ndimage.find_objects(raw_labeled)
            for i, slc in enumerate(slices):
                if slc is None:
                    continue
                mask = raw_labeled[slc] == (i + 1)
                area = int(np.sum(mask))
                h = slc[0].stop - slc[0].start
                w = slc[1].stop - slc[1].start
                bbox_area = h * w
                size = h * w / (64 * 64)
                regularity = area / bbox_area
                ys, xs = np.nonzero(mask)
                y_centroid = float(ys.mean() + slc[0].start)
                x_centroid = float(xs.mean() + slc[1].start)
                object_data.append(
                    {
                        "orig_idx": orig_idx,
                        "colour": colour,
                        "slice": slc,
                        "mask": mask,
                        "area": area,
                        "bbox_area": bbox_area,
                        "size": size,
                        "regularity": regularity,
                        "y_centroid": y_centroid,
                        "x_centroid": x_centroid,
                    }
                )
                orig_idx += 1

        object_data.sort(
            key=lambda obj: (
                -obj["regularity"],
                -obj["area"],
                -obj["colour"],
                obj["orig_idx"],
            )
        )
        return object_data

    def get_action_tensor(self, action: int) -> torch.Tensor:
        action_type = torch.zeros(ACTION_FEATURE_SIZE)
        colour = torch.zeros(16)
        regularity = torch.zeros(1)
        size = torch.zeros(1)
        y_centroid = torch.zeros(1)
        x_centroid = torch.zeros(1)

        if action <= 4:
            action_type[action] = 1
            regularity[0] = 1
            size[0] = 1
            y_centroid[0] = -1
            x_centroid[0] = -1
        elif action == ACTION7_INDEX:
            action_type[ACTION7_INDEX] = 1
            regularity[0] = 1
            size[0] = 1
            y_centroid[0] = -1
            x_centroid[0] = -1
        else:
            action_obj = self.object_data[action - CLICK_ACTION_START]
            action_type[-1] = 1
            colour[action_obj["colour"]] = 1
            regularity[0] = action_obj["regularity"]
            size[0] = action_obj["size"]
            y_centroid[0] = action_obj["y_centroid"]
            x_centroid[0] = action_obj["x_centroid"]

        return torch.cat([action_type, colour, regularity, size, y_centroid, x_centroid])

    def get_action_obj(self, action: int) -> GameAction:
        if action == 0:
            return GameAction.ACTION1
        if action == 1:
            return GameAction.ACTION2
        if action == 2:
            return GameAction.ACTION3
        if action == 3:
            return GameAction.ACTION4
        if action == 4:
            return GameAction.ACTION5
        if action == ACTION7_INDEX:
            return GameAction.ACTION7
        return self.get_click_action_obj(action)

    def get_click_action_obj(self, action: int) -> GameAction:
        obj = self.object_data[action - CLICK_ACTION_START]
        slc = obj["slice"]
        mask = obj["mask"]
        local_coords = np.argwhere(mask)
        idx = np.random.choice(len(local_coords))
        local_y, local_x = local_coords[idx]
        global_y = slc[0].start + int(local_y)
        global_x = slc[1].start + int(local_x)
        new_action = GameAction.ACTION6
        new_action.set_data({"x": global_x, "y": global_y})
        return new_action

    def get_fallback_action(self) -> int:
        candidates: list[int] = []
        if int(GameAction.ACTION1.value) in self.available_action_ids:
            candidates.append(0)
        if int(GameAction.ACTION2.value) in self.available_action_ids:
            candidates.append(1)
        if int(GameAction.ACTION3.value) in self.available_action_ids:
            candidates.append(2)
        if int(GameAction.ACTION4.value) in self.available_action_ids:
            candidates.append(3)
        if int(GameAction.ACTION5.value) in self.available_action_ids:
            candidates.append(4)
        if int(GameAction.ACTION7.value) in self.available_action_ids:
            candidates.append(ACTION7_INDEX)
        if int(GameAction.ACTION6.value) in self.available_action_ids:
            candidates.extend(range(CLICK_ACTION_START, self.num_actions))

        if candidates:
            return random.choice(candidates)
        return 0

    def zero_back(self) -> None:
        if all(v == 0 for v in self.action_rweights.values()):
            for state, action in self.prior_states:
                if state.action_rweights[action] == 1:
                    state.action_rweights[action] = 0
                    state.zero_back()


class StateGraph:
    def __init__(self, device: torch.device) -> None:
        self.init_state: State | None = None
        self.milestones: dict[tuple[str, int], State] = {}
        self.states: set[State] = set()
        self.action_counter: dict[tuple[str, int, int], list[int]] = {}
        self.game_id: str | None = None
        self.action_model: ActionModel | None = None
        self.device = device

    def get_state(self, latest_frame: FrameData) -> State:
        new_obj = State(latest_frame)
        existing_obj = next((state for state in self.states if state == new_obj), None)
        if existing_obj is not None:
            return existing_obj
        self.states.add(new_obj)
        return new_obj

    def update(self, prev_state: State, action: int, new_state: State) -> None:
        game_id = prev_state.game_id
        score = prev_state.score

        if action in prev_state.future_states:
            if prev_state.future_states[action] == new_state:
                return
            logger.warning("Markov violation detected for %s", game_id)

        prev_state.future_states[action] = new_state
        new_state.prior_states.append((prev_state, action))
        self.action_counter.setdefault((game_id, score, action), [0, 0])

        if new_state == prev_state:
            self.action_counter[(game_id, score, action)][0] += 1
            prev_state.action_rweights[action] = 0
            prev_state.zero_back()
            logger.warning("Bad action detected for %s", game_id)
            return

        self.action_counter[(game_id, score, action)][1] += 1
        prev_state.action_rweights[action] = 1

        if new_state == self.milestones.get((game_id, score)):
            return

        if new_state.score > prev_state.score:
            self.add_milestone(new_state)
            if AGENT_E < 1:
                self.train_model(game_id, score + 1)

    def add_milestone(self, state: State) -> None:
        key = (state.game_id, state.score)
        existing = self.milestones.get(key)
        if existing is not None:
            if existing != state:
                raise ValueError(f"Milestone collision for {key}")
            return
        self.milestones[key] = state

    def add_init_state(self, state: State) -> None:
        self.init_state = state
        self.add_milestone(state)
        self.game_id = state.game_id

    def get_level_training_data(
        self, old_milestone: State, new_milestone: State
    ) -> list[dict[str, torch.Tensor]]:
        final_grid = _current_grid(new_milestone.latest_frame)
        final_frame = (
            tuple(tuple(inner) for inner in final_grid) if final_grid else old_milestone.frame
        )
        state_data: dict[State, dict[str, Any]] = {
            new_milestone: {"distance": 0, "frame": final_frame}
        }
        max_distance = 0

        queue: deque[State] = deque([new_milestone])
        while queue:
            state = queue.popleft()
            current_distance = state_data[state]["distance"]
            for prev_state, _ in state.prior_states:
                if prev_state.score != old_milestone.score:
                    continue
                if prev_state in state_data:
                    continue
                state_data[prev_state] = {
                    "frame": prev_state.frame,
                    "distance": current_distance + 1,
                }
                max_distance = max(max_distance, current_distance + 1)
                queue.append(prev_state)

        queue = deque([old_milestone])
        while queue:
            state = queue.popleft()
            current_distance = state_data[state]["distance"]
            for _, future_state in state.future_states.items():
                if future_state.score != old_milestone.score:
                    continue
                if future_state in state_data:
                    continue
                state_data[future_state] = {
                    "frame": future_state.frame,
                    "distance": current_distance + 1,
                }
                max_distance = max(max_distance, current_distance + 1)
                queue.append(future_state)

        denom = max(max_distance, 1)
        final_data: list[dict[str, torch.Tensor]] = []
        for state in self.states:
            if state.score != old_milestone.score or state not in state_data:
                continue
            for action, future_state in state.future_states.items():
                if future_state not in state_data:
                    continue
                state_tensor = torch.tensor(state_data[state]["frame"], dtype=torch.long)
                action_tensor = state.get_action_tensor(action)
                if state.action_rweights[action] == 0:
                    score = torch.tensor(-MODEL_SCORE_MAG, dtype=torch.float32).unsqueeze(0)
                else:
                    state_distance = state_data[state]["distance"]
                    future_distance = state_data[future_state]["distance"]
                    score = torch.tensor(
                        MODEL_SCORE_MAG * (state_distance - future_distance) / denom,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                final_data.append(
                    {
                        "state": state_tensor,
                        "action": action_tensor,
                        "score": score,
                    }
                )
        return final_data

    def train_model(self, game_id: str, max_score: int, verbose: bool = True) -> None:
        device = self.device
        self.action_model = ActionModel(game_id).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.action_model.parameters(), lr=MODEL_LR)

        data: list[dict[str, torch.Tensor]] = []
        for score in range(0, max_score):
            old_milestone = self.milestones[(game_id, score)]
            new_milestone = self.milestones[(game_id, score + 1)]
            data.extend(self.get_level_training_data(old_milestone, new_milestone))

        if not data:
            logger.warning("No training data available for %s at level %s", game_id, max_score)
            return

        dataset = ActionModelDataset(data)
        dataloader = DataLoader(
            dataset,
            batch_size=MODEL_BATCH_SIZE,
            collate_fn=dataset.collate,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        start_time = time.time()
        for epoch in range(MODEL_NUM_EPOCHS):
            self.action_model.train()
            running_loss = 0.0
            for batch in dataloader:
                batch = BlindSquirrel._move_batch_to_device(batch, device)
                state_b = batch["state"]
                action_b = batch["action"]
                score_b = batch["score"]
                optimizer.zero_grad(set_to_none=True)
                preds = self.action_model(state_b, action_b)
                loss = criterion(preds, score_b)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * score_b.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            if verbose:
                logger.info(
                    "Model training | %s - level %s | %s frames | epoch %s/%s | loss %.4f",
                    game_id,
                    max_score,
                    len(dataloader.dataset),
                    epoch + 1,
                    MODEL_NUM_EPOCHS,
                    epoch_loss,
                )

            if time.time() - start_time > MODEL_MAX_TRAIN_TIME * 60:
                logger.warning(
                    "Reached maximum train time of %s minutes for %s",
                    MODEL_MAX_TRAIN_TIME,
                    game_id,
                )
                break


class ActionModel(nn.Module):
    def __init__(self, game_id: str) -> None:
        super().__init__()
        self.game_id = game_id
        self.grid_symbol_embedding = nn.Embedding(16, 16)
        self.stem = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        weights_enum = getattr(torchvision_models, "ResNet18_Weights", None)
        pretrained_weights = (
            weights_enum.IMAGENET1K_V1
            if weights_enum is not None and USE_PRETRAINED_BACKBONE
            else None
        )
        try:
            backbone = torchvision_models.resnet18(weights=pretrained_weights)
        except Exception as exc:
            logger.warning(
                "Falling back to randomly initialized ResNet18 for BlindSquirrel: %s",
                exc,
            )
            backbone = torchvision_models.resnet18(weights=None)

        backbone.conv1 = nn.Identity()
        backbone.bn1 = nn.Identity()
        backbone.relu = nn.Identity()
        backbone.maxpool = nn.Identity()
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.state_fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 64),
            nn.ReLU(inplace=True),
        )
        self.action_fc = nn.Sequential(
            nn.Linear(ACTION_FEATURE_SIZE + 16 + 4, 64),
            nn.ReLU(inplace=True),
        )
        self.head_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.grid_symbol_embedding(state)
        x = x.permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.state_fc(x)
        x_a = self.action_fc(action)
        x = torch.cat([x, x_a], dim=1)
        return self.head_fc(x)


class ActionModelDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, examples: list[dict[str, torch.Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]

    @staticmethod
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        state = torch.stack([example["state"] for example in batch], dim=0)
        action = torch.stack([example["action"] for example in batch], dim=0)
        score = torch.stack([example["score"] for example in batch], dim=0)
        return {"state": state, "action": action, "score": score}

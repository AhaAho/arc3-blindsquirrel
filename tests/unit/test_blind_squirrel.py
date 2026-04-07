from unittest.mock import Mock

import pytest
import torch
from arcengine import FrameData, GameAction, GameState

from agents.blind_squirrel import ACTION7_INDEX, BlindSquirrel, State, StateGraph


def make_frame(
    *,
    state: GameState = GameState.NOT_FINISHED,
    levels_completed: int = 0,
    available_actions: list[int] | None = None,
    full_reset: bool = False,
    game_id: str = "ft09",
) -> FrameData:
    return FrameData(
        game_id=game_id,
        frame=[
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        ],
        state=state,
        levels_completed=levels_completed,
        available_actions=available_actions or [],
        full_reset=full_reset,
    )


def make_agent() -> BlindSquirrel:
    return BlindSquirrel(
        card_id="card-1",
        game_id="ft09",
        agent_name="blindsquirrel",
        ROOT_URL="https://example.com",
        record=False,
        arc_env=Mock(),
    )


@pytest.mark.unit
def test_choose_action_bootstraps_from_initial_observation():
    agent = make_agent()
    latest_frame = make_frame(available_actions=[1, 6])

    action = agent.choose_action(agent.frames, latest_frame)

    assert agent._bootstrapped is True
    assert agent.graph is not None
    assert agent.graph.init_state == agent.current_state
    assert agent.prev_state == agent.current_state
    assert action.name in {"ACTION1", "ACTION6"}


@pytest.mark.unit
def test_state_accepts_integer_available_actions_and_maps_action7():
    state = State(make_frame(available_actions=[7]))

    assert state.action_rweights[ACTION7_INDEX] is None
    assert state.get_action_obj(ACTION7_INDEX) is GameAction.ACTION7
    assert state.action_rweights[0] == 0
    assert state.action_rweights[1] == 0
    assert state.action_rweights[2] == 0
    assert state.action_rweights[3] == 0
    assert state.action_rweights[4] == 0
    for idx in range(6, state.num_actions):
        assert state.action_rweights[idx] == 0


@pytest.mark.unit
def test_move_batch_to_device_moves_every_tensor_like_value():
    class FakeTensor:
        def __init__(self) -> None:
            self.seen_devices: list[torch.device] = []

        def to(self, device: torch.device) -> "FakeTensor":
            self.seen_devices.append(device)
            return self

    batch = {
        "state": FakeTensor(),
        "action": FakeTensor(),
        "score": FakeTensor(),
    }
    device = torch.device("cpu")

    moved = BlindSquirrel._move_batch_to_device(batch, device)

    assert moved == batch
    assert all(tensor.seen_devices == [device] for tensor in batch.values())


@pytest.mark.unit
def test_state_graph_zeroes_actions_that_return_to_current_milestone():
    graph = StateGraph(torch.device("cpu"))
    milestone = State(make_frame(available_actions=[1, 6]))
    detour = State(
        make_frame(
            available_actions=[1, 6],
            game_id="ft09",
        )
    )
    detour.frame = tuple(
        tuple(cell + 1 for cell in row)
        for row in detour.frame  # type: ignore[arg-type]
    )

    action = 0
    graph.add_init_state(milestone)
    graph.states.add(milestone)
    graph.states.add(detour)
    detour.action_rweights[action] = 1

    graph.update(detour, action, milestone)

    assert graph.action_counter[(detour.game_id, detour.score, action)] == [0, 1]
    assert detour.action_rweights[action] == 0

# Blind Squirrel 迁移审计记录

本文档记录 `arc3-blindsquirrel` 中 Blind Squirrel 迁移到新版 `arc_agi` / `arcengine` 后，到底改了哪些地方、为什么改，以及当前仍然和原作者 preview 版本不完全一致的点。

## 1. 审计基线

- 当前迁移仓库：`/mnt/hdd/chenyy/AGI3/arc3-blindsquirrel`
- 当前主分支最近两次迁移相关 commit：
  - `68b8449` `Migrate Blind Squirrel to latest arc_agi`
  - `f84db71` `Fix Blind Squirrel loop pruning regression`
- 上游对比基线：`upstream/main` at `135f20a`
- 原作者参考实现：`/mnt/hdd/chenyy/AGI3/wd13ca-blindsquirrel-ref`，分支 `blindsquirrel`

本次实际从 `upstream/main..main` 的代码差异共有 9 个文件：

- `agents/__init__.py`
- `agents/blind_squirrel.py`
- `agents/swarm.py`
- `agents/templates/langgraph_thinking/vision.py`
- `main.py`
- `pyproject.toml`
- `run_blindsquirrel_batches.py`
- `tests/conftest.py`
- `tests/unit/test_blind_squirrel.py`

## 2. 迁移到底改了什么

### 2.1 `agents/blind_squirrel.py`

这是迁移主体文件，基本可以分成 4 类改动。

#### A. 适配新版 `FrameData` / `arcengine`

- 将原版依赖的 `latest_frame.score` 改为 `latest_frame.levels_completed`
- 增加 `_frame_score()`，统一把“当前关卡进度”解释为 `levels_completed`
- 增加 `_current_grid()`，兼容新版 `FrameData.frame`
- `State.score` 不再读旧字段，而是统一来自 `_frame_score(latest_frame)`

这样做的原因是上游 `ARC-AGI-3-Agents` 已经从旧的 `score` 字段迁到 `levels_completed`。

#### B. 修复新版环境下的 agent 生命周期问题

原作者版本假设：

- 游戏开始前会先看到 `NOT_PLAYED + empty frame`
- 然后再进入第一帧有效观测

新版 `arc_agi` 包下，这个假设不稳定，导致 Blind Squirrel 初版迁移时会在首帧 bootstrap 上出错，严重时直接出现“0 actions 秒退”。

因此迁移时新增了：

- `_reset_tracking()`
- `_bootstrap_from_frame()`
- `latest_frame.full_reset` 处理
- `choose_action()` 中的首帧 bootstrap 兜底

这一组改动的目标是：

- 即使没有旧版那种空首帧，也能直接从第一张有效观测开始建图
- reset / full reset 后重新初始化状态图，不再依赖旧生命周期

#### C. 动作空间和 `available_actions` 兼容层

这是迁移里改动最大的部分之一。

新增了：

- `_available_action_ids()`：把 `available_actions` 统一转成 `int`
- `ACTION7_INDEX = 5`
- `CLICK_ACTION_START = 6`
- `ACTION_FEATURE_SIZE = 7`

并相应改了：

- `State._apply_available_action_mask()`
- `State.get_action_tensor()`
- `State.get_action_obj()`
- `State.get_fallback_action()`
- `_rweight_calc()`

这部分改动的目的有两个：

1. 新版 `available_actions` 在运行时可能是整数而不是旧对象，不能再直接按旧写法判断。
2. 新版 ARC 包已经出现 `ACTION7`，迁移时我把它显式编进了 Blind Squirrel 的内部动作空间。

需要特别说明：

- 这一步是“迁移到当前新版动作空间”的改动，不是原作者 preview 版本的严格复刻。
- 原作者 README 写的是 `5 basic actions + click + reset`，没有把 `ACTION7` 算进核心动作建模。
- 因此这一点本身就是当前迁移版和 preview 参考版的结构性偏差之一。

#### D. 训练和设备相关兼容

新增或调整了：

- `_resolve_device()`：显式解析 `cuda:0/cuda:1`
- `BlindSquirrel._move_batch_to_device()`：训练 batch 随模型 device 走
- `StateGraph(device)`：让图和模型知道目标 device
- `train_model()` 中 dataloader + batch 搬运逻辑
- `ActionModel` 中 torchvision 新 API 适配
- `USE_PRETRAINED_BACKBONE` 环境变量开关

迁移原因：

- 新版环境下，一次跑多个游戏时需要显式分卡，否则全堆到默认 `cuda`
- 训练 batch 和模型 device 不一致会炸混设备错误
- torchvision 新版本不再用旧的 `pretrained=True/False` 形式

这里还有一个关键行为差异：

- 当前迁移版默认 `USE_PRETRAINED_BACKBONE=False`
- 只有设 `BLINDSQUIRREL_PRETRAINED_BACKBONE=1` 才启用 ImageNet 预训练权重

这么做最初是为了离线环境下避免偷偷拉权重，但这也意味着：

- 当前迁移版默认行为和原作者 README 里宣称的“pre-trained ResNet-18”不一致
- 如果不显式开环境变量，就不是原版思路

#### E. `f84db71` 的后续修补：loop pruning 回归修复

在初次迁移后，又补了一次关键回归修复。

修复点在 `StateGraph.update()`：

- 当动作执行后回到了当前 score 对应的 milestone 状态时
- 这类动作现在会被置零并向前 `zero_back()`

这一步是为了恢复更接近原作者版本的 loop-pruning 行为，避免“绕一圈又回到当前 milestone”的动作被当成可继续尝试的正样本保留下来。

## 2.2 `agents/__init__.py`

改动很小，但必需：

- `from .blind_squirrel import BlindSquirrel`
- `__all__` 中加入 `BlindSquirrel`

作用：

- 让 `AVAILABLE_AGENTS` 能自动注册出 `blindsquirrel`
- 让 `main.py --help` 和 `--agent=blindsquirrel` 能识别这个 agent

## 2.3 `agents/swarm.py`

这里的改动主要是让 Blind Squirrel 在新版环境里能稳定批跑。

新增：

- `BLINDSQUIRREL_DEVICES` 环境变量解析
- `_resolve_device_cycle()`
- `_get_agent_kwargs()`

作用：

- 允许通过 `BLINDSQUIRREL_DEVICES=cuda:0,cuda:1` 轮转分配 GPU
- 在创建每个 game 的 agent 时把目标 `device` 传进去

如果没有这部分改动：

- 25 个 game 同进程跑时默认都落到一张卡
- 与多卡批跑需求不匹配

## 2.4 `main.py`

这里改的是运行入口，不是 Blind Squirrel 算法本体，但对“能不能真正离线跑”非常关键。

新增：

- `_configured_operation_mode()`
- `_get_local_games()`
- 离线优先从本地 `environment_files` 枚举游戏
- API 不通时 fallback 到本地枚举
- `RUN_LOG_PATH` 控制日志输出路径

作用：

- `OPERATION_MODE=offline` 时不再依赖 `/api/games`
- 让 `arc3-blindsquirrel` 可以在离线环境里枚举并运行本地缓存游戏
- 给分批 runner 提供单批独立日志文件

## 2.5 `run_blindsquirrel_batches.py`

这是迁移过程中新增的辅助脚本，不属于原作者代码。

作用：

- 以离线模式顺序分批运行 Blind Squirrel
- 避免 25 个游戏同时挂在一个进程里把 RAM 顶爆
- 为每一批写独立日志

它解决的是迁移之后的工程运行问题：

- Blind Squirrel 能跑，但 25 并发占用内存太高
- 分批跑可以释放批次间内存，方便拿完整一轮结果

## 2.6 `agents/templates/langgraph_thinking/vision.py`

这一处和 Blind Squirrel 算法无关，是顺手修的全仓导入兼容问题。

改动：

- 不再使用 `PIL.ImageDraw.Coords`
- 改成仓库内自己的类型别名 `HighlightCoords`

原因：

- 新版 Pillow 不再暴露 `ImageDraw.Coords`
- 会导致 `agents/__init__.py` 导入链在加载其他模板 agent 时直接崩溃
- 结果是就算你想跑 `blindsquirrel`，也会因为 import 链里别的模块炸掉而启动失败

因此这一步虽然不是 Blind Squirrel 逻辑，但属于“为了让 `main.py --agent=blindsquirrel` 真正跑起来”而必须修的兼容改动。

## 2.7 `pyproject.toml`

新增依赖：

- `scipy`
- `torch`
- `torchvision`

作用：

- Blind Squirrel 依赖 `scipy.ndimage`
- 动作价值模型依赖 `torch` / `torchvision`

需要单独记录一个现实问题：

- 当前 `pyproject.toml` 里的版本约束是 `torch>=2.11.0`、`torchvision>=0.26.0`
- 但实际跑通的环境是 `torch 2.5.1+cu121`、`torchvision 0.20.1+cu121`

也就是说：

- 这个文件表达的“声明依赖”和真实可运行环境并不完全一致
- 如果直接 `uv sync`，有可能拉回不兼容轮子

## 2.8 `tests/conftest.py`

改动很小：

- `FrameData` / `GameState` 改从 `arcengine` 导入
- `sample_frame` 使用 `levels_completed`

原因：

- 上游数据结构已经迁到 `arcengine`
- 旧测试夹具里的字段名和导入路径已经不对

## 2.9 `tests/unit/test_blind_squirrel.py`

这是迁移后的回归保护测试，新增了几类检查：

- 首帧 bootstrap 不秒退
- `available_actions` 用整数时也能处理
- `ACTION7` 的内部映射正确
- batch device 搬运逻辑正确
- loop-pruning 回归修复生效

这些测试主要保护的是：

- 新包接口兼容
- 多卡/训练兼容
- 我在迁移过程中修掉的两个关键回归点

## 3. 迁移后的实际结果

从运行结果看，迁移工作达成的是：

- Blind Squirrel 能在新版 `arc_agi` 下启动
- 能离线枚举和加载本地游戏
- 能稳定跑满批次并产出 scorecard
- 支持多卡轮转和分批批跑

但它没有达成的是：

- 还没有严格复现出 preview 宣传里的效果

本地观察到：

- 第一轮 25 public games 全量批跑，总共完成 `18 levels`
- preview public 三题 `ft09 + ls20 + vc33` 合计是 `5 levels`
- 后续修补后再单独跑 preview 三题，总 levels 仍然是 `5`

也就是说：

- 修补改变了每题内部行为轨迹
- 但没有把 preview public 三题的总 levels 继续抬高

## 4. 当前仍和原作者 preview 版本不一致的地方

这些差异需要明确记下来，否则容易把“迁移成功”和“结果严格复现成功”混为一谈。

### 4.1 评测条件不同

原作者 README 明确写的是：

- preview 竞赛是 `3 public training games + 3 private evaluation games`

而当前本地常见的两种跑法是：

- 25 个 public games 全量跑
- 只跑 public 三题 `ft09/ls20/vc33`

这两种都不是 preview 官方总榜的原始评测条件。

### 4.2 游戏版本不同

当前本地缓存的是新版 `arc_agi` 拉下来的 environment 版本。

这不保证和 2025 preview 时官方使用的游戏版本完全一致。

### 4.3 动作空间建模不同

原作者 README 的写法是：

- `5 basic actions + click + reset`

当前迁移版则：

- 把 `ACTION7` 明确纳入了内部动作空间建模
- `ACTION_FEATURE_SIZE` 从原版隐含的 `6` 扩展到了 `7`
- `CLICK_ACTION_START` 从原版逻辑的 `5` 变成了 `6`

这意味着当前迁移版更贴近“新版 ARC 包动作空间”，但更远离“preview 原版 Blind Squirrel 假设”。

### 4.4 预训练 backbone 的默认行为不同

原作者 README 写得很明确：

- Action Value Model 基于 pre-trained ResNet-18

而当前迁移版默认：

- `BLINDSQUIRREL_PRETRAINED_BACKBONE` 不设时，`USE_PRETRAINED_BACKBONE=False`

所以如果运行时没显式设置：

- 当前迁移版默认不是原作者 README 所说的预训练模型路径

## 5. 对这次迁移工作的结论

### 已经完成的部分

- Blind Squirrel 代码已成功接到新版 `arc_agi` / `arcengine`
- 修通了首帧 bootstrap、device、offline 枚举、批跑、Pillow 兼容等工程问题
- 现在可以在本地离线稳定跑完整批次

### 还没有完成的部分

- 还没有做到“严格复现 preview 原始 Blind Squirrel 效果”
- 当前迁移版仍带有若干“为了兼容最新版工具包而引入的行为偏差”

### 最值得做的下一步

如果目标是“尽量接近 preview 原作者版”，建议单独做一个 `preview_compat` 模式：

- 禁用 `ACTION7`
- 恢复原版动作编码假设
- 默认开启 pretrained backbone
- 只在 `ft09 / ls20 / vc33` 上做对照

如果目标是“面向未来 Kaggle / 最新 ARC 包”，则当前迁移版更适合作为继续改进的新基线，但不应再把它直接称为 preview 结果的严格复现。

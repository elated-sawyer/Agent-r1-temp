# 新增「延迟回溯」模式（retro_hybrid）

**日期**：2026-04-28

---

## 一、任务一句话概述

新增一种介于 `retro_noback_V4`（无回溯）和 `retro`（每步可回溯）之间的混合模式 `retro_hybrid_V4`，核心行为：

> **强制向前直到 hit `tool.maxstep` → 才开放一次 `back_state` → 成功回溯后回到强制前进 → 可循环**。

目的：避免现有 `retro` 模式"每步都能回退 → 模型总在浅层 state 打转 → 无法生成深路径"的问题，同时保留在真正走死路时修正的能力。

---

## 二、现状与问题

仓库中已有两套相关实现，新模式要**夹在它们之间**：

| 现有模式 | env 类 | 入口 | 行为 | 问题 |
|---|---|---|---|---|
| `retro` | [`ToolEnvRetro`](Agent-r1-temp/agent_r1/tool/tool_env_retro.py) | [`main_agent_retro.py`](Agent-r1-temp/agent_r1/src/main_agent_retro.py) | 每步都可 `back_state` | 模型在浅层反复回退，生成不出长路径，performance 严重掉 |
| `retro_noback_V4` | [`ToolEnvRetroNoBack`](Agent-r1-temp/agent_r1/tool/tool_env_retro_noback.py) | [`main_agent_retro_noback.py`](Agent-r1-temp/agent_r1/src/main_agent_retro_noback.py) | 无 `back_state` 工具 | hit maxstep 就死掉，没有纠错机会 |

注意 `retro_noback_V4` 中已经有一段"hit maxstep 的占位逻辑"但**不生效**（因为 `back_state` 没被注册）：

```python
# agent_r1/tool/tool_env_retro_noback.py:896-899
if current_step >= self.maxstep:
    current_message += StepFail.format(n=self.maxstep)
    current_message += FailBack                       # 让模型用 back_state，但工具不存在
    self.back_flag = True
```

我们要做的就是**让这段逻辑真的可用**。

---

## 三、新模式 `retro_hybrid_V4` 行为规范

### Phase A — 强制前进（默认，和 noback 完全一样）
- 工具：`single_step_retro`, `select_reaction`
- `back_state` 工具在 schema 中**存在但被门控**（见下方"设计决策"）
- 推进流程完全复用 `ToolEnvRetroNoBack` 的逻辑

### Phase B — 触发回溯（hit maxstep）
**触发条件**：某次 `select_reaction` 推进后，`current_step >= self.maxstep`

**env 行为**：
1. 拼接 `StepFail` + `FailBack` 消息（与现有 noback 一致）
2. 设 `self.back_flag = True`
3. 此时模型可以调 `back_state(state=...)` 并被接受

### Phase B → Phase A — 回溯成功
`back_state` 被成功执行（逻辑参考 [`tool_env_retro.py:876-915`](Agent-r1-temp/agent_r1/tool/tool_env_retro.py#L876-L915)）：

1. `self.current_state` / `self.current_molid` 跳到目标历史 state
2. `self.back_flag = False` ← **回到 Phase A**
3. 拼接 `GoBackMolecule` 或 `GoBackReaction` 消息（提示当前在哪个 state、未解分子是什么）
4. 模型继续用 `single_step_retro` / `select_reaction` 往前走

此后如果再次 hit maxstep，Phase B 可再次开启（允许多轮 A↔B 循环）。每个历史 state 的 step 数独立保存在 `self.step_dict`，回溯后从 `step_dict[target_state]+1` 继续累加——**现有代码已是这样，无需改动**。

### Phase A 时模型强行调 back_state
- env 返回 error 消息，不改变状态，trajectory 继续
- 建议措辞：`"You can only use back_state after reaching the maximum number of reaction steps. Please continue with single_step_retro or select_reaction."`

---

## 四、关键设计决策（实现前先选）

### 决策 1：`back_state` 工具的可见性

| 方案 | 优点 | 缺点 |
|---|---|---|
| **A. 始终在 tools schema 里，env 层做门控**（推荐） | schema 稳定、实现简单、不用改 vLLM 调用 | 模型可能在 Phase A 乱调，产生 1 条 error 消息 |
| B. 动态 tools：Phase A 不给 back_state，Phase B 才给 | 对模型干净 | 每轮都要重算 tools，框架改动大 |

**推荐方案 A**，在代码注释里标明选择的原因。

### 决策 2：新 env 的继承策略

- **推荐**：fork `tool_env_retro_noback.py` 得到 `tool_env_retro_hybrid.py`，在其中补上 back_state 分支（仓库现有风格就是 fork 而非继承，见 `tool_env_retro.py` / `tool_env_retro_.py` / `tool_env_retro_noback.py` 三兄弟）
- 备选：`class ToolEnvRetroHybrid(ToolEnvRetroNoBack): ...` override 少数方法——更 DRY，但跟仓库风格不一致

### 决策 3：成功合成后的行为（SuccessBack）

现有 `retro` 模式在合成成功时也会 `back_flag=True` 并提示模型"找更短路径"（[`tool_env_retro.py:867-868`](Agent-r1-temp/agent_r1/tool/tool_env_retro.py#L867-L868)）。

**新模式 `retro_hybrid` 不要复制这个行为**——和 `noback` 对齐，**找到一条路径就终止 trajectory**。否则又会把模型推回"反复回溯找更短"的 shallow 陷阱，违背本模式的初衷。

---

## 五、实现 checklist

### 5.1 新建文件

1. **`agent_r1/tool/tool_env_retro_hybrid.py`** — 主要工作
   - 起点：复制 `tool_env_retro_noback.py`
   - 类名：`ToolEnvRetroHybrid`（构造签名与 `ToolEnvRetroNoBack` 一致，`force_noloop` 要保留）
   - 改动点：
     - 在 `wrap_tool_args` 中加 `tool_name == self.tools[2].name` 分支，参考 [`tool_env_retro.py:678-683`](Agent-r1-temp/agent_r1/tool/tool_env_retro.py#L678-L683)
     - 在 `wrap_tool_args` 或 execute dispatch 里加 **back_flag 门控**：`self.back_flag=False` 时调 back_state → 返回门控 error
     - 在 `_update_state_variables_message`（或等效 dispatch）中加 back_state 成功分支，参考 [`tool_env_retro.py:876-915`](Agent-r1-temp/agent_r1/tool/tool_env_retro.py#L876-L915)，末尾记得 `self.back_flag = False`
     - 保留 maxstep 检查时 `self.back_flag = True` 的设置（已存在）
     - 移除/不添加"合成成功时 back_flag=True"的逻辑
     - 更新 env.fork / `__init__` 方法里的自引用（原本是 `ToolEnvRetroNoBack(...)`，改为 `ToolEnvRetroHybrid(...)`）

2. **`agent_r1/src/main_agent_retro_hybrid.py`**
   - 起点：复制 `main_agent_retro_noback.py`
   - 改 env 构造：`ToolEnvRetroHybrid(tools=..., max_turns=..., maxstep=..., topk=..., shuffle=..., force_noloop=...)`

3. **（条件性）`agent_r1/llm_agent/generation_retro_hybrid.py`**
   - **先尝试直接复用 `generation_retro.py`**（full-back 版本，已经处理 back_state 工具），如果能跑通就不新建
   - 有 bug 再 fork

4. **（条件性）`agent_r1/src/agent_ray_trainer_retro_hybrid.py`**
   - 同上：先复用 `agent_ray_trainer_retro.py` 或 `agent_ray_trainer_retro_noback.py`，必要时再 fork
   - 入口 `main_agent_retro_hybrid.py` 里 import 哪个 trainer 要对齐

### 5.2 修改文件

1. **`agent_r1/tool/tools/__init__.py`** — [`_default_tools`](Agent-r1-temp/agent_r1/tool/tools/__init__.py#L23) 中新增 branch：

   ```python
   elif env == 'retro_hybrid_V4':
       return [
           SingleStepRetroTool(mlp_model_dump='./one_step_model/retro_star_V4.ckpt'),
           SelectReactionTool(),
           BackStateTool(),
       ]
   ```
   注意 ckpt 路径与 `retro_noback_V4` 对齐。

2. **`run_serve_and_eval.sh`**（[第 67-85 行的 BACKTRACK 分支](Agent-r1-temp/run_serve_and_eval.sh#L67-L85)）新增 `hybrid` 分支：

   ```bash
   case "$BACKTRACK" in
       true|True|TRUE|1)
           MAIN_MODULE="agent_r1.src.main_agent_retro"
           TOOL_ENV_NAME="retro"
           BACK_TAG="back"
           ;;
       hybrid|Hybrid|HYBRID)
           MAIN_MODULE="agent_r1.src.main_agent_retro_hybrid"
           TOOL_ENV_NAME="retro_hybrid_V4"
           BACK_TAG="hybrid"
           ;;
       false|False|FALSE|0)
           MAIN_MODULE="agent_r1.src.main_agent_retro_noback"
           TOOL_ENV_NAME="retro_noback_V4"
           BACK_TAG="noback"
           ;;
       ...
   esac
   ```

3. **`run_eval_rjob.sh`** / **`training_script.sh`**：按需同步上述分支（如果你要跑训练 / 远程 eval）。

### 5.3 配置

- 不需要新 yaml；沿用 `tool.maxstep=30`, `tool.topk=10`, `tool.force_noloop=True` 等默认值
- `tool.env=retro_hybrid_V4` 由脚本传入即可

---

## 六、关键实现细节（容易漏的边界条件）

1. **`back_flag` 生命周期**：
   - 初始化（和每个新样本开始时）：`False`
   - hit `maxstep` 时：`True`
   - `back_state` 成功执行：`False`
   - **不要在其他任何地方改它**，否则门控语义会失效

2. **`back_state` 目标 state 的校验**必须保留（参考 [`tool_env_retro.py:884-888`](Agent-r1-temp/agent_r1/tool/tool_env_retro.py#L884-L888)）：
   - 目标 state 必须在历史中（`unsolved_dict` 或 `reaction_dict` 里有）
   - 目标 state 的 `step_dict[target] < maxstep`（否则回过去也立刻触发 Phase B，死循环）
   - 目标 state 如果是 molecule state，`unsolved_list` 不能为空

3. **Trajectory 终止条件**（与 noback 对齐）：
   - 所有 state 的 unsolved 都被解决 → 结束
   - 超 `max_turns` → 结束
   - 模型连续输出无效工具调用（按现有 `_postprocess_responses` 逻辑）
   - **不要**因为 `back_flag=True` 就强制结束（那会退化成 noback 的截停行为）

4. **Context 是 append-only**：回溯之后历史 token 不会被删——这是设计意图，模型能看到自己走过的弯路，避免重复探索。但要留意 `data.max_prompt_length=32768` 的上限，深路径 + 多次回溯可能撑满 context（这一点在 PR 描述里记录即可，不在本任务范围内解决）。

5. **消息模板复用**：`StepFail` / `FailBack` / `GoBackMolecule` / `GoBackReaction` / `ToCall` / `ToSelect` 等常量在 `tool_env_retro.py` 和 `tool_env_retro_noback.py` 里定义基本一致，直接复制过来即可，不必跨文件 import。

---

## 七、验收标准

提交前请手动验证：

1. **Import 能过**：
   ```bash
   python -c "from agent_r1.tool.tool_env_retro_hybrid import ToolEnvRetroHybrid; \
              from agent_r1.src.main_agent_retro_hybrid import *"
   ```

2. **短 maxstep 触发 hybrid 逻辑**：用 `tool.maxstep=3` 跑一个小 val 集，grep log 应能看到：
   - `Pathway fails because the maximum number of reaction steps 3 has been reached`
   - `back_state` 的实际 tool call
   - `GoBackMolecule` / `GoBackReaction` 回溯成功消息
   - 回溯后**继续出现** `single_step_retro` / `select_reaction` 调用

3. **Phase A 拦截 back_state**：log 里应能观察到（偶发）模型在 Phase A 调 back_state 时 env 返回了门控 error，且 trajectory 没崩。

4. **深度对比**（用同一组 val 数据、同一个 base model）：
   - `retro_hybrid_V4` 的 **平均 reaction step 深度**（"at least N reaction steps are used" 的分布）应该 **≥ `retro_noback_V4`**
   - `retro_hybrid_V4` 的平均深度应该 **明显 > 原始 `retro`**（这是本任务最核心的成功信号）

5. **跑完 190 样本不挂**：`BACKTRACK=hybrid bash run_serve_and_eval.sh` 能完整跑完 retro 验证集。

---

## 八、参考文件速查表

| 用途 | 文件 | 关键行 |
|---|---|---|
| 起点 env（复制这个） | `agent_r1/tool/tool_env_retro_noback.py` | 373-385（ctor）、896-899（maxstep 检查）、1034（env fork）|
| back_state 处理参考 | `agent_r1/tool/tool_env_retro.py` | 601-634（`_process_action_back`）、678-683（`wrap_tool_args` 分支）、876-923（执行分支）|
| 工具注册 | `agent_r1/tool/tools/__init__.py` | 23-39 |
| 入口参考（noback） | `agent_r1/src/main_agent_retro_noback.py` | 121 |
| 入口参考（back） | `agent_r1/src/main_agent_retro.py` | — |
| 脚本分支 | `run_serve_and_eval.sh` | 67-85 |
| Back tool 本身（几乎空壳） | `agent_r1/tool/tools/back_state_tool.py` | — |
| 消息模板常量 | `tool_env_retro_noback.py` / `tool_env_retro.py` | 326-333 |

# 数据接口与复现

数据正文不进入 Git。`configs/data_manifest.json` 固定保留的 task ID、task descriptor、SFT 源 ID 和每条消息 SHA-256；它是数据选择清单，不包含模型指标或 checkpoint 信息。

## 来源与处理

- [AgentGym/AgentTraj-L](https://huggingface.co/datasets/AgentGym/AgentTraj-L)，revision `cb518f5359f0b635a6b8c0c5cbf0469939a927c7`：原始多轮示范。
- [AgentGym/AgentGym-RL-Data-ID](https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID)：原项目训练/测试任务范围；实际实验划分由本仓库 manifest 冻结，不依赖下载时重新随机划分。
- ALFWorld 原生 JSON/PDDL/compiled tw-pddl：使用下载脚本内固定 SHA-256。目录名 `alfworld-0.3.3` 是兼容项目目录命名，不表示所有文件来自同一版本号；运行时代码和资产分别固定。
- WebShop：AgentGym commit 内包含的 1,000 商品数据。使用 seed 233 生成固定价格、synthetic goals 并 shuffle，构建 Lucene 1k 索引。

划分先隔离 test ID，再按 task family 和相同示范分组生成 train/dev。SFT 清理包含格式、重复及长度条件；ALFWorld 同时检查给定可用动作列表与回合数。保留示范只替换统一任务提示词，保留观察与动作正文。离线格式审查不等于全部示范都经原生环境成功验证。

`download` 使用已冻结清单复现这个处理结果，而非重跑一次可能受依赖版本影响的历史筛选。它从公开源恢复消息、替换首轮固定 prompt/acknowledgement，并逐条比对原处理结果的消息哈希；不一致会停止。已下载原始 JSON 时可加 `--raw-dir /path/to/AgentTraj-L`，仍执行同样的哈希验证。

SFT 与 RL train 可以覆盖相同任务；不能把这类重叠当成数据泄漏。train/dev/test 的 task ID 互斥；train/dev 任务族互斥。当前冻结 test 并未保证与 train 在所有潜在任务族定义上互斥；不能宣称本划分是严格的 unseen-type 测试。

## 处理后布局

```text
data/
  processed/
    alfworld/{sft_train,sft_dev,rl_train,rl_dev,rl_test}.jsonl
    webshop/{sft_train,sft_dev,rl_train,rl_dev,rl_test}.jsonl
  assets/{webshop_catalog,webshop_product_prices}.json
  environments/alfworld-0.3.3/json_2.1.1/...
```

每行一个 JSON 对象，UTF-8。`item_id` 在同一 split 中必须唯一，且须是文件名安全的标识符。

### SFT schema

```json
{
  "item_id": "example_task",
  "messages": [
    {"role":"user", "content":"固定 benchmark prompt", "loss":false},
    {"role":"assistant", "content":"固定 acknowledgement", "loss":false},
    {"role":"user", "content":"环境初始观察", "loss":false},
    {"role":"assistant", "content":"Thought: Inspect.\nAction: look", "loss":true}
  ]
}
```

后续为 user 观察 / assistant 响应交替。每个实际响应必须 `loss=true`；首轮 acknowledgement 不计算 loss；结束在 assistant。WebShop 格式必须含 Thought 与 `Action: search[...]` / `click[...]`；ALFWorld 允许仅 Action。角色、损失掩码、格式与非空内容在编码前严格校验。

编码由 `training/protocol.py` 控制，Qwen3 非 thinking 前缀固定为：

```text
<|im_start|>assistant
<think>

</think>

```

仅监督响应正文及 `<|im_end|>`。所有 turn 的生成前缀必须与训练序列的同位置 token 完全一致；超出 4096 直接报出任务，不静默截断。teacher/student 共用输入协议。

### ALFWorld RL schema

```json
{
  "item_id":"unique_task_id",
  "environment_index":0,
  "game_file":"data/environments/alfworld-0.3.3/json_2.1.1/train/.../game.tw-pddl",
  "task_type":"pick_and_place_simple",
  "task_spec":"task_family",
  "native_split":"train"
}
```

实际 reset 按 `game_file`，不会假定不同版本环境的数字索引一致。相对路径以仓库根目录为基准。奖励/成功由原生 `state.won` 决定。

### WebShop RL schema

```json
{
  "item_id":"webshop_0",
  "environment_index":0,
  "category":"example",
  "instruction_text":"任务指令",
  "goal_family_key":"冻结的任务族标识"
}
```

公开重建会加入 catalog 内的商品、属性、预算、选项等字段。环境按 `environment_index` reset，启动时将实际 goal 列表与 catalog 全量比对。原生终局 reward ∈ [0,1]；success 为 reward ≥ 1。不要混用不同商品库/catalog/价格表。

### 自定义数据

提供上述 schema 的 JSONL，通过配置的 `sft_data`、`train_data`、`dev_data` 接入。`--tasks` 可覆盖独立评测文件。当前 sampler 固定 4 条/GRPO 组，训练 split 至少应能容纳一个 update 的不同任务数；不支持把少量任务复制 ID 来凑批。

运行 `scripts/prepare_data.py validate --directory YOUR_DATA_ROOT` 检查 ID、掩码、格式与重复。自定义数据不会自动具有仓库 manifest 的划分保证，需自行检查语义重复、任务族与来源。

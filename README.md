# Teacher-Guided Token Advantage GRPO

面向 **ALFWorld / WebShop** 的单卡 agent 训练代码：Qwen3-4B 教师 SFT → GRPO、Qwen3-1.7B 学生 SFT 冷启动，以及学生 GRPO、OPD、OPD+GRPO、teacher top-K token-level advantage + GRPO。

## 方法与代码

| 训练方式 | 入口/核心实现 | 监督来源 |
|---|---|---|
| 教师/学生 SFT | `scripts/train_sft.py`、`training/engine.py` | 示范中带 `loss: true` 的 assistant 输出，含 EOS |
| GRPO | `scripts/train_rl.py`、`training/runtime.py:grpo_update` | 原生环境终局奖励，4 条轨迹组内标准化 |
| OPD | 同上，`opd_update` | 学生前缀下教师 log-prob，sampled reverse-KL 梯度 |
| OPD+GRPO | 同上，`hybrid_update` | GRPO + `opd_weight × OPD` + 固定参考策略 KL |
| Token advantage + GRPO | `training/objectives.py`、`tokenadv_update` | 轨迹优势 + `token_lambda ×` 教师概率奖励优势 |

vLLM 批量展开，随后 sleep/offload；Transformers 依次计算冻结教师/参考策略信号及学生梯度。每批只更新一次，再将新 LoRA 载入 vLLM。保留完整观察历史，不截断到任意最近几轮；超过长度预算时结束当前轨迹。

```text
configs/        两个 benchmark 的 SFT、教师 GRPO、四种学生方法配置与数据划分清单
scripts/        安装、下载、数据准备、训练、评测、环境自检
training/       共享序列化、原生环境客户端、采样 RPC、损失、续训
patches/        WebShop 商品选项奖励补丁
requirements/   训练 / vLLM / 原生环境三套依赖
tests/         数值、梯度、轨迹对齐和配置测试
```

## 环境准备

目标平台为 Linux x86_64、支持 BF16 的 NVIDIA GPU、Python 3.10。训练与推理解耦到不同 Python 环境，WebShop 使用 Java 11。建议至少 32GB 显存、64GB 主存，48GB 显存更便于扩大 micro batch；可用容量由预检决定，不能只按参数量判断。首次下载需联网；准备完成后训练不调用远程模型 API。

在仓库根目录执行：

```bash
# Debian/Ubuntu；如已安装可跳过。其他发行版安装等价软件。
sudo apt-get update
sudo apt-get install -y python3.10-venv build-essential git curl openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

bash scripts/setup_python.sh
bash scripts/setup_benchmarks.sh
envs/train/bin/python scripts/download_models.py
```

`JAVA_HOME` 应指向实际 JDK 11；Conda JDK 用户可指向对应环境根目录。已具备环境时可以不重装，在配置中更改 `train_python`、`inference_python`、`bench_python`。具体版本、资产和离线迁移见 [环境说明](docs/ENVIRONMENT.md)。

## 数据准备

公开示范来自 [AgentGym/AgentTraj-L](https://huggingface.co/datasets/AgentGym/AgentTraj-L)，固定 revision；RL 任务基于 [AgentGym/AgentGym-RL-Data-ID](https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID) 的任务范围及仓库内冻结划分。WebShop 使用 1,000 商品版本与 synthetic goals；不是完整商品库或 human-goal 版本。ALFWorld 使用 TextWorld/PDDL。

| Benchmark | SFT train | SFT dev | RL train | RL dev | RL test |
|---|---:|---:|---:|---:|---:|
| ALFWorld | 2,104 | 242 | 2,176 | 244 | 200 |
| WebShop | 3,530 | 392 | 3,537 | 393 | 200 |

SFT dev 仅保留为数据接口；训练里的在线评测统一使用 **RL dev**。这里的 test 是项目冻结任务划分，不能等同于 ALFWorld 官方 unseen split。SFT/RL train 共享任务是预期设计；train/dev/test ID 互斥，train/dev 按任务族分组。更多细节见 [数据接口](docs/DATA.md)。

```bash
# setup_benchmarks.sh 已生成 WebShop catalog 和固定价格表。
envs/train/bin/python scripts/prepare_data.py download
envs/train/bin/python scripts/prepare_data.py validate

# 环境检查不加载模型、不使用 GPU。
envs/train/bin/python scripts/check_environment.py --benchmark alfworld
envs/train/bin/python scripts/check_environment.py --benchmark webshop
```

下载器使用 `configs/data_manifest.json` 重建划分，逐条验证 SFT 消息 SHA-256。文件中只有任务/划分元信息与校验值，不包含示范文本、模型权重或训练产物。也可以从现有项目导入：

```bash
envs/train/bin/python scripts/prepare_data.py import-project \
  --source-project /absolute/path/to/existing/project
```

导入命令仅复制两个 benchmark 的处理后数据和 WebShop catalog/价格表；环境资源、商品索引与模型应另行安装或挂载。

## 模型接口

基础模型使用 Hugging Face 本地目录，包含 config、tokenizer 和未量化 safetensors；默认 `models/Qwen3-4B`、`models/Qwen3-1.7B`。下载命令记录解析后的模型 commit，可通过 `--revision COMMIT` 固定版本。

adapter 使用 PEFT 目录，至少包含 `adapter_config.json` 与 `adapter_model.safetensors`。当前实现限定同词表 Qwen3、LoRA **r=16、alpha=32**。SFT dropout=0.05，在线训练与采样 dropout=0。全程 BF16 基座 + FP32 可训练 LoRA；不使用 QLoRA。

每个 benchmark 的教师、学生分别训练。以下路径变量由使用者设置，命令没有隐含或预选的历史 checkpoint：

```bash
export TEACHER_SFT_ADAPTER=/absolute/path/to/teacher_sft_adapter
export TEACHER_GRPO_ADAPTER=/absolute/path/to/teacher_grpo_adapter
export STUDENT_SFT_ADAPTER=/absolute/path/to/student_sft_adapter
```

## 教师和学生 SFT

```bash
# 教师 4B；将 alfworld 改为 webshop 即为另一任务。
envs/train/bin/python scripts/train_sft.py \
  --config configs/alfworld_teacher_sft.json --run runs/alfworld_teacher_sft

# 学生 1.7B 冷启动。
envs/train/bin/python scripts/train_sft.py \
  --config configs/alfworld_student_sft.json --run runs/alfworld_student_sft
```

默认 3 epochs，在 0.25、0.5、1、2、3 epoch 后保存完整续训 checkpoint，释放训练进程后执行完整 RL dev greedy。边界向上取整到 optimizer step；跨边界保持同一 cosine schedule 和 optimizer 状态，不重置 warmup。

| 配置 | 默认值 |
|---|---|
| 学习率 / 调度 | `1e-4` / cosine / warmup ratio `0.05` |
| micro batch × 梯度累积 | `2 × 16`，有效 batch 32 条轨迹 |
| 最大序列长度 | 4096，不静默截断 |
| 优化器 / 梯度裁剪 | fused AdamW / 1.0 |
| gradient checkpointing | 开启，non-reentrant |
| 训练目标 | 仅标记的 assistant token，按累积窗口监督 token 总数归一化 |

可用 `--set learning_rate=5e-5` 等覆盖现有配置项。参数调整需要新 `--run` 目录；同一目录重跑会检查配置并继续已完成阶段之后的训练/评测。

## 教师 GRPO

```bash
envs/train/bin/python scripts/train_rl.py both \
  --config configs/alfworld_teacher_grpo.json --run runs/alfworld_teacher_grpo \
  --set "parent_checkpoint=$TEACHER_SFT_ADAPTER"
```

教师 GRPO 使用 4B 基座、SFT adapter 起点、固定 SFT 参考策略、学习率 `5e-6`。WebShop 对应 `configs/webshop_teacher_grpo.json`。

## 学生在线训练

四种方法分别使用独立运行目录；请为当前 benchmark 设置对应模型路径。

```bash
# 纯 GRPO：不需要教师 adapter。
envs/train/bin/python scripts/train_rl.py both \
  --config configs/alfworld_grpo.json --run runs/alfworld_grpo \
  --set "parent_checkpoint=$STUDENT_SFT_ADAPTER"

# 纯 OPD。
envs/train/bin/python scripts/train_rl.py both \
  --config configs/alfworld_opd.json --run runs/alfworld_opd \
  --set "parent_checkpoint=$STUDENT_SFT_ADAPTER" \
  --set "teacher_checkpoint=$TEACHER_GRPO_ADAPTER"

# 常规 OPD + GRPO；此处权重乘在 OPD loss 上。
envs/train/bin/python scripts/train_rl.py both \
  --config configs/alfworld_hybrid.json --run runs/alfworld_hybrid \
  --set "parent_checkpoint=$STUDENT_SFT_ADAPTER" \
  --set "teacher_checkpoint=$TEACHER_GRPO_ADAPTER" --set opd_weight=0.5

# Token-level advantage + GRPO；lambda 乘在局部优势上。
envs/train/bin/python scripts/train_rl.py both \
  --config configs/alfworld_tokenadv.json --run runs/alfworld_tokenadv_k64_l05 \
  --set "parent_checkpoint=$STUDENT_SFT_ADAPTER" \
  --set "teacher_checkpoint=$TEACHER_GRPO_ADAPTER" \
  --set teacher_top_k=64 --set token_lambda=0.5
```

WebShop 替换配置文件与运行目录中的 `alfworld` 为 `webshop`。K/λ 扫描示例（串行，避免单 GPU 进程争抢）：

```bash
for k in 16 64; do
  for lam in 0.5 1.0; do
    envs/train/bin/python scripts/train_rl.py both \
      --config configs/webshop_tokenadv.json --run "runs/webshop_tokenadv_k${k}_l${lam}" \
      --set "parent_checkpoint=$STUDENT_SFT_ADAPTER" \
      --set "teacher_checkpoint=$TEACHER_GRPO_ADAPTER" \
      --set "teacher_top_k=$k" --set "token_lambda=$lam"
  done
done
```

`both` 先预检再训练。也可分别使用 `preflight` 和 `train`；正式训练拒绝缺失、失败或与当前代码/数据/配置不匹配的预检记录。预检包含真实环境采样、HF/vLLM 概率对齐、一次更新、满 4096 token 压力测试及 LoRA 更新/恢复检查。

| 参数 | 学生 GRPO / hybrid / tokenadv | 纯 OPD |
|---|---|---|
| 更新数 | 100 | 100 |
| 学习率 | `2e-6` | `5e-6` |
| 每次更新轨迹数 | 8 任务 × 4 = 32 | 64 任务 × 1 = 64 |
| micro batch / scoring batch | 2 / 2 | 2 / 2 |
| 采样 T / top-p / top-k | 1 / 1 / -1 | 1 / 1 / -1 |
| PPO clip / 固定参考 KL β | 0.2 / 0.01 | 不使用 |
| warmup | 5 更新，随后恒定 LR | 同左 |
| vLLM 并发 / 批 token 预算 | 32 / 8192 | 32 / 8192 |
| 训练打分 padded token 预算 | 16384 | 16384 |
| dev greedy | 0、5、10、25、50、75、100 更新 | 同左 |

ALFWorld 最多 30 轮、WebShop 最多 10 轮，每轮最多输出 256 token，总上下文 4096。默认保留每一步 adapter + optimizer/RNG 状态，便于精确续训；请按存储容量自行归档。重跑同一命令从最新完整 checkpoint 继续，评测中断可重新执行，未完成的更新不会被标记为已完成。

默认关闭前缀缓存、开启 CUDA Graph。增大并发或 micro batch 时以新运行目录重新预检，例如 `--set max_concurrent_trajectories=64 --set training_micro_batch=4`；它们是容量选项，不改变每次更新的轨迹总数。预检失败时不会启动正式训练。

## 目标函数与注意事项

令 `pT` 为教师分布、`pS_old` 为更新前学生分布，`S_K` 为教师 top-K 支持集。所有打分使用温度 1：

```text
r_t(v) = pT(v | h_t) · 1[v ∈ S_K]
b_t    = Σ(v ∈ S_K) pS_old(v | h_t) · pT(v | h_t)
a_t    = r_t(y_t) - b_t
A_t    = A_group + λ · a_t
```

教师 top-K 概率来自**完整词表 softmax**，不在 top-K 内重新归一化。教师奖励、学生基线、局部优势均在参数更新前冻结；先把优势相加，再进行一次 PPO clipping。λ=0 时退化为相同轨迹集合上的 GRPO 目标。局部教师信号不保证每个正确步骤的最终优势为正。

常规 OPD 使用 `-(log pT(y)-stopgrad(log pS(y))) × log pS(y)` 的 sampled reverse-KL surrogate，按整批生成 token 平均。hybrid 则是该项乘 `opd_weight` 后，加到 GRPO 与固定参考 KL 上。它与概率奖励 token advantage 是不同方法。

纯 GRPO 默认跳过奖励完全相同的组，最多尝试 256 组/更新；耗尽预算则报错，不做部分更新。OPD 使用全部 on-policy 轨迹；hybrid/tokenadv 保留零方差组，组优势设为 0，教师信号仍参与更新。格式错误、循环与失败会记录，不作为 OPD 成功轨迹筛选条件。

采样固定为未截断 T=1，以匹配该实现的概率与期望基线；不能只把 temperature 改成 0.7 而仍宣称使用同一无修正估计器。评测的随机 pass@1/pass@5 可用 T=0.7、top-p=0.9。

## 独立评测与检查

```bash
# --adapter 指向你要测的 adapter；默认 RL dev。
envs/inference/bin/python scripts/evaluate.py \
  --config configs/webshop_grpo.json --adapter "$STUDENT_SFT_ADAPTER" \
  --output outputs/webshop_dev --mode greedy

# test 只通过显式参数接入；训练过程不读取 test 选择 checkpoint。
envs/inference/bin/python scripts/evaluate.py \
  --config configs/webshop_grpo.json --adapter "$STUDENT_SFT_ADAPTER" \
  --tasks data/processed/webshop/rl_test.jsonl --output outputs/webshop_test --mode greedy

# pass5 为每任务最多 5 次、成功即停；同时记录首次尝试成功率。
envs/inference/bin/python scripts/evaluate.py \
  --config configs/alfworld_grpo.json --adapter "$STUDENT_SFT_ADAPTER" \
  --output outputs/alfworld_pass5 --mode pass5

CUDA_VISIBLE_DEVICES="" envs/train/bin/python -m unittest discover -s tests -v
```

评测目录写入 `summary.json`、请求清单、逐任务轨迹与进度；训练目录写入 `status.json`、`metrics.jsonl`、采样批次和完整续训状态。成功定义是 ALFWorld `won`、WebShop reward ≥ 1；格式遵循率是所有实际输出回合中通过格式检查的比例，不能替代任务成功率。

本次打包检查的范围与限制见 [验证记录](docs/VALIDATION.md)。上游代码、提示词和数据归属见 [第三方说明](THIRD_PARTY_NOTICES.md)。

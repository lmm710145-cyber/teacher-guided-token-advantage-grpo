# 环境与迁移

## 分离运行时

| 环境 | Python | 关键依赖 | 用途 |
|---|---|---|---|
| `envs/train` | 3.10 | torch 2.8.0、Transformers 4.57.6、PEFT 0.17.1 | SFT / teacher scoring / gradient update |
| `envs/inference` | 3.10 | vLLM 0.10.2、Transformers 4.57.6 | rollout / online evaluation |
| `envs/bench` | 3.10 | TextWorld 1.7.0、torch 2.2.2 CPU、spaCy 3.3.0、Pyserini 0.17.0 | 原生 PDDL / WebShop |

不把三套依赖安装到同一个 Python 环境。WebShop 旧依赖需要 NumPy 1.26.4、Werkzeug <3、Java 11；vLLM 的 PyTorch/CUDA 依赖由其固定版本解析。Linux NVIDIA 驱动应兼容所装 PyTorch CUDA wheel。安装后可将 `pip freeze` 与运行目录一起归档。

`setup_python.sh` 创建 venv；WebShop 依赖也可以用 Conda JDK 11，设置 `JAVA_HOME` 与 `PATH`。如果出现 `Unable to find javac`，先确认：

```bash
"$JAVA_HOME/bin/java" -version
"$JAVA_HOME/bin/javac" -version
```

## 固定上游与补丁

- [AgentGym](https://github.com/WooooDyy/AgentGym)：`3ef9235d23e68e7c2920c5422ad957dc8ced5c6c`。
- [ALFWorld](https://github.com/alfworld/alfworld)：`aaba6870f86c5be6a08a491f32a50b906227bc3e`。
- ALFWorld 资产分别来自 [0.2.2 release](https://github.com/alfworld/alfworld/releases/tag/0.2.2) 的 JSON/PDDL/tw-pddl；下载脚本对三份归档做 SHA-256 验证，不使用最新版下载器替换成另一套 compiled games。
- `patches/webshop_option_reward.patch`：将 goal option 字典传入 reward 匹配时取 `.values()`，避免把 `(key,value)` 元组当作字符串比较。该补丁会影响 reward 定义，必须与实验一起披露；本仓库默认始终使用此补丁。
- WebShop 固定价格表由 seed 233 生成；环境每次加载时显式替换原生价格生成函数为该表，避免其他模块随机数使用改变任务。

安装脚本只应用上述 WebShop 补丁，不导入原服务器其他 benchmark 修改。动作解析函数从固定 AgentGym 源码读取，避免引入其完整 controller 依赖；格式审查使用独立严格规则，解析成功和格式合规是两个指标。

## 运行时细节

默认 `enable_prefix_caching=false`。此固定 vLLM/LoRA 组合的适配器切换需要概率对齐预检，不通过检查不得直接启用缓存。保留 CUDA Graph；预检允许经过 Transformers 同前缀概率复核的 BF16 近似平局 argmax 差异，但不容忍明显的分布错误。

ALFWorld PDDL 的 planner 库可能在反复 reset 后仍保留已删除文件映射；sampler 在累计 reset 数将超过 64 时、两批完整轨迹之间重建环境子进程，释放映射。不在任务进行中重启环境。

只有生成 token 参与 online loss，包括 Thought、Action、EOS；观察 token 不监督。原始 token 前缀不能通过重新 decode/encode 近似替代；无法合并到最终历史的 turn 单独保留为训练 segment，并仍以整条轨迹 token 数归一化。

每个 `run` 有进程锁。online checkpoint 原子提交，包含 optimizer、PyTorch CPU/CUDA RNG、Python RNG、任务 cursor 和配置指纹；不承诺不同 GPU/CUDA 内核之间 bitwise 重现。vLLM 采样种子来自任务/attempt/turn/update，dev 不随 update 改种子。

## 迁移到另一台机器

最小代码仓库不包含以下大文件：

- 基础模型：用 `scripts/download_models.py` 重新获取，或挂载已有目录。
- 公开 SFT 正文与处理后的任务 JSONL：用 `scripts/prepare_data.py download` 重建。
- 环境资产与 WebShop 索引：用 `scripts/setup_benchmarks.sh` 下载/重建。
- Python 环境：按 requirements 安装；不用复制原服务器绝对路径 venv。
- 训练产物：新训练不需要；若要续训，单独传输运行目录和其使用的起点/教师 adapter。

续训不是只复制 adapter：还需 `training_state.pt`、`state.json`、`complete.json`、原配置、预检记录、相关采样批次和起点 adapter。指纹包含解析后的绝对路径；改变运行根路径后的恢复不会自动绕过指纹检查。精确跨路径迁移需显式核查并迁移清单；最简单的可验证办法是在新机使用相同目录布局。用已有 adapter 开启一个新 run 则不恢复旧 optimizer，属于重新初始化优化器的续接实验。

不要通过编辑 `preflight.json` 将失败改为通过。降低 micro batch、并发、vLLM memory utilization 后用新目录预检。默认 CPU RAM 会同时保留学生、固定参考、教师与 sleeping vLLM 的 CPU 副本，主存也要预留。

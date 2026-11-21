# 打包验证记录

本记录描述代码整理后的工程检查，不包含模型实验成绩或历史 checkpoint 选择。

## 已完成

- Python 编译检查、未定义变量静态检查、shell 语法检查通过。
- 14 份训练配置可解析；模型/解释器/数据路径转换与 K、λ 覆盖通过检查。
- 对外 CLI 的 `--help` 可在未加载模型时运行。
- **19 项 CPU 单元/集成测试通过**：稀疏 top-K 与完整词表结果一致；top-K 不重新归一化；采样 token 不在 top-K 时奖励为 0；局部优势期望为 0；基线/教师信号停止梯度；λ=0 与 GRPO loss/gradient 对齐；先合并优势再 clipping；trajectory token 归一化；原始前缀分段无丢失/重复；OPD surrogate 与 reverse-KL 梯度一致；近似平局容差拒绝明显差异；各方法零方差组处理；小型 Qwen3 的分块词表投影 loss/gradient 与 dense 前向一致；SFT 分段保存和完整 optimizer/scheduler/RNG 续训可保持训练总步数与调度。
- 从现有固定版本公开源快照重建两套数据，全部保留 SFT 消息逐条匹配 manifest 哈希；RL split ID/格式/重复检查通过。
- 对全部 **2,104 + 3,530 条 SFT train** 使用 Qwen3-1.7B tokenizer 编码，所有监督位置与生成前缀逐条对齐，均在 4096 长度内。
- 在独立临时代码目录调用真实原生环境：ALFWorld、WebShop 各检查两项 reset + step；WebShop 启动时全部 catalog instruction 与实际环境 goals 一致。
- 代码及文档扫描未发现原服务器地址、登录凭据、API key 或历史实验 checkpoint 路径。

CPU 数值/模型测试运行于现有 Linux Python 3.12、PyTorch 2.8.0、Transformers 4.57.6、PEFT 0.17.1、Accelerate 1.15.0；原生环境使用独立 Python 3.10。测试进程均设置 `CUDA_VISIBLE_DEVICES=""`。

## 尚未执行的验证

此次整理未进行全新机器从零联网安装测试，也未启动重构后仓库的完整 GPU 训练，以免占用现有实验的 GPU。已安装环境中的 CPU 检查不能代替新机器 CUDA、vLLM 内核和显存预检。

正式在线训练入口默认先运行真实任务采样、HF/vLLM log-prob 检查、一次梯度更新、满上下文压力和 adapter 热切换/恢复检查。通过结果绑定代码、数据、配置和起点 adapter；失败不会进入正式训练。

WebShop 检查需要设置实际 JDK 11 的 `JAVA_HOME`。当前代码没有隐藏原服务器的 JDK 路径；新机器应按环境文档设置。

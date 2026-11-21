# 第三方来源

本仓库从项目现有训练实现整理；新增部分提供可配置入口、公开数据重建、依赖安装和测试。仓库没有代作者决定新的整体开源许可证。发布时可由权利人选择项目许可证；下列上游条款独立适用。

- AgentGym：任务 prompt、acknowledgement、动作解析协议，以及外部 WebShop runtime；固定 commit 见环境文档。MIT 版权与许可全文保留于 `licenses/AgentGym-MIT.txt`。环境代码通过 setup 脚本获取，未将其完整复制进本仓库。
- ALFWorld / TextWorld / WebShop：各自源码和任务资产仍遵循上游许可证及数据使用条件；安装时保留其原仓库许可。
- AgentGym/AgentTraj-L、AgentGym/AgentGym-RL-Data-ID：数据来源见 `docs/DATA.md`。Git 内仅保存处理/划分 manifest，不包含示范正文和商品资源。
- Qwen3、Transformers、PEFT、vLLM、PyTorch：运行依赖与模型分别遵循其上游许可；本仓库不重新发布模型权重。

参考入口：

- https://github.com/WooooDyy/AgentGym
- https://github.com/alfworld/alfworld
- https://github.com/princeton-nlp/WebShop
- https://huggingface.co/datasets/AgentGym/AgentTraj-L
- https://huggingface.co/Qwen/Qwen3-4B
- https://huggingface.co/Qwen/Qwen3-1.7B

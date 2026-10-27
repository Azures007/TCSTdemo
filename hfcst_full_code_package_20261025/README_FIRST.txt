HFCST 完整源码包（基于本次审计包整理）

内容：
- core/：模型、数据处理、光谱变换、传输求解、风险控制与训练器源码
- scripts/：训练入口与三数据集 source-only 学习率对照脚本
- configs_hfcst/：三个数据集的现有配置和对照实验配置
- AUDIT_REPORT.md：审计说明
- CODE_CHANGES.patch：本次代码差异
- core/trainer_hfcst.py.bak_metricfix_20261025_111515：随原审计包提供的训练器备份

注意：此包不包含原始遥感数据、训练结果、模型权重或 Python/CUDA 环境。
请先备份服务器上现有的 hfcst_work 目录，不要直接覆盖唯一副本。

服务器项目目录（根据用户此前提供的终端路径）：
~/autodl-tmp/TCSTdemo/hfcst_work

如果仅检查代码，先运行：
python -m py_compile core/*.py scripts/train_hfcst.py

运行实验前请先设置 DATA_ROOT，并检查 configs_hfcst/ 中数据路径与本机目录一致。

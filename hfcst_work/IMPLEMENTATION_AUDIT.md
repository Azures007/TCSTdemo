# HFCST-v9 实现审计与验证记录

## 1. 保护原版的方式

- 新增 `core/*_hfcst.py`、`scripts/train_hfcst.py`、`scripts/predict_hfcst.py`、`configs_hfcst/*.yaml`、`tests/test_hfcst.py`、本文档与 `README_HFCST.md`。
- 未覆盖原 `core/model.py`、`core/data.py`、`core/trainer_v8.py`、`core/transport_solver.py` 或原训练入口。
- 原版代码中的 QP 入口仍直接依赖 CVXPY；HFCST 新入口使用独立的 `core/transport_solver_hfcst.py`，在没有 CVXPY 时使用 SciPy SLSQP 后备求解器。

## 2. 原问题与新实现位置

| 需求/旧问题 | 新实现位置 | 处理方式与验证 |
|---|---|---|
| 光谱分支监督不足、共享分类头竞争 | `core/model_hfcst.py`、`core/trainer_hfcst.py` | 融合、空间、光谱三头分别计算 CE；合成端到端训练已运行。 |
| 光谱幅值被逐像素 L2 归一化抹去 | `core/data_hfcst.py`、`core/model_hfcst.py` | 保留原始有限值，按源域训练中心像素的绝对值分位数共同缩放；空间分支内部单独做 L2 归一化。 |
| 波段数不同时均匀索引重采样 | `core/data_hfcst.py` | 波段不一致且无显式映射时抛出错误。 |
| 全局 MMD 不能约束类别条件分布 | `core/discrepancy_hfcst.py` | 类别条件光谱描述子 MMD + 融合特征 MMD；软目标权重与可靠性参与每类独立归一化。 |
| 优化 MMD、验收另看 SWD | `core/trainer_hfcst.py`、`core/risk_controller_hfcst.py` | QP 梯度、候选前验收与目标无标签验证均使用同一类别条件传输目标；Bootstrap 以目标空间块为单位。 |
| 没有约束样本时仍可能迁移 | `core/task_geometry_hfcst.py`、`core/transport_solver_hfcst.py` | 空约束、非有限梯度、求解失败、越界或后验不可行时零更新并拒绝。 |
| 类别残差缺乏支持约束 | `core/spectral_transform_hfcst.py`、`core/trainer_hfcst.py` | 依据无标签目标软权重的有效样本数启用残差；无支持类别残差固定为零，并按类别单独求解信赖域 QP。 |
| 候选被拒绝后仍执行纯源域 CE 周期 | `core/trainer_hfcst.py` | 未通过迁移前验收不训练；训练后失败恢复完整快照并停止适应。 |
| 回滚只恢复模型权重 | `capture_transaction_state` / `restore_transaction_state` in `core/trainer_hfcst.py` | 快照包括模型、优化器、EMA 教师、传输参数、调度器和提交周期计数。 |
| 目标域 GT 泄漏到训练或选择 | `run_hfcst()` in `core/trainer_hfcst.py` | 适应阶段只调用 `load_cube` 读取目标影像；最终检查点保存后才调用 `load_gt` 进行一次最终评估。端到端测试在目标 GT 被打开时断言 `final.pt` 已存在。 |
| 训练/约束/校准/验证 Patch 重叠 | `core/data_hfcst.py` | 先按全局空间块分配 split，再仅取完整 Patch 支持区域位于所属块内的中心；另有跨集合中心距离审计。 |
| 没有机器可读诊断 | `core/trainer_hfcst.py` | `events.jsonl`、`data_audit.json`、`results.json` 记录 QP、类别支持、候选原因、Bootstrap、校准风险、提交/回滚和最终指标。 |

## 3. 测试执行结果

### HFCST 新增测试

执行命令：

```bash
pytest -q tests/test_hfcst.py --disable-warnings --maxfail=1
```

实际结果：**16 passed**。覆盖：

1. 恒等变换；
2. DCT 正交性；
3. 残差为零时退化为共享变换；
4. 类别路由；
5. 无支持类别残差为零；
6. 加权 MMD 数值和同分布零差异；
7. 共享/残差传输梯度有限；
8. 硬 Margin 与信赖域 QP；
9. 无有效类别/无约束的安全退回；
10. 无传输改善时拒绝候选；
11. 源域风险超预算时拒绝候选；
12. 模型、优化器、教师、传输参数和调度器完整回滚；
13. 全局空间块 Patch 隔离；
14. 图像加载不需要目标 GT；
15. 合成数据端到端运行，实际经过共享 QP、候选搜索、类别残差、训练后验收、提交、保存与最终评估，并检查目标 GT 仅在 `final.pt` 保存后读取；
16. 无可靠目标类别支持时不执行无效 CE 周期，最终模型与基线模型一致。

### 原版兼容性测试

执行命令：

```bash
pytest -q tests/test_core.py tests/test_v801.py tests/test_v810_p0p1p2.py \
  tests/test_mmd_synthetic.py tests/test_mmd_synthetic_v2.py --disable-warnings
```

实际结果：**17 passed, 3 failed**。三项失败均发生在旧版 `core/transport_solver.py` 导入 `cvxpy` 时，当前执行环境报 `ModuleNotFoundError: No module named 'cvxpy'`。这三项不是已通过；需在完整环境安装 `requirements.txt` 中的 CVXPY 后重新执行。HFCST 新版 QP 在当前环境实际使用 `scipy_slsqp`，新增硬约束测试通过。

## 4. 尚未验证的内容与风险

- 本次上传的归档不包含 Houston13/Houston18、Dioni/Loukia、Shanghai/Hangzhou 的真实 `.mat` 文件，因此三组真实训练与真实 OA/AA/Macro-F1/Kappa 均**未运行**。
- 数据集的 GT 编码按原 TCST 约定处理：背景为 0、前景类别为 1..C。若实际数据编码不同，必须先核对并修改标签映射。
- 光谱波段数相同不等于物理波长完全对应；当前配置仅在波段数相同时默认采用同序映射。若传感器响应不同，需要可信波长元数据或人工审计后的显式映射。
- 空间块隔离会减少可用样本。若某类别在训练、约束、校准或验证区域中样本不足，结果可能不稳定，或系统会安全拒绝适应；不能以重叠 Patch 作为回退。
- 当前 MMD 使用多尺度 RBF 核和中位距离启发式；`K=8`、`rho_a/rho_b=0.10`、损失权重和验收预算都是初始配置，不代表最优参数。
- 目标软类别概率与可靠性是无标签估计，不保证伪类别正确。训练日志必须检查类别有效样本数、预测分布与可靠性；不应只看最终 OA。
- QP 的局部线性 Margin 约束不能替代非线性源域校准风险检查；实现中两者均保留。真实运行仍应审查求解器状态和每个候选的拒绝原因。
- Bootstrap 的可信度取决于空间块数量和代表性。独立目标验收块不足时，代码会因 Bootstrap 不可用拒绝候选，而不是将不确定性视为成功。

## 5. 真实数据复现前检查清单

- [ ] 安装 `requirements.txt` 并复测原版兼容性测试；
- [ ] 设置 `DATA_ROOT`，检查三个 YAML 中所有影像与 GT 路径；
- [ ] 核对每个 `.mat` 的影像键、GT 键、形状、波段轴和类别编码；
- [ ] 核对波长/波段对应关系；若波段数不一致，提供显式映射；
- [ ] 先运行每个任务并检查 `data_audit.json` 的类别计数与 Patch 隔离；
- [ ] 检查 `events.jsonl` 中 QP 求解器、约束残差、每类有效目标样本数、候选接受/拒绝和回滚；
- [ ] 只有三个任务都完成真实训练后，才汇总 OA/AA/Macro-F1/Kappa，并以多个随机种子报告均值与标准差；
- [ ] 不使用目标域 GT 调参或挑选检查点。

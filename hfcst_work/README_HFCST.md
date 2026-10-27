# HFCST-v9：层次化可行性约束光谱传输

本目录是在 `TCST_v8.1.0_Strict_Final` 基础上**新增**的独立 HFCST-v9 实现。原版 `core/`、`configs/`、`scripts/train.py`、`tests/` 和 `README_V810.md` 保留，不覆盖旧模型入口。

> 当前上传的归档中没有三个数据集的 `.mat` 原始文件。因此此处可验证的是代码、合成数据测试与训练流程；真实 Houston13→Houston18、Dioni→Loukia、Shanghai→Hangzhou 的 OA/AA/F1 尚未运行，不报告或推断真实数据性能。

## 1. 算法实现概览

1. **双视图编码器**（`core/model_hfcst.py`）
   - 3D 光谱—空间卷积提取局部体特征；压缩光谱维后使用 2D 残差上下文和多尺度空洞卷积。
   - 中心引导池化结合全局、中心和相似度注意力特征。
   - 独立中心光谱分支提取原始光谱、标准化形状、一阶/二阶差分和对数幅值。
   - `context_logits`、`spectral_logits` 和融合 `logits` 使用独立分类头。训练损失分别监督三个输出。
   - 样本级门控满足 `gate ∈ (0, max_gate)`，默认 `max_gate=0.5`。

2. **共同尺度与空间隔离**（`core/data_hfcst.py`）
   - 源域 GT 用于生成训练、Margin 约束、校准和验证集合；以整个空间块为单位分配集合。
   - 采样中心与其分配块边界至少相距 Patch 半径，跨集合 Patch 不共享像素。
   - 目标域影像加载时不读取目标 GT；目标估计和验收集合由互不重叠的空间块组成。
   - 用源域训练中心光谱的绝对值分位数估计唯一尺度，并对源域和目标域使用同一尺度。
   - 波段数不同且没有显式 `band_mapping.source_indices/target_indices` 时直接报错，不静默均匀重采样。

3. **层次化光谱变换**（`core/spectral_transform_hfcst.py`）
   - `phi_c = phi_shared + delta_phi[c]`，通过低维 DCT 基生成增益和偏移。
   - 共享参数初始为零、残差初始为零，因此初始映射为恒等变换。
   - 没有足够可靠目标支持的类别，其残差被固定为零。

4. **类别条件传输目标**（`core/discrepancy_hfcst.py`）
   - 结合光谱描述子 MMD 与融合特征 MMD。
   - 源域按真实源标签分组，目标域按 EMA 教师软类别概率与可靠性加权。
   - 每个有效类别独立归一化，可靠类别按类别平均，避免仅由大类支配。
   - 残差参数具有 L2 收缩项。全局 MMD/SWD 不作为唯一接受标准。

5. **Margin-QP**（`core/task_geometry_hfcst.py`、`core/transport_solver_hfcst.py`）
   - 约束样本从当前分类正确的源域约束区中选取，同时包含边界附近与较稳定样本。
   - QP 直接使用类别条件传输目标的梯度，采用硬 Margin 约束与信赖域。
   - 优先使用 CVXPY 可用求解器；当前环境缺少 CVXPY 时，新版 HFCST 使用 SciPy SLSQP 求解同一硬约束问题。无约束、求解失败、非有限梯度或不可行后验检查均返回零更新并拒绝迁移。

6. **事务式验收**（`core/risk_controller_hfcst.py`、`core/trainer_hfcst.py`）
   - 目标估计集和目标验收集分别生成 `q/r`，候选搜索过程中固定软类别分配。
   - 先搜索共享变换；通过后才搜索类别残差。
   - 验收使用相同的类别条件传输目标、空间块 Bootstrap、源域校准新增错误率/最差类别错误率/Margin 损失、光谱合理性和目标伪类别稳定性。
   - 仅当迁移候选通过迁移前验收后才执行变换源域训练；候选被拒绝时不会执行纯源域 CE 作为替代周期。
   - 后验失败时恢复模型、优化器、EMA 教师、调度器、传输参数和周期计数器，并停止适应。
   - `baseline.pt` 始终保留。最终模型保存为 `final.pt` 后才允许打开目标 GT，目标 GT 只用于最终 OA、AA、Macro-F1、Kappa 报告，不参与选择。

## 2. 环境与数据目录

Python 3.10+，建议使用独立虚拟环境。安装项目依赖：

```bash
pip install -r requirements.txt
pip install pytest
```

`requirements.txt` 中保留原版依赖（包括 `cvxpy`）。HFCST 在 CVXPY 不可用时可使用 SciPy QP 后备求解器，但原版旧测试中的 `core/transport_solver.py` 直接依赖 CVXPY。

设置 `DATA_ROOT` 为包含以下目录的**数据集根目录**：

```text
$DATA_ROOT/
├── Houston/
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── HyRANK/
│   ├── Dioni.mat
│   ├── Dioni_12gt.mat
│   ├── Loukia.mat
│   └── Loukia_12gt.mat
└── Shanghai_Hangzhou/Shanghai-Hangzhou/
    ├── Shanghai.mat
    ├── Shanghai_3gt.mat
    ├── Hangzhou.mat
    └── Hangzhou_3gt.mat
```

```bash
export DATA_ROOT=/absolute/path/to/Datasets
```

若你的实际文件名或目录结构不同，请编辑对应 YAML 的 `dataset.*` 路径。三个配置默认要求源域与目标域波段数相同；如果不同，必须提供有物理依据的显式波段索引映射。

## 3. 三个数据集的训练命令

在项目根目录执行。默认自动选择 CUDA（如可用），也可以显式指定 `--device cuda:0` 或 `--device cpu`。

### 3.1 Houston13 → Houston18

```bash
python -u scripts/train_hfcst.py --config configs_hfcst/houston.yaml --device cuda:0
```

### 3.2 Dioni → Loukia（HyRANK）

```bash
python -u scripts/train_hfcst.py --config configs_hfcst/hyrank.yaml --device cuda:0
```

### 3.3 Shanghai → Hangzhou

```bash
python -u scripts/train_hfcst.py --config configs_hfcst/shanghai_hangzhou.yaml --device cuda:0
```

若没有 NVIDIA GPU，可删除 `--device cuda:0` 或将其改为 `--device cpu`。每次实验的输出目录分别为 `results_hfcst/houston/`、`results_hfcst/hyrank/` 和 `results_hfcst/shanghai_hangzhou/`。

## 4. 推理命令

训练结束后，使用已保存的检查点对目标影像推理；该脚本不会加载 GT：

```bash
python -u scripts/predict_hfcst.py \
  --config configs_hfcst/houston.yaml \
  --checkpoint results_hfcst/houston/final.pt \
  --device cuda:0
```

另外两个任务分别替换配置和检查点路径。输出为 `.mat` 中的 `prediction`（类别编号为 1..C，背景/无效像素为 0）以及同名 `.npy` 文件。推理使用检查点中保存的源域共同尺度。

## 5. 输出文件和日志

每个任务的输出目录包含：

- `baseline.pt`：源域监督训练后、任何域适应前的基线模型。
- `last_committed.pt`：最近一个通过迁移前和训练后验收的状态（只有存在已提交适应周期时才生成）。
- `final.pt`：最终模型；若适应失败或没有候选通过，保留最后一个已提交状态/源域基线。
- `data_audit.json`：影像形状、无效像素、波段映射、共同尺度、源域各集合类别计数、空间块和 Patch 隔离审计。
- `events.jsonl`：逐周期目标值、QP 求解器/状态/更新范数、有效目标样本数、候选接受/拒绝原因、Bootstrap、校准风险、回滚和提交记录。
- `results.json`：可机器读取的最终汇总。目标 OA/AA 等仅在 `final.pt` 保存后计算。

若候选连续被拒绝，日志会记录原因并停止适应，不会为了凑足周期而执行纯 CE 训练。

## 6. 测试命令

```bash
# HFCST 新增合成数据测试（无需真实 HSI 数据）
pytest -q tests/test_hfcst.py

# 原有版本兼容性测试（需要 requirements.txt 中的 cvxpy）
pytest -q tests/test_core.py tests/test_v801.py tests/test_v810_p0p1p2.py \
  tests/test_mmd_synthetic.py tests/test_mmd_synthetic_v2.py
```

真实数据实验之前，应先检查 `data_audit.json`，确认每个源域集合有足够样本、目标估计/验收区非空、波段映射可信，并确认日志中记录了实际 QP 求解器。若严格空间隔离导致类别在训练/校准/验证区不足，代码会记录或拒绝适应，不会通过重叠 Patch 偷偷补齐。

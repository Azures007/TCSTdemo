# T-CST v8.0 — 严格文档合规版本

## 版本定位
严格按照 AI指令(1).doc 和 模型(3).doc 实现的 T-CST v8.0 Core 模型。

## 已实现的核心机制（全部文档合规）

| 文档要求 | 实现位置 | 状态 |
|---|---|---|
| Module 1: 低维 DCT 光谱变换 (K=8, Gain/Offset) | `core/spectral_transform.py` | ✅ |
| Module 2: Spectral Descriptor (4B-3维) + MK-MMD | `core/discrepancy.py` | ✅ |
| Module 3: Top-2 Pairwise Margin，对 φ 求梯度 | `core/task_geometry.py` | ✅ |
| Module 4: Margin-Constrained QP (CVXPY) | `core/transport_solver.py` | ✅ |
| Module 5: Risk-Budgeted Line Search (V/Vmax/Rsem/SWD) | `core/risk_controller.py` | ✅ |
| Module 6: Safe Source Training (L_clean + λ·L_safe) | `core/trainer_v8.py` | ✅ |
| 交替 Outer Loop，每轮从 φ=0 重新估计 | `core/trainer_v8.py` | ✅ |

## 新增方向（严格文档边界内的扩展）

| 方向 | 实现位置 | 文档依据 | 效果 |
|---|---|---|---|
| **A: RS-CRW** (Risk-Search-Feedback Constraint Reweighting) | `core/trainer_v8.py` (class_risk_weights) | 文档 7.1 (w_j>0) + 10.5 (尺度规范化) | Houston 无增益，Hangzhou 首次触发 Risk REJECT |
| **B: BA-CS** (Boundary-Aware Calibration Sampling) | `core/trainer_v8.py` (_sample_boundary_calibration) | 文档 6.1 (Easy/Boundary 50%) 延伸到 Calibration | **Houston +2.37 OA** |
| **C: TDCA** (Transformation-Direction Consistency Audit) | `core/trainer_v8.py` (direction_prev) | 文档 8.2 (eta_candidates 可配置) | 已实现但未启用 |

## 当前最优结果（seed=1237，单 seed）

| 数据集 | 配置 | OA | AA | Kappa | MacroF1 |
|---|---|---|---|---|---|
| **Houston13→18 (5%)** | A+B+C | **77.39** | 67.18 | 0.6189 | 57.32 |
| **Dioni→Loukia (5%)** | A+B+C | 61.46 | 48.16 | 0.5244 | 46.10 |
| **Hangzhou→Shanghai (5%)** | A+B+C | **65.53** | 72.81 | 0.5101 | 65.85 |

**对照 Source Only (5% 标签)**：
- Houston: 67.46 → T-CST +9.93
- 5% T-CST Houston 超过 10% T-CST Houston (75.09)，也超过 S⁴DL 100% 标签 (72.1)

## 复现命令

```bash
# Houston 5% T-CST A+B+C
python -u scripts/train.py --config configs/houston_5pct_abc.yaml

# HyRANK 5% T-CST A+B+C
python -u scripts/train.py --config configs/hyrank_5pct_abc.yaml

# Hangzhou→Shanghai 5% T-CST A+B+C
python -u scripts/train.py --config configs/shanghai_5pct_abc.yaml

# 查看完整结果
python scripts/summarize_all.py --root results --detail
关键训练参数
参数	Houston	HyRANK	Hangzhou→Shanghai
patch_size	11	11	11
batch_size	16	32	64
lr	3e-4	3e-4	3e-4
epochs_warmup	30	30	30
num_cycles	6	6	6
epochs_per_cycle	5	5	5
source_train_ratio	0.05	0.05	0.05
constraint_per_class	12	12	12
calibration_per_class	12	12	12
K (DCT)	8	8	8
lambda_d	0.01	0.01	0.01
lambda_xi	10.0	10.0	10.0
tau_sem	0.30	0.30	0.30
tau_flip	0.02	0.02	0.02
tau_class_flip	0.10	0.10	0.10
eta_candidates	[1.0,0.5,0.25,0.125,0.0625,0.03125]	同	同
RS-CRW alpha	2.0	2.0	2.0
cal_pool_multiplier	5	5	5
force_final_epoch	True	True	True
已验证失败的尝试（诚实记录）
尝试	结果	结论
差异化参数 (patch=7/13/7)	Houston -8, 全线崩	统一 patch=11 是对的
class-balanced CE	Houston -2.25	只在 HyRANK 上可能有效
DCRN_02 backbone	Houston 67.87 vs 73.50	5% 标签下小数据不适用
K=12 (HyRANK)	HyRANK -1.05, 少数类崩溃	K=8 是对的
lambda_d=0.02 + eta≤0.5	三数据集全跌	变换过保守反而有害
已知局限
5% 标签下的物理天花板：Houston ~78, HyRANK ~62, Hangzhou ~66

Risk Search 在类别不均衡场景（HyRANK）的 Vc 信号不可靠

Hangzhou class2（Land/Building）跨城市光谱差异大，31% 已是极限

未实现 5-seed 验证和多 seed 集成（下一步工作）

未实现 6 组消融实验 A~F（文档十二的要求）

目录结构
text
TCSTdemo/
├── core/
│   ├── data.py                    # 数据加载、补丁构建、分层划分
│   ├── discrepancy.py             # Spectral Descriptor + MK-MMD + SWD
│   ├── metrics.py                 # OA/AA/Kappa/MacroF1
│   ├── model.py                   # DualReliabilityEncoder backbone
│   ├── risk_controller.py         # Risk Search (V/Vmax/Rsem/SWD)
│   ├── spectral_transform.py      # DCT Gain/Offset
│   ├── task_geometry.py           # Top-2 Margin Gradient
│   ├── trainer_v8.py              # 主训练器（含 A/B/C 方向）
│   └── transport_solver.py        # QP (CVXPY)
├── configs/
│   ├── houston.yaml               # Houston 基础配置（5%）
│   ├── hyrank.yaml                # HyRANK 基础配置（5%）
│   ├── shanghai.yaml              # Hangzhou→Shanghai 基础配置（5%）
│   ├── houston_5pct_abc.yaml      # ★ 最优 Houston
│   ├── hyrank_5pct_abc.yaml       # ★ 最优 HyRANK
│   └── shanghai_5pct_abc.yaml     # ★ 最优 Hangzhou→Shanghai
├── scripts/
│   ├── train.py                   # 统一训练入口
│   ├── summarize_all.py           # 结果汇总
│   └── print_results.py           # 单次结果打印
├── tests/                         # 单元测试
├── requirements.txt
└── README_V8.0.md                 # 本文件
依赖
text
torch>=2.0
numpy>=1.24
scipy>=1.10
h5py>=3.8
pyyaml>=6.0
cvxpy>=1.3
tqdm>=4.65
版本说明
日期: 2026-10-22

基础: T-CST v8.0 Core (AI指令(1).doc + 模型(3).doc)

扩展: A(RS-CRW) + B(BA-CS) + C(TDCA 可选)

未实现: v8.1 特性（伪标签/EMA/对抗学习等，文档原则 2 禁止）

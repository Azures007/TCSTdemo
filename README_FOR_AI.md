# T-CST (Test-time Spectral Transform) v9.0 - 阶段性最终版

## 当前最高准确率 (单模型, seed=1237, 严格 150-shot)
| 数据集 | OA (%) | AA (%) | Kappa | 最优配置 |
|---|---|---|---|---|
| Houston13→18 | **80.12** | 73.45 | 0.6605 | Patch=11, L2+Amp, Spec=0.5, Mode=D |
| Dioni→Loukia (HyRANK) | **57.54** | 47.61 | 0.4767 | Patch=11, L2+Amp, Spec=0.4, CB=0.2, Pseudo(thr=0.7) |
| Shanghai→Hangzhou | **76.45** | 72.19 | 0.5350 | Patch=5, L2+Amp, Spec=0.4, CB=0.2, Pseudo(thr=0.8) |

## 核心架构与创新点
1. **Backbone**: DualReliabilityEncoder (Context 空间分支 + Spectral 光谱分支 + 融合门控)。
2. **物理先验注入**: 
   - 输入保持 L2 归一化。
   - 开启 `amplitude_channel=True`，强行注入绝对幅值 `log(||x||)`，解决跨传感器亮度差异。
   - 引入光谱辅助损失 `lambda_spec`，强制光谱分支学习，防止其被空间分支淹没。
3. **T-CST (Test-time Spectral Transform)**: 
   - 全局 Gain+Offset (DCT基 K=8)。
   - Margin-QP 约束方向 + Risk Search 回溯（防负迁移）。
   - 安全的 Safe Training（保留 clean 分支锚点）。
4. **选模机制**: `ckpt_mode=D` (基于 Source 验证集 NLL 的合格集选择，解决 Mode A 在 100% 验证准确率下的饱和丢弃问题)。
5. **目标域自训练**: 类别均衡伪标签 (Class-Balanced Pseudo-Label)，解决长尾分布小类被大类吞噬的问题。

## 失败的尝试（请不要建议这些）
1. **TTA (6路旋转)**: 彻底崩盘。高光谱对空间方向极其敏感，旋转 90/180 度会破坏物理上下文，S→H 暴跌 10 个点。
2. **增加 Epochs**: 严重过拟合。伪标签机制在长轮数下会发生“确认偏误”，导致小类特征彻底坍缩。
3. **500-shot / 全量数据**: 在缺乏特征解耦网络的情况下，增加源域数据量会加剧源域过拟合（10% 数据量时崩至 45%）。
4. **Mixup**: 配合增加的数据量，依然导致源域过拟合，效果极差。
5. **SupCon (监督对比学习)**: 在开启 AMP 半精度且与伪标签结合时，导致 CUDA 底层抛出 Segmentation Fault (段错误)，静默闪退，无法在现有硬件下稳定运行。
6. **Z-score 归一化**: 虽然救了 S→H，但彻底摧毁了 Houston 和 HyRANK 的光谱形状特征。

## 核心痛点
- **HyRANK (57.54%)**: 12 类极度长尾，且存在严重的类条件偏移。150-shot 下，极小类（<30像素）根本无法提取特征，伪标签也无法选到它们。
- **S→H (76.45%)**: 3 类，严重混合像元。11x11 甚至 7x7 的 Patch 都会混入大量背景，导致植被（Plant）识别率极低。

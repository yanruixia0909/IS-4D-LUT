# IS-4D LUT

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

依据论文：
> **IS-4D LUT: Incorporating Intensity and Semantics Aware into 3D Lookup Table for
> Low-Light Image Enhancement**（Yan, Jin, Zuo, Zhang, Su, Wang）

本工程把论文第 3 节的方法完整实现为可训练、可推理的 PyTorch 代码，并保证开箱即可跑通
（自带合成数据，无需先下载 LOL / FiveK / SMID / SDSD）。

---

## 1. 方法要点 → 代码映射

| 论文内容 | 公式 | 代码位置 |
|---|---|---|
| 3D LUT / 三线性插值（作为背景） | Eq. 2–6 | `is4dlut/models/lut4d.py`（扩展为 4D） |
| 基 LUT 融合 `L_4D = Σ ω_i ψ_i` | Eq. 1 | `QuadrilinearLUT.fuse()` + `LGM.rho` |
| 强度图 `F_d = D(F_t)`、`F_p = I_L + F_d` | Sec. 3.1 | `is4dlut/models/lgm.py` |
| 4D 索引与 16 邻点四线性插值 | Eq. 3–6 | `QuadrilinearLUT._sample_single()` |
| 语义感知嵌入 SAEM | Eq. 8–9 | `is4dlut/models/saem.py` |
| 注意力融合 AFM（自注意力 + 双向交叉注意力） | Sec. 3.2 | `is4dlut/models/afm.py` |
| 语义融合金字塔 + 重建 `I_o = R(F_s)` | Eq. 10 | `is4dlut/models/sfm.py` |
| `L_total1 = α_s L_s + α_m L_m + α_mse L_mse + α_ssim L_ssim` | Eq. 11 | `is4dlut/losses.py::IS4DLUTLoss.generator` |
| `L_total2 = L_recon + λ_sc L_sc + λ_sa L_sa` | Eq. 12 | 同上（重建 / 直方图 / 语义对抗） |
| 训练设置 Adam(0.9,0.999)、lr 1e-4、batch 1、1e5 迭代 | Sec. 4.1.3 | `is4dlut/config.py`、`is4dlut/engine.py` |

`L_s`（4D 平滑）与 `L_m`（4D 单调）分别由 `lut_total_variation()`、`lut_monotonicity()`
在融合后的 4D 表上计算。

---

## 2. 目录结构

```
Is 4D LUT/
├── is4dlut/
│   ├── config.py            # 超参数（论文 preset + demo preset + 推断项说明）
│   ├── datasets.py          # 配对数据 / 无参考数据 / 合成低光数据
│   ├── engine.py            # 训练循环（D/G 交替、AMP、学习率衰减、checkpoint）
│   ├── losses.py            # 7 项损失 + PatchGAN 判别器
│   ├── metrics.py           # PSNR / SSIM / ΔE_ab
│   ├── utils.py             # 中文路径读写、颜色工具、种子
│   └── models/
│       ├── blocks.py        # Conv/ReLU、ResBlock、Encoder、IntensityDecoder
│       ├── lut4d.py         # 4D LUT + 四线性插值 + 平滑/单调正则
│       ├── lgm.py           # LUT 生成模块
│       ├── saem.py          # 语义感知嵌入
│       ├── afm.py           # 注意力融合
│       ├── sfm.py           # 语义感知融合模块
│       └── is4dlut.py       # 整体网络
├── tools/
│   ├── demo.py              # 训练 + 推理 + 指标（一分钟跑完）
│   ├── train.py             # 正式训练入口
│   ├── inference.py         # 单图/文件夹推理
│   ├── evaluate.py          # 配对数据评估
│   ├── compare_gt.py        # 低光/增强/GT 三联图 + 逐图 PSNR/SSIM
│   ├── benchmark.py         # 效率测试（预热 50 + CUDA 计时 200 次，mean±std）
│   └── make_synthetic_dataset.py  # 生成 low/high 目录
├── tests/test_smoke.py      # 恒等 LUT、前/反向、损失、数据自检
├── docs/experiment_setup.md # 论文用实验配置（逐项对应代码 + 实测效率）
├── configs/                 # 论文 & demo 配置样例（json）
├── checkpoints/  results/  data/
└── requirements.txt
```

---

## 3. 环境

在 **Python 3.10 + PyTorch 2.x + CUDA 11.8 + RTX 4060 Ti 16 GB** 上验证通过（Windows / Linux 均可）。

```powershell
pip install -r requirements.txt
```

依赖：`torch>=2.0`、`numpy`、`opencv-python`（不依赖 yaml / torchvision / PIL）。

---

## 4. 快速开始

### 4.1 自检（CPU，几秒）

```powershell
cd IS-4D-LUT          # 仓库根目录
python tests/test_smoke.py
```

预期输出：

```
[ok] 4D LUT identity interpolation, max error = 5.96e-08
[ok] differentiable histogram + Wasserstein distance
[ok] forward + backward, total loss = ...
[ok] synthetic low/normal pairs
All smoke tests passed.
```

### 4.2 一分钟 Demo（合成数据：训练 → 推理 → 指标）

```powershell
python tools/demo.py --iters 5 --device cuda
```

会在 `results/demo/` 下生成 `comparison_XX.png`（低光 / 增强 / GT 三联图）、
`metrics.json`（每张图的 PSNR、SSIM）以及 `checkpoints/demo/is4dlut_last.pth`。

### 4.3 在真实配对数据上训练

数据布局（自动识别多种命名）：

```
dataset_root/
├── low/    xxx.png
└── high/   xxx.png
# 也支持 input/target、our485/low+high、train/low+high、Train/Input+GT
```

也可以先用脚本造一份合成配对数据：

```powershell
python tools/make_synthetic_dataset.py --out data/synthetic_pairs --num 8 --size 128
```

论文规模训练（N=33、M=3、batch=1、lr=1e-4、1e5 迭代）：

```powershell
python tools/train.py --preset paper --data-root D:/datasets/LOL/our485 `
    --val-root D:/datasets/LOL/eval15 --device cuda
```

快速冒烟（小 LUT、少迭代）：

```powershell
python tools/train.py --data-root data/synthetic_pairs --iters 2 --n-grid 9 `
    --base-channels 16 --num-levels 2 --crop-size 96 --real-data --device cuda
```

### 4.4 推理（单图或整个文件夹）

```powershell
python tools/inference.py --ckpt checkpoints/is4dlut_last.pth `
    --input E:\path\to\images --out-dir results/inference --device cuda --max-size 0
```

`--max-size` >0 时会把长边缩放到该尺寸后再增强；`--suffix` 控制输出后缀（默认 `_is4dlut`）。

### 4.5 评估（配对数据）

```powershell
python tools/evaluate.py --ckpt checkpoints/is4dlut_last.pth `
    --data-root D:/datasets/LOL/eval15 --device cuda --out results/metrics.json
```

---

## 5. 超参数

**论文明确给出的**（`config.py::paper_config`）：

| 项目 | 值 |
|---|---|
| LUT bins | **N = 33**（Sec. 3.1 / Sec. 4.3） |
| 优化器 | Adam，β1=0.9，β2=0.999 |
| 学习率 | 1×10⁻⁴ |
| batch size | 1 |
| 迭代次数 | 1×10⁵ |
| 硬件 | NVIDIA 4060 Ti |

**论文未写明、由本实现推断的**（`config.py::INFERRED_SETTINGS` 也列了一份）：

| 项目 | 取值 | 依据 |
|---|---|---|
| 基 LUT 数量 M | 3 | 论文只说 “M context-driven base LUTs”；3 是图像自适应 LUT 的常用设置 |
| 强度通道 d_in | `luminance(F_p)`（0.299/0.587/0.114） | “input intensity value”；可切换为 `F_d`（`ModelConfig.intensity_from="intensity"`） |
| 编码器宽度 | base=32，3 次下采样 | 论文只给了示意图 |
| 语义来源 | 内部语义编码器（输入 `[I_r, I_L]`） | 论文未说明使用外部语义标签或预训练模型 |
| 损失权重 | α_s=1e-4、α_m=10、α_mse=1、α_ssim=0.1、β=1、λ_sc=0.1、λ_sa=0.1 | 对齐 Fig. 10 的扫描范围 |
| L_sc 实现 | 可微 soft histogram + Wasserstein-1（逐语义区域）+ 直方图均衡辅助项 | 论文描述“分区约束 + 可微直方图 + 均衡辅助” |
| L_sa 实现 | 全局 + 局部 PatchGAN，hinge loss，语义显著性加权 | 论文只说明“局部/全局判别器 + 语义先验引导” |
| 语义区域数 K | 3 | `ModelConfig.n_regions` |

---

## 6. 本机已验证结果

| 项目 | 命令 | 结果 |
|---|---|---|
| 4D LUT 恒等插值 | `python tests/test_smoke.py` | 最大误差 **5.96e-08**（16 邻点四线性插值正确） |
| 前向/反向 + 全部损失 | 同上 | 损失有限、梯度存在 |
| Demo（CPU） | `tools/demo.py --iters 3 --device cpu` | 0.21 s/it |
| Demo（GPU） | `tools/demo.py --iters 5 --device cuda` | 0.12 s/it |
| 论文规模（N=33, M=3, base=32） | `tools/train.py --preset paper --synthetic --iters 1` | **29.53 M 参数**，0.69 s/it @128²（4060 Ti，AMP） |
| 真实配对目录训练 | `tools/train.py --data-root data/synthetic_pairs ...` | 正常训练并保存 checkpoint |
| 评估 | `tools/evaluate.py ...` | 配对数据评估与指标导出 |
| 真实图片推理 | `tools/inference.py --input ...DICM...jpg` | 960×640 与 `--max-size 512` 均正常输出 |

---

## 7. 复现说明与已知差异

1. **数据**：论文用 LOL(v1/v2)、FiveK、SMID、SDSD 训练/测试。本工程不附带数据集，
   但 `datasets.py` 已适配常见目录结构；把数据放好并把 `--data-root` 指过去即可。
   要复现表格数字需按论文设置训练 1×10⁵ 迭代。
2. **语义分支**：论文没有交代语义特征的来源（外部语义分割？自监督？），因此这里用
   与图像分支同构的内部编码器，输入 `[I_r, I_L]`，再由 SAEM/AFM 建立跨模态对齐。
   如果你们实际使用的是某个分割/语义网络，只需替换 `SFM.sem_encoder` 的输出接口。
3. **4D LUT 存储**：基表参数量为 `M × 3 × N⁴`；N=33、M=3 时约 10.7 M 参数
   （总参数 29.5 M）。显存占用随 N⁴ 增长，若显存不足可先减小 `n_grid`（如 17）。
4. **插值实现**：采用 16 邻点四线性插值（Eq. 6 的完整展开），先在强度轴、再依次在
   k/j/i 轴上做分离式线性插值，与公式等价且中间显存更可控。
5. **判别器**：论文未给结构，本实现采用 3 层 PatchGAN（全局 + 局部中心裁剪）。
6. **AMP**：`torch.cuda.amp` 默认开启（仅在 CUDA 下生效）；如遇数值问题可加 `--no-amp`。
7. **注意力缩放**：论文 Eq. 9 写作除以 `C`，代码里用标准的 `1/sqrt(C)` 缩放，避免
   softmax 过平坦导致 SAEM 退化；如需严格照搬，把 `saem.py` 的 `math.sqrt(c)` 改为 `c`。
8. **不包含**：论文中的推理时间/参数量表格、可视化对比图、NVIE/PI/BRISQUE 等无参考指标
   （这些需要额外模型/数据集，可用 `metrics.py` 的 PSNR/SSIM/ΔE 先跑配对评估）。

---

## 8. 常用自定义

```python
from is4dlut.config import paper_config

cfg = paper_config()
cfg.model.n_grid = 17            # 更省显存
cfg.model.n_basis = 5            # 更多基 LUT
cfg.model.intensity_from = "intensity"   # 用 F_d 作为第 4 维
cfg.loss.lambda_sa = 0.0         # 关闭对抗损失
```

```powershell
python tools/train.py --preset paper --n-grid 17 --data-root ... --device cuda
```

---

## 9. 用 LOLv1 训练（实测配置）

数据目录（自行下载 [LOLv1](https://daooshee.github.io/BMVC2018website/) 后按下面的结构摆放）：

```
path\to\LOLv1\
├── Train\input\   *.png   (485 张低光)
├── Train\target\  *.png   (485 张 GT，文件名与 input 一一对应)
└── Test\input|target      (15 对，另有 2 个 txt 划分文件会被自动忽略)
```

论文配方（N=33、M=3、batch=1、lr=1e-4、1e5 迭代）：

```powershell
python tools/train.py --preset paper `
    --data-root path\to\LOLv1\Train `
    --val-root  path\to\LOLv1\Test `
    --out-dir results\lolv1_train --ckpt-dir checkpoints\lolv1 `
    --device cuda
```

短程训练示例（1 万迭代）：

```powershell
python tools/train.py --preset paper `
    --data-root path\to\LOLv1\Train --val-root path\to\LOLv1\Test `
    --iters 10000 --log-every 200 --val-every 2000 --save-every 2000 `
    --lr-decay-start 8000 --no-amp --val-images 3 `
    --out-dir results\lolv1_train --ckpt-dir checkpoints\lolv1 --device cuda
```

`--val-images N`（默认 0=关闭）会在每次校验时把前 N 张验证图的
`低光 | 增强 | GT` 三联图存到 `results\lolv1_train\iterXXXXXXX\`；权重存到
`checkpoints\lolv1\is4dlut_iterXXXXXXX.pth`（含优化器状态，约 287 MB/个）。

> **实测结论：这份工作负载请用 `--no-amp`（fp32）。** 在 4060 Ti 上 fp32 与 AMP 的吞吐几乎相同
> （0.14–0.19 s/it @256²），而 fp16 下梯度缩放会周期性溢出（LUT 融合/SSIM 反传需要 fp32）。
> 代码已把 LUT 融合与采样、SSIM、直方图损失、判别器都固定为 fp32，并在 `engine.py` 里加了
> 非有限损失跳步保护，但整体仍建议 fp32 训练。

比较结果（低光 / 增强 / GT 三联图 + PSNR/SSIM）：

```powershell
python tools/compare_gt.py --ckpt checkpoints\lolv1\is4dlut_last.pth `
    --data-root path\to\LOLv1\Test --out-dir results\lolv1_compare --num 5
```

中断后继续训练：

```powershell
python tools/train.py --preset paper `
    --data-root path\to\LOLv1\Train --val-root path\to\LOLv1\Test `
    --iters 100000 --resume checkpoints\lolv1\is4dlut_last.pth `
    --out-dir results\lolv1_train --ckpt-dir checkpoints\lolv1 --device cuda
```

> AMP 说明：fp16 下 AFM/SAEM 的 C×C 注意力、4D LUT 融合与采样、SSIM 都会溢出，
> 代码已把这几处强制为 fp32（卷积仍走 AMP），并加了非有限损失/梯度跳步保护；
> 即便如此，本工作负载仍推荐直接 `--no-amp`。

---

## 10. 论文用实验配置（可直接写进文章）

完整的“实验设置”文本（数据集与预处理 / 优化器与学习率 / LGM 与语义分支结构 /
7 项损失及权重 / 判别器 / 推理与效率测试协议）逐项对应本仓库代码，并附实测时延：

📄 **`docs/experiment_setup.md`**（逐项配置对照代码 + 与常见转述写法的差异表）

效率测试脚本（批大小 1、FP32、预热 50 次、CUDA 事件计时 200 次、报告 mean±std）：

```powershell
python tools/benchmark.py --ckpt checkpoints\lolv1\is4dlut_last.pth --size 400x600
```

| 分辨率 | mean ± std | FPS | 峰值显存 |
|---|---|---|---|
| 256×256 | 15.09 ± 0.49 ms | 66.3 | 622 MB |
| 400×600 | 65.04 ± 0.41 ms | 15.4 | 1178 MB |
| 512×512 | 71.60 ± 0.45 ms | 14.0 | 1246 MB |

---

## 11. 已知限制

1. **未复现论文全部表格结果**：本仓库提供训练与评估流程；论文表格结果需结合相应数据和第 9 节的完整训练配置另行验证。
2. **语义分支为内部编码器**：论文未说明语义特征来源，本实现用与图像分支同构的编码器
   （输入 `[I_r, I_L]`）。若你使用分割/语义网络，只需替换 `SFM.sem_encoder` 的输出接口。
3. **实现范围**：本仓库提供可运行的模型、训练与评估流程，尚未验证达到论文全部表格结果；具体实现差异见第 7 节。

---

## 12. 许可与引用

- 代码以 **MIT License** 发布，见 [LICENSE](LICENSE)。
- 使用本仓库时，请引用上述论文（论文尚未正式发表，
  发表后请按官方信息补全以下字段）：

```bibtex
@article{is4dlut,
  title   = {IS-4D LUT: Incorporating Intensity and Semantics Aware into 3D Lookup
             Table for Low-Light Image Enhancement},
  author  = {Yan and Jin and Zuo and Zhang and Su and Wang},
  journal = {--},   % 待论文正式发表后补全
  year    = {--}    % 待论文正式发表后补全
}
```

> 若同时需要引用本实现，可写：*"Code available at https://github.com/yanruixia0909/IS-4D-LUT"*。

- 数据集（LOL / FiveK / SMID / SDSD 等）请遵守各自原始许可，本仓库不分发任何数据集。




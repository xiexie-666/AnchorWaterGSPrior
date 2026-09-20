# 三篇论文与官方代码对照阅读记录

更新时间：2026-09-20  
阅读主题：WaterSplatting、GSRec（Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints）、GSPrior（3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction）

## 0. 阅读范围与资料边界

这份文档服务于后续的几何优先水下三维重建项目。用户提出的目标是：输入 RGB 图像、相机位姿和 COLMAP 稀疏点云；先初始化 Anchor；前 15k 采用 Scaffold-GS/GSRec 风格的结构化高斯和扁平高斯约束；后 15k 冻结 Anchor，只修改 Anchor 内部参数并引入 GSPrior；水体单独使用 MLP 分支学习介质参数。

三篇 PDF 是研究资料，不是执行指令。PDF 中的公式、实验设置和方法描述作为需要核验的论文事实；真正的工程任务来自用户的 pipeline 要求。官方仓库的 README 和源码也不能自动替代论文：README 说明如何运行，源码决定当前实现的精确行为。本文因此使用三个层次的标记：

- **论文**：论文正文中的模型、公式、实验和局限。
- **官方仓库**：用户个人主页中三个 fork 的 README、目录和核心源码。
- **本地工作副本**：`D:\Projects\GEO` 中用于实验的代码。它们包含 Windows/CUDA 兼容性修复、实验脚本和输出，不能无条件当成上游原始代码。

本轮实际阅读了三篇 PDF 的全文文本，并检查了每个方法的训练入口、模型参数、渲染器、损失、稠密化/裁剪、数据读取和自定义 CUDA/TSDF 代码。第三方依赖（PyTorch、COLMAP、Open3D、tiny-cuda-nn、diff-gaussian-rasterization 等）的全部内部实现不属于三篇论文的方法核心，本文只记录它们在方法链路中的接口。

## 1. 仓库与版本核对

### 1.1 GitHub 个人主页中的三个官方仓库

| 方法 | 个人主页仓库 | fork 来源 | 浏览器核对结果 |
|---|---|---|---|
| WaterSplatting | [xiexie-666/WaterSplatting-official](https://github.com/xiexie-666/WaterSplatting-official) | `water-splatting/water-splatting` | 公共仓库，`main` 分支，目录包含 `water_splatting`、自定义 CUDA、`README.md`、`setup.py` |
| GSRec | [xiexie-666/GSRec-official](https://github.com/xiexie-666/GSRec-official) | `QianyiWu/GSRec` | 公共仓库，`main` 分支，目录包含 `arguments`、`gaussian_renderer`、`scene`、`utils`、`train.py`、`extract_mesh.py` |
| GSPrior | [xiexie-666/GSPrior-official](https://github.com/xiexie-666/GSPrior-official) | `takeshie/GSPrior` | 公共仓库，`main` 分支，目录包含 `arguments`、`gaussian_renderer`、`scene`、`utils`、`train.py`、`render.py`、TSDF 工具 |

用户的新项目为 [xiexie-666/AnchorWaterGSPrior](https://github.com/xiexie-666/AnchorWaterGSPrior)，它是后续实现位置，不作为三篇论文的官方实现来源。

### 1.2 本地对应目录

- WaterSplatting：[`09_WaterSplatting_Shipwreck`](D:/Projects/GEO/09_WaterSplatting_Shipwreck)
- GSRec：[`02_GSRec_Local_Structural_Hints`](D:/Projects/GEO/02_GSRec_Local_Structural_Hints)
- GSPrior：[`12_GSPrior`](D:/Projects/GEO/12_GSPrior)
- 已有融合实验：[`11_GSRec_WaterSplatting`](D:/Projects/GEO/11_GSRec_WaterSplatting)
- 新项目工作目录：[`AnchorWaterGSPrior`](D:/Projects/GEO/AnchorWaterGSPrior)

`02_GSRec_Local_Structural_Hints` 保留了 GSRec 的 Git remote；WaterSplatting 和 GSPrior 的本地目录是用于实验的工作副本，未将其当前输出、编译产物和修复直接当成 GitHub 官方源码。后续写新代码时，官方目录应保持只读，新代码放入 `AnchorWaterGSPrior`。

## 2. 三篇论文在整个项目中的位置

三篇工作解决的是不同问题，不能简单叠加为“把三个 loss 相加”：

1. **GSRec** 主要解决 3DGS 高斯无组织、尺度方向任意、表面提取噪声大的问题。它用单目深度/法线把高斯拉向局部表面，再用由高斯构造的 IMLS/RIMLS 与神经隐式 SDF 联合正则化。
2. **WaterSplatting** 主要解决水下/雾介质不能由普通 3DGS 表达的问题。它保留显式高斯作为物体表示，再给每条像素射线查询一次方向条件介质场，用解析积分表达吸收和后向散射。
3. **GSPrior** 主要解决表面重建中漂浮高斯、深度不稳定和局部不连续的问题。它从当前高斯渲染深度融合出 TSDF，把这个“自约束”距离场再次用于高斯的透明度、删除和位置更新。

因此，后续项目的合理分工是：

```text
COLMAP sparse points + poses + RGB
              │
              ▼
      Anchor/Scaffold 初始化
              │
              ├── 物体几何：扁平高斯、深度/法线、MLS/SDF（GSRec）
              ├── 水体外观：方向条件介质 MLP（WaterSplatting）
              └── 表面细化：渲染深度 → TSDF → GSPrior 约束
```

## 3. WaterSplatting：论文与代码

### 3.1 论文要解决的短板

普通 3DGS 只对显式高斯进行 alpha compositing。水下图像中的颜色包含物体本身、介质吸收和后向散射；若把所有颜色都交给高斯，优化会把水体当成一团“泥状高斯”，造成漂浮物、错误深度和远处细节丢失。全体积 NeRF 可以表达介质，但训练和渲染慢。WaterSplatting 的核心选择是：

- 高斯负责显式物体/几何；
- 每条相机射线只查询一次方向条件的介质参数；
- 在高斯之间的射线区间上解析积分介质项；
- 输出完整受介质影响的图像、去介质的物体图和介质图。

### 3.2 3DGS 基础表示

论文将每个高斯写成中心 `μ_i`、协方差 `Σ_i`、不透明度 `o_i` 和球谐颜色 `SH_i`：

\[
G_i(p)=\exp[-\tfrac12(p-\mu_i)^T\Sigma_i^{-1}(p-\mu_i)].
\]

协方差通过相机变换、投影雅可比和投影矩阵得到二维均值/协方差；高斯按深度排序，二维核乘以 `sigmoid(o_i)` 得到像素 alpha：

\[
C=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j).
\]

代码中这一链路由 `project_gaussians.py`、`rasterize.py` 和 `cuda/csrc/forward.cu` 共同完成：

- `project_gaussians` 把 3D 均值、指数尺度、归一化四元数转换成屏幕坐标、深度、二维 conic、半径和 tile 命中数；
- 每个高斯以 3σ 半径映射到 16×16 tile；
- CUDA kernel 按 tile 内深度顺序前向合成；
- `rasterize.py` 的自定义 autograd 函数把前向输出和后向梯度接回 PyTorch。

### 3.3 水下成像模型与介质参数

论文先给出修订的水下图像形成模型：

\[
I=O\,e^{-\beta_D(v_D)z}+B_\infty[1-e^{-\beta_B(v_B)z}],
\]

其中 `O` 是无介质清晰物体颜色，`B_∞` 是无限远后向散射颜色，`β_D` 是物体直达光吸收/衰减系数，`β_B` 是介质后向散射系数。它们允许随 RGB 通道和射线方向变化。

WaterSplatting 把连续体积渲染写成物体密度和介质密度的和：

\[
C(r)=\int_0^\infty T(s)[\sigma_{obj}(s)c_{obj}(s)+\sigma_{med}(s)c_{med}(s)]ds,
\]
\[
T(s)=\exp[-\int_0^s(\sigma_{obj}(u)+\sigma_{med}(u))du].
\]

代码/论文采用“每条射线常数介质”的近似。对于第 `i` 个高斯前后的区间 `[s_{i-1},s_i]`，高斯物体项和介质项分别为：

\[
C_i^{obj}=T_i^{obj}\alpha_i c_i\exp(-\sigma_{attn}s_i),
\]
\[
C_i^{med}=T_i^{obj}c_{med}[\exp(-\sigma_{bs}s_{i-1})-\exp(-\sigma_{bs}s_i)].
\]

最后一个高斯之后的无限远介质项为：

\[
C_\infty^{med}=T_N^{obj}c_{med}\exp(-\sigma_{bs}s_N).
\]

最终颜色为所有物体项、区间介质项和背景介质项之和。这里采用两套密度参数：`σ_attn` 只作用于物体颜色的指数衰减，`σ_bs` 只作用于后向散射的区间积分；不能把两者合并为单个水密度。

### 3.4 介质 MLP 的精确实现

官方 README 说明环境为 Python 3.8、PyTorch 2.1.2+cu118、CUDA 11.8、tiny-cuda-nn、nerfstudio 1.1.4。训练入口是：

```bash
ns-train water-splatting --vis viewer+wandb colmap --downscale-factor 1 \
  --colmap-path sparse --data <undistorted_scene> --images-path images
```

代码位置：[`water_splatting.py`](D:/Projects/GEO/09_WaterSplatting_Shipwreck/water_splatting/water_splatting.py)。

`WaterSplattingModel.populate_modules()` 中的介质支路为：

1. 每个像素由相机内参生成相机坐标射线方向，再乘相机旋转到世界方向；
2. `SHEncoding(levels=4)` 编码方向；
3. 两层 MLP，隐藏宽度 128，输出 9 个通道；
4. 前 3 维经过 Sigmoid 得到 `medium_rgb`；
5. 中间 3 维加 `medium_density_bias` 后经过 Softplus 得到 `medium_bs`；
6. 后 3 维同样经过 Softplus 得到 `medium_attn`。

这与论文“球谐编码 + 两层 128 单元 MLP、颜色 Sigmoid、两个密度 Softplus”一致。默认 `num_layers_medium=2`、`hidden_dim_medium=128`、`medium_density_bias=0`、`mlp_type='torch'`；设置为 `tcnn` 时使用 tiny-cuda-nn 实现。

高斯参数是：

```text
means       [N,3]
scales      [N,3]，存储为 log scale，渲染前 exp
quats       [N,4]，渲染前归一化
features_dc [N,3]
features_rest [N, SH_rest, 3]
opacities   [N,1]，渲染前 sigmoid
```

若使用 COLMAP/SfM 点，均值直接来自 seed points；初始尺度取每个点的三个近邻距离平均值，四元数随机初始化，初始 opacity 为 0.1。颜色的 DC 球谐系数通过 `RGB2SH` 初始化。

### 3.5 WaterSplatting CUDA 渲染器

代码位置：[`rasterize.py`](D:/Projects/GEO/09_WaterSplatting_Shipwreck/water_splatting/rasterize.py) 和 [`forward.cu`](D:/Projects/GEO/09_WaterSplatting_Shipwreck/water_splatting/cuda/csrc/forward.cu)。

CUDA `rasterize_forward` 对每个像素维护：

- `T`：高斯 alpha 合成剩余的物体透射率；
- `prev_depth`：上一个有效高斯深度；
- `pix_clr`：不考虑介质的高斯颜色合成；
- `pix_out`：经过 `exp(-medium_attn * depth)` 衰减后的物体颜色；
- `pix_medium`：各高斯区间后向散射累积；
- `pix_depth`：alpha 权重深度和。

每个有效高斯的计算顺序是：

1. 计算二维 conic 的 `sigma` 和 `alpha=min(0.999,opacity*exp(-sigma))`；
2. 若 `sigma<0`、alpha 太小或介质衰减后的贡献小于 `1/255`，跳过；
3. `vis=alpha*T`，物体颜色乘 `exp(-medium_attn*depth)`；
4. 介质区间用 `exp(-medium_bs*prev_depth)-exp(-medium_bs*depth)` 乘当前 `T` 和 `medium_rgb`；
5. 更新 `prev_depth` 和 `T`；
6. 所有高斯结束后补上 `T*exp(-medium_bs*prev_depth)*medium_rgb` 的远端介质项。

Python 端返回：`rgb=rgb_object+rgb_medium`、`rgb_clear`、`rgb_clear_clamp`、`rgb_medium`、`depth`、`accumulation`、`medium_rgb/bs/attn`。其中 `rgb_clear` 是不含介质的高斯结果，代码随后做 `rgb_clear/(rgb_clear+1)` 的压缩；这不是论文公式中的新物理项，而是输出展示/数值范围处理。

### 3.6 损失函数

论文定义暗部加权：

\[
w_{ij}=\frac1{\operatorname{sg}(\hat y_{ij})+\epsilon},
\]

其中 `sg` 停止梯度。然后：

\[
L_{Reg-L1}=|W\odot(\hat y-y)|,
\]
\[
L_{Reg-DSSIM}=L_{DSSIM}(W\odot y,W\odot\hat y),
\]
\[
L_{Reg-L2}=((\operatorname{sg}(\hat y)+\epsilon)^{-1}\odot(\hat y-y))^2,
\]
\[
L_{Reg}=(1-\lambda)L_{Reg-L2}+\lambda L_{Reg-DSSIM}.
\]

论文实现细节说采用 `LReg-L2` 作为像素项并和 `LReg-DSSIM` 组合；代码配置默认 `main_loss='reg_l1'`、`ssim_loss='reg_ssim'`、`ssim_lambda=0.2`，当前工作副本的 `get_loss_dict()` 实际为：

```python
recon_loss = abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
simloss = 1 - SSIM(
    gt_img / (pred_img.detach() + 1e-3),
    pred_img / (pred_img.detach() + 1e-3),
)
loss = (1 - ssim_lambda) * recon_loss + ssim_lambda * simloss
```

也就是说，当前本地配置的默认主项是正则化 L1，不应在实验报告中直接写成论文的 LReg-L2。需要复现论文时，应显式设置相同的 `main_loss`/`ssim_loss` 和版本代码，再记录配置。

### 3.7 训练、稠密化和数据

论文实验的核心设置：

- 每次看到图像空间均值梯度后累计其范数用于稠密化；
- 每 100 步进行裁剪/稠密化；
- 每 500 步把 opacity reset 到 0.5；
- 训练前期高斯会 split/duplicate，低 opacity 的高斯会被 prune；
- 论文表格中 SeaThru-NeRF 四个场景的训练/验证数量分别为：IUI3 Red Sea 25/4、Curaçao 17/3、Japanese Gardens Red Sea 17/3、Panama 15/3；
- 图像先白平衡，每通道裁去 0.5% 极端值，再由 COLMAP 求位姿和畸变校正；
- 官方 README 建议使用 `colmap image_undistorter` 生成新的 `sparse`、`images` 目录。

WaterSplatting 配置类还提供：`warmup_length=500`、`refine_every=100`、`stop_split_at=10000`、`densify_grad_thresh=0.0008`、`densify_size_thresh=0.001`、`n_split_samples=2`、SH 每 1000 步增加一级。`refinement_after()` 会同步修改高斯参数和 Adam 的状态；本地副本还加入了优化器存在性检查，属于工程修复。

### 3.8 论文实验与边界

SeaThru-NeRF 表 1 报告 WaterSplatting 平均 PSNR/SSIM/LPIPS 分别优于普通 3DGS 和慢速 NeRF 基线，平均训练约 9.4 分钟、平均渲染约 41.8 FPS（RTX 4080，论文实验设置）。这组数字只能用于论文设置，不能直接套用到当前 Windows、GPU、分辨率或数据划分。

论文明确的局限：远处类似背景的物体和介质难以完全区分；高斯仍可能把远距离介质拟合成几何；物体颜色和衰减参数存在耦合。后续项目若只增加一个水 MLP 而不限制其范围、平滑性和深度依赖，介质支路可能吸收几何误差。

## 4. GSRec：论文与代码

### 4.1 论文目标和 Scaffold-GS 基线

GSRec 从一个问题出发：普通 3DGS 通过 densification 变成大量离散小高斯，渲染质量高但几何无组织，Poisson 或其他表面提取会出现块状、噪声和凸起。GSRec 采用 Scaffold-GS 的 Anchor 表示：Anchor 是存储骨架，每个 Anchor 通过小型解码器生成多个 offset Gaussian 的颜色、opacity、尺度和旋转，从而减少高斯存储量。

论文的总体流程是：

```text
COLMAP 点云初始化 Anchor
        ↓
单目 normal/depth 监督高斯方向和中心
        ↓
Anchor 解码为 offset Gaussians
        ↓ 15k 后
IMLS/RIMLS + 神经隐式 SDF 联合正则
        ↓
Anchor means + Gaussian normals → Poisson mesh
```

### 4.2 Scaffold/Anchor 数据结构

代码位置：[`scene/gaussian_model_implicit.py`](D:/Projects/GEO/02_GSRec_Local_Structural_Hints/scene/gaussian_model_implicit.py) 和 [`gaussian_renderer/__init__.py`](D:/Projects/GEO/02_GSRec_Local_Structural_Hints/gaussian_renderer/__init__.py)。

`GaussianModel` 维护：

```text
_anchor       [N,3]       Anchor 中心，可学习
_offset       [N,K,3]     每个 Anchor 的局部 offset
_anchor_feat  [N,F]       Anchor 特征
_scaling      [N,6]       前 3 维给 Anchor 网格尺度，后 3 维给解码尺度
_rotation     [N,4]       解码后的旋转输入
_opacity      [N,1]       Anchor opacity 初值；offset opacity 由 MLP 生成
```

初始化时：

1. 读取 COLMAP sparse point cloud；
2. 按 `voxel_size` 去重，默认 `0.001`；若 `voxel_size<=0`，用点间距离中位数；
3. Anchor 位置来自去重后的点；
4. offset 全零，Anchor feature 全零，rotation 初始化为单位四元数；
5. scaling 用 `log(sqrt(dist2))` 初始化并复制到 6 维；
6. 初始 opacity 是 `inverse_sigmoid(0.1)`；
7. `n_offsets=10`，feature dimension 默认 32。

`generate_neural_gaussians()` 对每个可见 Anchor：

- 用 `mlp_opacity(feat)` 输出 K 个 opacity logit，保留大于 0 的 offset；
- `mlp_color([feat, view direction, distance])` 输出每个 offset 的 RGB；
- `mlp_cov(feat)` 输出 3 个尺度 logit 和 4 个旋转参数；
- `scaling = anchor_scaling * sigmoid(scale_logits)`；
- `offsets = local_offset * anchor_scaling`；
- `xyz = anchor + offsets`；
- 最小尺度对应的旋转列向量被当作高斯法线。

因此这里的“高斯”是由 Anchor 内部解码的 offset 高斯；后 15k 冻结 Anchor 时，必须明确冻结 `_anchor` 的梯度或 optimizer group，同时仍允许 `_offset`、`_anchor_feat`、`mlp_*` 等内部参数更新。

### 4.3 单目 normal 约束与扁平高斯

对协方差分解 `Σ=R S S^T R^T`，论文把最小尺度轴当作高斯的局部法线：

\[
n_i=r_{k},\qquad k=\arg\min(s_{i1},s_{i2},s_{i3}).
\]

渲染器将每个高斯法线和 alpha 一起进行 alpha compositing，得到渲染法线 `N_hat`，再根据相机姿态转换到相机坐标，与 Omnidata 预测的单目法线 `N_bar` 比较：

\[
L_{normal}=\|\bar N-\hat N'\|_2^2+\|1-\bar N^T\hat N'\|_2^2.
\]

为使高斯变成薄片，论文给出尺度正则：

\[
L_{reg}=\|s_1\|_1+\left\|\frac{s_2}{s_3}+\frac{s_3}{s_2}-2\right\|_1,
\]

其中 `s1` 是最小尺度。第一项压小法线方向厚度；第二项防止另外两个尺度之一也趋近 0，避免 needle-like 高斯造成法线歧义。

本地训练代码把这项实现为：

```python
scaling_reg = scaling.min(dim=1)[0].mean() \
    + ((scaling.topk(2)[0] ** 2).sum(1)
       / scaling.topk(2)[0].prod(1) - 2).mean()
base_loss += 0.01 * scaling_reg
```

这正是后续“扁平高斯”分支最直接可复用的实现，但需要注意：GSRec 论文默认 `lambda_r=0.01`，当前代码将 `0.01` 直接写入 base loss，并不是 `opt.lambda_norm_reg` 的同名参数。

### 4.4 单目 depth 约束

渲染器以高斯中心在相机坐标系的 Z 值作为深度：

\[
\hat D=\sum_i z_i\alpha_i\prod_{j<i}(1-\alpha_j).
\]

Omnidata 深度通常是非度量的，所以论文用尺度和偏置对齐：

\[
L_{depth}=\|\bar D-(a\hat D+b)\|_2^2.
\]

本地实现调用 `ScaleAndShiftLoss(render_depth, gt_depth*50+0.5, mask)`，其中 `gt_depth*50+0.5` 是该数据预处理约定；复用到新数据时不能默认这个数值缩放正确，必须记录深度网络输出范围和有效 mask。

数据读取在 [`scene/dataset_readers.py`](D:/Projects/GEO/02_GSRec_Local_Structural_Hints/scene/dataset_readers.py)：每张图像旁边必须存在 `image_name_depth.npy`、`image_name_normal.npy`，可选 `image_name_mask.npy`；法线从 `[0,1]` 变到 `[-1,1]`。这说明 GSRec 官方输入不只是用户 pipeline 中的 RGB、位姿、COLMAP 点云，还要求额外单目 depth/normal 先验。

### 4.5 IMLS、RIMLS 与神经隐式网络

给一组带法线的高斯中心 `μ_l,n_l`，IMLS 在查询点 `x` 处定义：

\[
F_{IMLS}(x)=
\frac{\sum_l \theta_l(x)\langle x-\mu_l,n_l\rangle}
     {\sum_l\theta_l(x)},
\qquad \theta_l(x)=o_lG_l(x).
\]

它在局部窄带中近似表面 SDF 的零水平集。GSRec 用 Instant-NGP 风格多分辨率 hash grid + 两层 MLP 学习 `F_MLP`，但这个 MLP 不作为最终 mesh 表示，而是用于把局部 MLS 的几何梯度传播到高斯。

论文的联合损失：

\[
L_{joint}=\sum_{x,\mu_l}
\left(
|F_{IMLS}(x)-F_{MLP}(x)|^2+
\left\|\frac{\nabla F_{MLP}(\mu_l)}{\|\nabla F_{MLP}(\mu_l)\|}-n_l\right\|^2+
 (\|\nabla F_{MLP}(x)\|^2-1)^2
\right).
\]

第一项是零阶 MLS/SDF 值匹配，第二项是高斯法线和隐式梯度匹配，第三项是 eikonal 约束。RIMLS 在 `θ_l` 上再乘一个关于 `n_l` 与归一化 `∇F_MLP(x)` 差异的 1D Gaussian kernel：

\[
\phi(q)=\exp(-q^2/\sigma_n^2).
\]

当前代码的 `utils/general_utils.py::mls_sdf` 对每个查询点选择 `k_near` 个高斯：

1. 用 `cov3D` 的逆协方差计算 `exp(-0.5 offset^T Σ^{-1} offset)`；
2. 乘 opacity，并删除权重小于 `1e-12` 的项；
3. 只保留 `offset.norm()<offset_weight` 的局部邻域，默认训练脚本为 `fmls_sdf_offset=0.01`；
4. 可选地乘法线差异权重 `exp(-||n_p-n_q||²/normal_weight)`；
5. 归一化权重后，对 `(offset·normal_p)` 求加权平均得到 `f_mls`。

神经隐式网络 `implicit_sdf` 使用 16 层、每层 2 维的 hash grid 特征，`base_resolution=16`、`desired_resolution=2048`、`log2_hashmap_size=19`，再拼接 6 频率的 sin/cos positional encoding，后接两层 feature-dim MLP。最后一层使用 geometric initialization；`inside_out` 改变最后层权重/偏置符号。

### 4.6 GSRec 的两阶段训练

论文给出的阶段是：

\[
L_{stage1}=L_{color}+\lambda_dL_{depth}+\lambda_nL_{normal}+\lambda_rL_{reg},
\]
\[
L_{stage2}=L_{stage1}+\lambda_jL_{joint}.
\]

默认 `λ_d=0.1`、`λ_n=0.1`、`λ_r=0.01`、`λ_j=1`，总训练 30k；15k 后 Anchor 数量停止增长和裁剪，再加入联合 MLS/SDF 正则。代码入口 [`train.py`](D:/Projects/GEO/02_GSRec_Local_Structural_Hints/train.py) 的关键顺序是：

1. 随机抽一张训练相机；
2. `prefilter_voxel` 进行 Anchor 可见性筛选；
3. 渲染 RGB、depth、normal、median depth；
4. 计算 RGB L1/SSIM、depth、normal L1/cos/gradient 和尺度正则；
5. `iteration > sdf_start_iter` 时，用渲染 median depth 反投影采样点；
6. KD-tree 找每个查询点的 `k_near=50` 个高斯，计算 MLS 与 SDF MLP；
7. 计算渲染高斯点和随机点的 eikonal/normal consistency；
8. 反向传播后，在 `update_from=1500` 到 `update_until=15000` 之间按 100 步统计 offset 梯度并增删 Anchor；
9. 15k 时删除稠密化统计量，之后只优化现有参数和 MLP/SDF。

官方 `train_single.sh` 明确使用 `--iterations 30000`、`--sdf_start_iter 15000`、`--knear 50`、`--fmls_sdf_offset 0.01`、`--fmls_use_normal`、`--ps_depth 8` 和 `--sdf_inside_out`。

需要区分两个“冻结”概念：GSRec 论文/代码冻结的是 **Anchor 的数量**，并不冻结 Anchor 参数；现有 Anchor 的位置、offset、特征和 MLP 仍会优化。用户要求的后 15k “锚点冻结”更严格，应在新项目中冻结 `_anchor` 的梯度或从 optimizer 中移除 Anchor 参数，同时保留内部参数更新。

### 4.7 Mesh 输出和实验边界

GSRec 最终用优化后的 Gaussian means 和 normals 做 Poisson reconstruction；神经隐式网络只是正则化器。论文在 Replica（8 个场景）和 ScanNet 上评估 Chamfer、Normal Consistency 和 F-score，Replica F-score 阈值为 5 cm。论文报告 GSRec 的 Replica 平均 Chamfer 7.08、平均 F-score 67.22，高于 SuGaR 两种变体；这些数值依赖 Replica 的真实尺度和官方评测脚本，不能直接换算为 COLMAP 无标定单位。

## 5. GSPrior：论文与代码

### 5.1 方法动机

GSPrior 不依赖外部深度网络或预训练 SDF。它先用当前 3DGS 渲染多视图深度，使用 TSDF 融合得到距离场 `f^t`，再将这个距离场作为下一阶段高斯优化的“自约束先验”。随着训练继续，新的渲染深度更一致，于是 TSDF prior 也被周期性更新；同时逐步减小窄带宽度，使约束变紧。

### 5.2 TSDF prior 构建

论文中：

\[
d_i'=S(\{g_j\},\{z_{ji}\},p_i),
\qquad
f^t=F(\{d_i'(t)\}).
\]

`f^t` 是三维 TSDF 网格，零水平集是当前粗糙表面。网格值在 `[-1,1]` 截断；训练时只在距离表面的窄带内约束高斯。

官方代码的 TSDF 流程位于 [`utils/grid_fuser.py`](D:/Projects/GEO/12_GSPrior/utils/grid_fuser.py)、[`utils/fusion.py`](D:/Projects/GEO/12_GSPrior/utils/fusion.py)：

1. 对所有训练相机渲染 `plane_depth`；
2. 深度大于 `depth_trunc` 的像素置零；
3. 用渲染法线与视线夹角过滤大于 80° 的深度；
4. 第一遍估计所有视锥的世界坐标 bounds；
5. 以 `voxel_size` 初始化 TSDFVolume；
6. 第二遍把每张 depth 与相机内外参积分到 TSDF；
7. Marching Cubes 提取 `gsprior_<iter>.ply`；
8. 保存 `tsdf_<iter>.npy`、`vol_origin_<iter>.npy`、`voxel_size_<iter>.npy`，再加载给高斯模型。

代码中的 `TrilinearInterpolation` 把高斯中心先变到网格坐标，再归一化到 `[-1,1]` 做三线性插值；在越界处 clamp 到网格边界。因此 TSDF 的单位是当前场景坐标单位，不能在没有尺度标定时写成米或厘米。

### 5.3 三类高斯约束

#### (1) 删除窄带外的高斯

对第 `j` 个高斯中心插值 `s_j=f^t(μ_j)`。若插值结果为 TSDF 边界 `+1` 或 `-1`，则认为高斯位于窄带外并删除。论文把它描述为压缩高斯分布，减少远离表面的 outlier 对深度的负面影响。

#### (2) 透明度约束

在窄带中按 `δ^t` 分成：

\[
N_{on}=\{g_j:|s_j|\le\delta^t\},
\qquad
N_{off}=\{g_j:\delta^t<|s_j|\le1\}.
\]

距离权重：

\[
\epsilon_j=\frac1{(1+|s_j|)^2}.
\]

透明度损失：

\[
L_{SCP}=\frac1M\left[
\sum_{g_k\in N_{on}}\epsilon_k(o_k-1)^2+
\sum_{g_{k'}\in N_{off}}\epsilon_{k'}o_{k'}^2
\right].
\]

代码在 [`scene/gaussian_model.py`](D:/Projects/GEO/12_GSPrior/scene/gaussian_model.py) 的 `get_opacity()` 中完成 TSDF 插值；训练循环在 TSDF 阶段把 `|s|<=0.3` 的点视为表面，并用 `1e-3 * loss_surface + 1e-3 * loss_free` 加入总损失。这里的 `0.3` 是当前代码的 band/δ 实现值，不能误写成论文固定的普适常数。

#### (3) 把高斯推向表面

TSDF 网格使用中心差分得到梯度：

\[
\nabla f^t(\mu_j)=\frac1{2\epsilon}
\begin{bmatrix}
f(x+\epsilon,y,z)-f(x-\epsilon,y,z)\\
f(x,y+\epsilon,z)-f(x,y-\epsilon,z)\\
f(x,y,z+\epsilon)-f(x,y,z-\epsilon)
\end{bmatrix},
\]

然后：

\[
\mu_j\leftarrow\mu_j-s_j\nabla f^t(\mu_j).
\]

论文强调这不是每一步通过 loss 反复 pull，而是在稠密化后作为位置更新，从而稳定训练。官方仓库的 `scene/gaussian_model.py` 保留了与 TSDF 插值相关的删除/位置逻辑；训练默认前 15k 是普通 GS 稠密化，后续才开启 prior 分支。新项目如果要求“15k 后完全禁止新增/删除 Anchor”，应把 GSPrior 的删除操作改为仅产生 mask/诊断，不能直接调用 `prune_points()` 删除 Anchor。

需要特别记录当前代码和论文调度之间的差异：本地 `densify_and_prune()` 在 `iteration>30000` 且 `gs_state=False` 时，才会每 1000 步用 TSDF 边界值删除一部分 inactive Gaussian，并以有限差分梯度做很小的 `1e-7` 位置修正；在更早的 prior 阶段仍会走 clone/split 分支。论文的文字描述是“每次 densification 后先 pull 再 remove”，但仓库代码把这项操作放到了 30k 之后的特殊分支。因此新项目不能只按论文图示实现，必须先决定采用论文调度还是复现仓库调度，并在配置中显式记录。

### 5.4 Planar Gaussian 和损失

论文称使用 planar Gaussians，以提高几何表达。损失包括：

- `L_RGB`：RGB 的 MAE、SSIM，以及跨视图 NCC；
- `L_Depth`：同一条光线上高斯交点深度的加权方差，令深度分布变薄：

  \[
  L_{Depth}=\sum_{u=0}^{U-1}\sum_{u'=0}^{u-1}
  \omega_u\omega_{u'}(\rho_u-\rho_{u'})^2;
  \]

- `L_NS`：渲染法线和由 depth 求导得到的法线之间的加权 L1；
- `L_NM`：以参考视图平面法线/距离构造单应矩阵，在邻视图进行几何一致性约束；
- `L_SCP`：TSDF prior 的 opacity 约束。

总体为：

\[
L=L_{RGB}+\lambda_1L_{Depth}+\lambda_2L_{NS}+\lambda_3L_{NM}+\lambda_4L_{SCP},
\]

论文给出 `λ1=0.01`、`λ2=0.1`、`λ3=0.1`、`λ4=0.01`。代码 [`train.py`](D:/Projects/GEO/12_GSPrior/train.py) 的实际训练循环更容易复现：

1. RGB L1 + DSSIM；
2. `single_view_weight_from_iter=7000` 后加入 depth-normal 与渲染 normal 的一致性，默认权重 0.015；
3. 进入 prior 状态后，加入 surface/off-surface opacity loss；
4. `multi_view_weight_from_iter=7000` 后抽相邻相机或虚拟相机，进行深度重投影几何 loss；
5. 在有效区域抽 patch，通过平面单应性 warp 到邻视图，计算 LNCC/NCC；
6. 反向传播、稠密化/裁剪和 optimizer step。

### 5.5 官方代码的训练阶段与默认参数

个人主页 README 的快速启动是：

```bash
python train.py -s <path_to_scene> -m <path_to_output>
python render.py -s <dataset_path>/<scene_name> -m <output_path>/<scene_name>
```

默认训练迭代是 40k；默认 `densify_from_iter=500`、`densify_until_iter=15000`、普通稠密化每 500 步、GS 变体每 300 步、opacity reset 每 3000 步；默认 TSDF 更新点为 `[20000,30000,40000]`，对应截断值 `[96,48,24]`，`tsdf_voxel_size=0.005`、`tsdf_depth_trunc=5.0`。

`gs_mode_swither()` 在达到第一个 TSDF 更新点后切换到 prior 状态。每次更新时，代码先运行 `run_tsdf_fusion()`，保存 TSDF 和 mesh，再调用 `gaussians.set_tsdf()`。恢复 checkpoint 时，会加载最近一个已完成的 TSDF 文件，保证断点继续训练时 prior 不丢失。

GSPrior 使用 `diff-plane-rasterization`；渲染器输出 `plane_depth`、`rendered_normal`、`rendered_distance` 和 alpha。`render_normal()` 通过 depth 的空间梯度计算参考法线。高斯法线是最小尺度轴，并根据观察方向翻转，使法线朝向相机。

### 5.6 论文实验与限制

论文在 NeRF-Synthetic、DTU、Tanks and Temples 和 Mip-NeRF 360 上评估：分别使用 L1 Chamfer/PSNR、CD/PSNR、F-score 和 PSNR/SSIM/LPIPS；表面统一使用 TSDF Fusion 提取。论文强调不需要学习额外 SDF 或外部 prior，在几何精度和速度之间取得平衡。

当前本地 `12_GSPrior` 的 Shipwreck/DTU 实验对场景尺度做过适配：某些 COLMAP 导出使用大尺度，因此运行脚本会改用很大的 TSDF voxel/depth trunc。评估中使用的 uncalibrated COLMAP 单位不能标成真实厘米；这属于本地实验协议，和论文 DTU 官方尺度不能混写。

## 6. 三个方法的代码级差异

| 维度 | WaterSplatting | GSRec | GSPrior |
|---|---|---|---|
| 主要表示 | 高斯物体 + 每像素一次查询的方向条件介质 | Scaffold Anchor + offset Gaussian + neural SDF 正则 | 3D/planar Gaussian + 由渲染深度融合的 TSDF |
| 主要几何约束 | 间接来自 RGB/介质分离和 alpha 深度 | 单目 depth、normal、扁平尺度、IMLS/RIMLS、eikonal | TSDF 距离带、opacity、删除、位置投影、跨视图几何 |
| 是否学习 SDF | 否 | 是，但只作联合正则器 | 否，使用自融合 TSDF 先验 |
| 表面输出 | 代码可输出 depth/clear/object/medium；论文重点是 NVS | Gaussian means + normals 做 Poisson | TSDF Marching Cubes mesh |
| Anchor 增长/裁剪 | 普通高斯 split/duplicate/prune | 15k 前 Scaffold Anchor 增删，之后数量固定 | 普通 GS 稠密化到 15k；prior 阶段还可能按 TSDF prune，需为用户需求改写 |
| 介质分支 | 必须有：RGB、backscatter、attenuation | 无 | 无 |
| 外部先验 | COLMAP，论文中水下白平衡/位姿 | COLMAP + Omnidata depth/normal | 只用当前高斯渲染 depth；不需要外部几何 prior |

## 7. 对用户 15k + 15k pipeline 的直接落地解释

### 7.1 初始化阶段

输入应整理为：

```text
scene/
├── images/              RGB 图像
├── sparse/0/            cameras.bin、images.bin、points3D.bin
├── depth/               可选的单目 depth（GSRec 阶段需要）
├── normal/              可选的单目 normal（GSRec 阶段需要）
└── split.json           明确 train/test，避免训练测试泄漏
```

初始化逻辑建议采用 GSRec 的 `create_from_pcd()`：COLMAP 点按 voxel 去重成 Anchor；每个 Anchor 建立 K 个 offset；初始 offset 为 0，特征为零；尺度由邻域距离决定。WaterSplatting 的普通高斯初始化可以作为对照实现，但主项目应保留 Anchor 骨架。

### 7.2 前 15k：Scaffold-GS/GSRec + 扁平高斯 + 水分支

前 15k 的总目标可以写成：

\[
L_{0:15k}=L_{water-photo}+\lambda_dL_{depth}
 +\lambda_nL_{normal}+\lambda_{flat}L_{flat}
 +\lambda_{anchor}L_{anchor\text{-}reg}.
\]

其中：

- `L_water-photo` 使用 WaterSplatting 的 `rgb_object + rgb_medium` 与观测 RGB 比较；
- `L_depth`、`L_normal` 和 `L_flat` 采用 GSRec；
- Water MLP 输出 9 维，至少保持 RGB/bs/attn 的物理范围；
- Anchor 在 15k 前允许 Scaffold 式增删，但每次增删要同步 optimizer state、Anchor feature、offset、opacity 和 Water MLP 状态；
- `L_flat` 直接采用 GSRec 的最小尺度 + 两大尺度 harmonic-like 正则，防止针状高斯；
- 所有水下实验必须分别保存 `rgb`, `rgb_clear`, `rgb_medium`, `medium_rgb`, `medium_bs`, `medium_attn`，否则无法判断水分支是否吸收了几何。

### 7.3 后 15k：Anchor 数量冻结 + GSPrior

从 15k 开始应执行一次不可逆的结构切换：

1. 停止 Anchor growing；
2. 停止 Anchor pruning；
3. 固定 Anchor 的数量，建议将 `_anchor.requires_grad_(False)` 并从 optimizer 的 Anchor group 移除；
4. 允许 `_offset`、`_anchor_feat`、offset decoder、颜色 decoder、opacity decoder、Water MLP 和必要的尺度/旋转参数继续优化；
5. 用当前渲染 depth 做 TSDF Fusion；
6. 对高斯中心三线性插值 TSDF；
7. 使用 GSPrior 的 opacity surface/free 约束；
8. 使用 TSDF 梯度做位置投影时，只更新 offset/内部位置参数，不直接改 Anchor；
9. 以固定 Anchor 集合继续渲染、更新 TSDF 和优化水体参数。

这和论文 GSPrior 的默认行为有一个关键差别：GSPrior 官方实现的 prior 阶段仍保留一部分标准 Gaussian densification/pruning 逻辑；用户的需求要求冻结 Anchor，因此新项目必须在 `AnchorDensifier` 层增加硬开关，不能直接照搬 `prune_points()`。

### 7.4 水体支路的稳定性要求

WaterSplatting 的 `medium_rgb/bs/attn` 与物体颜色、深度存在可辨识性耦合。新项目至少要记录并考虑：

- `medium_rgb` 用 Sigmoid，范围为 `[0,1]`；
- `medium_bs/attn` 用 Softplus，必要时设置上限或正则；
- 对方向编码使用固定 SH 等级，避免 MLP 用高频方向变化拟合纹理；
- 对 `medium_bs/attn` 增加空间/方向平滑或低频先验；
- 先冻结几何或先 warm-up 水分支，再联合优化，防止水参数替代高斯几何；
- 训练和测试严格使用相同白平衡、COLMAP 畸变校正和 split；
- 评测中同时报告外观（PSNR/SSIM/LPIPS）和几何（Chamfer/F-score/normal consistency），不能只用 RGB 指标声称表面重建改善。

## 8. 建议的新项目模块接口

后续在 `AnchorWaterGSPrior` 中建议保持以下边界，便于逐项消融：

```text
anchor_scene/
  colmap_loader.py       # 相机、稀疏点、split
  anchor_model.py        # Anchor/offset/decoder 与冻结接口
  flat_gaussian_loss.py  # GSRec 尺度正则
  geometry_priors.py     # depth/normal/MLS，可选
  water_field.py         # SH + MLP，输出 rgb/bs/attn
  water_renderer.py      # 高斯项 + 介质区间解析积分
  tsdf_prior.py          # 深度融合、插值、梯度、band
  train.py               # 0-15k、15k-30k 状态机
  evaluate.py            # RGB、depth、mesh、几何指标
```

建议让渲染器返回一个字典，而不是只返回一张 RGB 图：

```python
{
    "rgb": rgb_object + rgb_medium,
    "rgb_clear": rgb_clear,
    "rgb_object": rgb_object,
    "rgb_medium": rgb_medium,
    "depth": depth,
    "alpha": alpha,
    "medium_rgb": medium_rgb,
    "medium_bs": medium_bs,
    "medium_attn": medium_attn,
    "gaussian_xyz": xyz,
    "gaussian_normal": normal,
    "anchor_ids": anchor_ids,
}
```

这样可以独立计算 WaterSplatting 的外观损失、GSRec 的几何损失和 GSPrior 的 TSDF 约束，也能在消融实验中关闭某一分支而不改变其余渲染路径。

## 9. 已核实与仍需在新项目中验证的事项

### 已核实

- 三篇 PDF 的全文已提取并逐段核对方法、公式、训练阶段和实验设置。
- 三个个人主页官方仓库已在 GitHub 浏览器打开并核对 fork 来源、main 分支和 README/目录。
- WaterSplatting 的自定义 CUDA forward/backward、方向 MLP 和 loss 入口已检查。
- GSRec 的 Anchor 初始化、offset 解码、最小尺度法线、depth/normal/flat loss、MLS/SDF 联合优化和 15k densification cutoff 已检查。
- GSPrior 的 TSDF 两遍融合、三线性插值、band/opacity 约束、渲染 depth-normal、跨视图几何和训练阶段切换已检查。

### 新项目开始前必须做的验证

- 对三个个人 fork 固定 commit SHA，并在新仓库 README 中记录许可证、上游链接和修改来源；
- 将官方代码、实验修复和新代码分目录，避免把 `outputs`、`.pyd`、缓存和数据推入 GitHub；
- 在同一个小场景上先验证 `rgb_clear + rgb_medium == rgb`、alpha/depth 范围和梯度；
- 验证 15k 切换后 Anchor 数量、Anchor 梯度、optimizer 参数组确实不再变化；
- 验证 TSDF 插值的坐标轴顺序、`vol_origin`、`voxel_size` 和场景尺度；
- 对水下数据固定 train/test split，记录介质 MLP 是否只在训练图像上优化；
- 先做 Water-only、GSRec-only、GSPrior-only 和三者联合四组最小消融，再进行长训练。

## 10. 参考资料与源码入口

### 论文

- [WaterSplatting.pdf](D:/Paper/水下/WaterSplatting.pdf)，Li et al., 3DV 2025，arXiv:2408.08206。
- [Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints.pdf](<D:/Paper/Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints.pdf>)，Wu, Zheng, Cai, ECCV 2024。
- [2026_CVPR_Self-constrained Priors.pdf](<D:/Paper/surface/surface/2026_CVPR_Self-constrained Priors.pdf>)，Noda, Liu, Han，CVPR 2026 / arXiv:2603.19682。

### 核心源码

- [WaterSplatting model](D:/Projects/GEO/09_WaterSplatting_Shipwreck/water_splatting/water_splatting.py)
- [WaterSplatting rasterizer binding](D:/Projects/GEO/09_WaterSplatting_Shipwreck/water_splatting/rasterize.py)
- [WaterSplatting CUDA forward](D:/Projects/GEO/09_WaterSplatting_Shipwreck/water_splatting/cuda/csrc/forward.cu)
- [GSRec training](D:/Projects/GEO/02_GSRec_Local_Structural_Hints/train.py)
- [GSRec Anchor/SDF model](D:/Projects/GEO/02_GSRec_Local_Structural_Hints/scene/gaussian_model_implicit.py)
- [GSRec renderer](D:/Projects/GEO/02_GSRec_Local_Structural_Hints/gaussian_renderer/__init__.py)
- [GSRec MLS implementation](D:/Projects/GEO/02_GSRec_Local_Structural_Hints/utils/general_utils.py)
- [GSPrior training](D:/Projects/GEO/12_GSPrior/train.py)
- [GSPrior Gaussian model](D:/Projects/GEO/12_GSPrior/scene/gaussian_model.py)
- [GSPrior renderer](D:/Projects/GEO/12_GSPrior/gaussian_renderer/__init__.py)
- [GSPrior TSDF fuser](D:/Projects/GEO/12_GSPrior/utils/grid_fuser.py)
- [GSPrior TSDF grid interpolation](D:/Projects/GEO/12_GSPrior/utils/grid_sdf.py)

# AnchorWaterGSPrior

基于三篇官方实现的几何优先水下三维重建原型：

1. 输入 RGB 图像、COLMAP 相机位姿和稀疏点云；
2. 用 Scaffold-GS/GSRec 风格的 Anchor + Offset Gaussian 初始化；
3. 前 `15k` 步允许 Anchor densification，并使用扁平高斯约束；
4. 从 `15k` 步开始冻结 Anchor 数量和 Anchor 中心，只更新 Anchor 内部参数；
5. 用 GSPrior 风格的自渲染深度 TSDF 先验约束高斯；
6. 用独立的方向条件 Water MLP 学习介质颜色、后向散射和衰减。

这是一个纯 PyTorch 的可运行基线。它默认使用可微的点投影 splat，避免依赖三篇官方仓库中的 CUDA 扩展；安装对应环境后可替换 `anchor_water_gsprior/renderer.py` 为 CUDA rasterizer。

## 快速开始

```powershell
D:\App\Miniconda3\envs\3D-UIR\python.exe train.py `
  --data D:\Projects\Dataset\my\Shipwreck `
  --output outputs\shipwreck_smoke `
  --iterations 20 `
  --phase1 10 `
  --image-size 320 `
  --max-anchors 2048
```

完整默认流程为 `30000` 步，前 `15000` 步为 Scaffold/GSRec 阶段，后 `15000` 步启用 Anchor 冻结和 TSDF prior：

```powershell
D:\App\Miniconda3\envs\3D-UIR\python.exe train.py \
  --data D:\Projects\Dataset\my\Shipwreck \
  --output outputs\shipwreck_30k
```

`--phase1` 可用于小规模断点/单元验证；它不改变正式 pipeline 的默认值。训练过程中每隔 `--log-every` 步写入 `metrics.jsonl`，不使用高频监控。

## 输入格式

支持 3DGS/Colmap 目录：

```text
scene/
├── images/              # RGB 图像
├── sparse/0/            # cameras.bin/images.bin/points3D.bin 或 .txt/.ply
└── ground_truth/        # 可选，用于几何指标
```

SeaThru-NeRF 的 `images_wb` 会优先于 `images`，相机和点云从 `sparse/0` 读取。训练/测试按 COLMAP 图像顺序切分，结果中明确记录是否存在几何 GT。

## 输出

- `checkpoints/step_*.pt`：模型和状态机状态；
- `renders/`：RGB、clear/object/medium、depth；
- `metrics.jsonl`：训练损失、Anchor 数量、冻结状态和显存信息；
- `results.json`：PSNR/SSIM/LPIPS（若安装 torchmetrics）和几何 Chamfer/F-score（若提供 GT）；
- `anchors_final.ply`：最终 Anchor 中心和颜色。

## 代码边界

- `colmap.py`：读取相机、位姿和稀疏点云；
- `model.py`：Anchor、Offset Gaussian、扁平约束和 Water MLP；
- `renderer.py`：可微投影和物体/介质合成；
- `tsdf.py`：渲染深度驱动的轻量 TSDF prior；
- `train.py`：15k 状态切换、冻结验证、保存和评估。

论文与官方代码的逐项中文记录见 [`THREE_PAPERS_CODE_READING_CN.md`](THREE_PAPERS_CODE_READING_CN.md)。

本实现是新的实验代码，官方仓库仅作为方法参考；训练结果不能直接声称复现论文指标。COLMAP 未标定场景的几何指标使用场景坐标单位，不能直接写成厘米或米。

当前本机 smoke test 和指标口径见 [`RESULTS.md`](RESULTS.md)。

## 诊断与修复记录

`diagnose_pipeline.py` 可以单独检查 COLMAP 点云的相机投影覆盖率，并在给定 checkpoint 时检查渲染输出的数值范围。当前版本修复了三处会导致“空渲染/均匀暗图”的问题：

1. COLMAP 点云会做场景归一化；相机平移也同步应用相同的相似变换，投影坐标现在与原始 COLMAP 世界坐标一致。
2. 在没有 `plyfile` 时增加了二进制 little-endian PLY 读取器，因此 Shipwreck 的 `points3D.ply` 不会静默失效。
3. 渲染器改用按深度排序的前向 alpha 合成，并对视野外/相机后方点做筛选；不再把每条射线的高斯权重强制归一化为 alpha=1。

训练脚本现在显式划分 train/test，训练循环只访问 train frames，最终指标只渲染 test frames，并在 `results.json` 中记录帧列表和交集检查。

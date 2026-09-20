# 验证结果

本文件记录当前原型在本机数据上的可复核验证。命令使用项目环境中的 Python：

```powershell
D:\App\Miniconda3\envs\3D-UIR\python.exe train.py `
  --data D:\Projects\Dataset\my\Shipwreck `
  --output outputs\smoke_shipwreck_v4 `
  --iterations 12 --phase1 6 --image-size 160 --max-anchors 128

D:\App\Miniconda3\envs\3D-UIR\python.exe train.py `
  --data D:\Projects\Dataset\SeathruNeRF_dataset\Curasao `
  --output outputs\smoke_curasao_v4 `
  --iterations 12 --phase1 6 --image-size 160 --max-anchors 128
```

两次运行均在 CUDA 上完成，均经过 phase 1 到 phase 2 的切换；phase 2 日志中的 `anchor_frozen=true` 且 Anchor 数量保持 128。每个结果目录中包含 `metrics.jsonl`、checkpoint、5 个渲染视图和 `results.json`。

| 数据集 | 步数 | 评估视图 | PSNR | SSIM | LPIPS | Chamfer | F-score | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Shipwreck | 12 | 5 | 6.7431 | 0.0042 | 未计算 | 0.1183 | 0.1324 | 几何值为归一化场景坐标；GT 为 `ground_truth/point_cloud.ply` |
| SeaThru-NeRF Curasao | 12 | 5 | 13.4382 | 0.0000 | 未计算 | 不可用 | 不可用 | 当前目录没有可核验几何 GT |

## 指标口径

- PSNR 和 SSIM 是训练结束后前 5 个 COLMAP 视图的平均值，图像按 `--image-size` 缩放。
- 当前实现没有强制下载 LPIPS 预训练权重，因此 `results.json` 将 LPIPS 写为 `null`，不会伪造数值。
- Shipwreck 的 Chamfer/F-score 使用预测 Anchor 中心与 GT 点云，两个点云分别做中位数中心和 98% 半径归一化后计算，阈值为 0.05。COLMAP 没有绝对尺度，所以这些几何值不是米或厘米。
- 12 步运行是结构 smoke test，用于验证数据读取、可微渲染、Water MLP、扁平约束、15k 状态机接口和冻结逻辑。它不是论文级 30k 复现，也不应与论文表格直接比较。

## 额外结构测试

`AnchorWaterModel.densify()` 已用 4 个 Anchor 的独立测试验证：可按 opacity 复制 Anchor、offset、feature、颜色、scale 和 opacity；冻结后返回 0，不再增加 Anchor。正式训练脚本只在 phase 1 调用 densification，phase 2 不再调用。

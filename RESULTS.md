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
| Shipwreck | 500 | 5 | 19.9580 | 0.0377 | 未计算 | 0.1226 | 0.0946 | CUDA；Anchor=128；`outputs/validation_shipwreck_500` |
| SeaThru-NeRF Curasao | 500 | 5 | 19.7126 | 0.1483 | 未计算 | 不可用 | 不可用 | CUDA；Anchor=128；`outputs/validation_curasao_500` |
| SeaThru-NeRF Curasao | 30000 | 5 | 20.1106 | 0.2548 | 0.7708 | 不可用 | 不可用 | 完整 30k；Anchor=8192；`outputs/curasao_full_30k` |
| Shipwreck | 30000 | 5 | 20.6565 | 0.0376 | 0.7465 | 0.0928 | 0.3709 | 完整 30k；Anchor=8192；`outputs/shipwreck_full_30k` |

## 指标口径

- PSNR 和 SSIM 是训练结束后前 5 个 COLMAP 视图的平均值，图像按 `--image-size` 缩放。
- LPIPS 使用本机已缓存的 AlexNet 权重，对训练结束导出的前 5 个视图取平均；若运行环境没有 `lpips` 或权重，结果会安全地写为 `null` 并保留原因。
- Shipwreck 的 Chamfer/F-score 使用预测 Anchor 中心与 GT 点云，两个点云分别做中位数中心和 98% 半径归一化后计算，阈值为 0.05。COLMAP 没有绝对尺度，所以这些几何值不是米或厘米。
- 12 步运行是结构 smoke test，用于验证数据读取、可微渲染、Water MLP、扁平约束、15k 状态机接口和冻结逻辑。它不是论文级 30k 复现，也不应与论文表格直接比较。
- 500 步运行是较大规模流程验证，仍不是论文级 30k 复现。当前逐点 Python/PyTorch rasterizer 的实测速度约为 Shipwreck 0.31 秒/步、Curasao 0.18 秒/步；按 30k 步线性估算约需 2.6 小时和 1.5 小时，正式长跑前应确认机器可接受该时长。
- 两套完整 30k 运行均使用 CUDA、640 缩放、8192 个初始 Anchor、前后各 15000 步。Curasao 实际耗时约 6815 秒（约 1 小时 53 分），Shipwreck 约 11055 秒（约 3 小时 4 分）。
- 两套完整运行均在第 15001 步切换到 phase 2，日志显示 `anchor_frozen=true` 且 Anchor 数量保持 8192；完整运行中 `densify_events=0`，因为初始稀疏点云经 `--max-anchors 8192` 后已达到上限。
- 完整结果仍属于本项目原型的实验结果，不等同于三篇论文的官方复现。渲染指标只统计训练结束时导出的前 5 个视图。

## 额外结构测试

`AnchorWaterModel.densify()` 已用 4 个 Anchor 的独立测试验证：可按 opacity 复制 Anchor、offset、feature、颜色、scale 和 opacity；冻结后返回 0，不再增加 Anchor。正式训练脚本只在 phase 1 调用 densification，phase 2 不再调用。

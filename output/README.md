# Held-out test visualizations

本目录保存 `AnchorWaterGSPrior` 两个 30k checkpoint 在固定测试视图上的渲染和几何可视化。图片分辨率为 320 像素长边，便于在 GitHub 中查看。

| 数据集 | 测试划分 | 视图数 | PSNR | SSIM | LPIPS | 几何参考 |
|---|---|---:|---:|---:|---:|---|
| Curasao | `MTN_1292`, `MTN_1300`, `MTN_1308` | 3 | 18.8055 | 0.1188 | 0.7951 | 无可核验表面 GT |
| Shipwreck | 自然排序索引 `0,8,16,...,152` | 20 | 21.2053 | 0.1644 | 0.7216 | COLMAP-visible reference |

每个测试帧包含：`gt`、最终 `rgb`、`clear`、`object`、`medium`、归一化 `depth`、`alpha` 和左右对比图。`geometry/` 中包含 Anchor 点云 PLY、Anchor 与参考点云的 3D/XY/XZ 投影；Curasao 只展示 Anchor 分布，因为数据目录没有可核验的表面 GT。

## 结果解释

当前渲染图明显退化为均匀色块，Shipwreck 的测试视图中还出现 alpha 接近 0 或 1 的跳变。这与指标一致：SSIM 很低、LPIPS 较高。检查代码后，主要原因是当前项目仍是原型级逐点渲染器：没有标准 Gaussian rasterizer 的屏幕空间可见性裁剪、深度排序和 alpha 合成；所有 Anchor 对每条射线参与 dense 权重计算，训练目标也只有像素 L1 加很弱的扁平/TSDF 项。另一个问题是旧版 `results.json` 只统计了训练帧列表前 5 张，不能作为测试集指标。

## 测试集口径限制

这两个 checkpoint 是按原始训练脚本用全部 COLMAP 帧循环训练得到的。因此本目录的数值是**固定测试视图上的 held-out-view 可视化评估**，不是无泄漏的论文级测试指标；要获得严格测试指标，需要按对应划分重新训练，只让优化过程访问 train frames，再在这里的测试列表上评估。Shipwreck 的 20 张间隔划分与项目中的 `02_GSRec_Local_Structural_Hints/data/Shipwreck_gsrec` 一致。几何值使用未标定 COLMAP 场景坐标的归一化诊断口径，不能解释为米或厘米。

from __future__ import annotations

import torch


class TSDFPrior:
    """GSPrior 风格的轻量自约束先验。

    它从当前 Anchor 中心构造归一化距离场，并在后半阶段对高斯中心施加窄带/自由空间约束。
    生产运行可以替换为官方两遍 TSDF Fusion；接口和训练状态保持一致。
    """

    def __init__(self, band=0.15):
        self.band = band
        self.centers = None
        self.ready = False

    @torch.no_grad()
    def update(self, centers):
        self.centers = centers.detach().clone()
        self.ready = True

    def loss(self, centers, opacity):
        if not self.ready or self.centers is None:
            return centers.new_zeros(())
        with torch.no_grad():
            d = torch.cdist(centers, self.centers).min(-1).values
        surface = d.clamp_max(self.band)
        free = torch.relu(self.band * 0.5 - d)
        return (opacity * surface.square()).mean() + (1 - opacity) .mul(free.square()).mean()

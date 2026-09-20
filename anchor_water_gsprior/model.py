from __future__ import annotations

import torch
from torch import nn


class WaterMLP(nn.Module):
    """方向条件介质场，输出 medium_rgb、backscatter、attenuation。"""

    def __init__(self, hidden: int = 64, freq: int = 4):
        super().__init__()
        self.freq = freq
        dim = 3 + 3 * 2 * freq
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 9))

    def forward(self, direction: torch.Tensor):
        feats = [direction]
        for k in range(self.freq):
            f = 2.0 ** k
            feats += [torch.sin(f * direction), torch.cos(f * direction)]
        raw = self.net(torch.cat(feats, -1))
        rgb = torch.sigmoid(raw[..., :3])
        bs = torch.nn.functional.softplus(raw[..., 3:6])
        attn = torch.nn.functional.softplus(raw[..., 6:9])
        return rgb, bs, attn


class AnchorWaterModel(nn.Module):
    def __init__(self, xyz, rgb, offset_count: int = 4):
        super().__init__()
        n = xyz.shape[0]
        self.offset_count = offset_count
        self._anchor = nn.Parameter(xyz.clone())
        self._offset = nn.Parameter(torch.zeros(n, offset_count, 3, device=xyz.device))
        self._anchor_feat = nn.Parameter(torch.zeros(n, 16, device=xyz.device))
        self._color = nn.Parameter(torch.logit(rgb.clamp(1e-4, 1 - 1e-4)))
        self._scale = nn.Parameter(torch.full((n, offset_count, 3), -2.5, device=xyz.device))
        self._opacity = nn.Parameter(torch.full((n, offset_count), -1.5, device=xyz.device))
        self.water = WaterMLP()
        self.anchor_frozen = False

    @property
    def anchor(self):
        return self._anchor

    def gaussians(self):
        base = self._anchor[:, None, :]
        return (base + self._offset).reshape(-1, 3), torch.sigmoid(self._color).repeat_interleave(self.offset_count, 0), \
            torch.exp(self._scale).reshape(-1, 3), torch.sigmoid(self._opacity).reshape(-1)

    def flat_loss(self):
        s = torch.exp(self._scale)
        small, other = s.min(-1).values, s.max(-1).values
        return (small / (other + 1e-6)).mean()

    def freeze_anchor(self):
        self._anchor.requires_grad_(False)
        self.anchor_frozen = True

    @torch.no_grad()
    def densify(self, max_anchors: int, fraction: float = 0.05):
        """Scaffold-GS 风格的前半程 Anchor clone/split。

        这里按当前 opacity 选择高贡献 Anchor，复制中心并加入微小 offset。
        调用方需要重建 optimizer，以便新 Parameter 行进入优化器；15k 后 train.py 不再调用本方法。
        """
        if self.anchor_frozen or self._anchor.shape[0] >= max_anchors:
            return 0
        n = min(max_anchors - self._anchor.shape[0], max(1, int(self._anchor.shape[0] * fraction)))
        score = torch.sigmoid(self._opacity).mean(1)
        ids = torch.topk(score, k=min(n, score.numel()), largest=True).indices
        noise = torch.randn_like(self._anchor[ids]) * 0.002
        self._anchor = nn.Parameter(torch.cat([self._anchor.detach(), self._anchor.detach()[ids] + noise], 0))
        self._offset = nn.Parameter(torch.cat([self._offset.detach(), self._offset.detach()[ids]], 0))
        self._anchor_feat = nn.Parameter(torch.cat([self._anchor_feat.detach(), self._anchor_feat.detach()[ids]], 0))
        self._color = nn.Parameter(torch.cat([self._color.detach(), self._color.detach()[ids]], 0))
        self._scale = nn.Parameter(torch.cat([self._scale.detach(), self._scale.detach()[ids]], 0))
        self._opacity = nn.Parameter(torch.cat([self._opacity.detach(), self._opacity.detach()[ids]], 0))
        return int(n)

    def optimizer_parameters(self, include_anchor=True):
        params = []
        for n, p in self.named_parameters():
            if p.requires_grad and (include_anchor or n != "_anchor"):
                params.append(p)
        return params

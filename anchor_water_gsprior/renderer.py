from __future__ import annotations

import torch


def project(xyz, frame, device, image_size=320):
    h0, w0 = frame.camera.height, frame.camera.width
    scale = min(1.0, float(image_size) / max(h0, w0))
    q = torch.as_tensor(frame.qvec, device=device, dtype=xyz.dtype)
    q = q / (torch.linalg.vector_norm(q) + 1e-12)
    qw, qx, qy, qz = q.unbind()
    r = torch.stack((
        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
        2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
        2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy),
    )).reshape(3, 3)
    t = torch.as_tensor(frame.tvec, device=device, dtype=xyz.dtype)
    cam = xyz @ r.T + t
    z = cam[:, 2].clamp_min(1e-3)
    u = (frame.camera.fx * scale * cam[:, 0] / z + frame.camera.cx * scale)
    v = (frame.camera.fy * scale * cam[:, 1] / z + frame.camera.cy * scale)
    return torch.stack([u, v, z], -1), int(round(h0 * scale)), int(round(w0 * scale))


def _render_chunk(model, frame, uv, xyz, color, scale, opacity, p, h, w):
    delta = uv[:, None, :] - p[None, :, :2]
    sigma = (scale[:, :2].mean(-1).clamp_min(1e-3) * max(h, w)).detach() + 1.0
    weights = torch.exp(-0.5 * (delta.square().sum(-1) / sigma[None].square())) * opacity[None]
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)
    rgb_object = weights @ color
    depth = (weights * p[None, :, 2]).sum(-1)
    direction = torch.zeros_like(xyz)
    direction[:, 2] = -1.0
    medium_rgb, medium_bs, medium_attn = model.water(direction.mean(0, keepdim=True).expand(len(uv), -1))
    medium = medium_rgb * (1.0 - torch.exp(-medium_attn * depth[:, None])) + medium_bs * 0.05
    alpha = weights.sum(-1).clamp(0, 1)
    rgb_clear = rgb_object
    rgb = rgb_clear * alpha[:, None] + medium * (1 - alpha[:, None])
    return {"rgb": rgb.clamp(0, 1), "rgb_object": rgb_object.clamp(0, 1), "rgb_clear": rgb_clear.clamp(0, 1),
            "rgb_medium": medium.clamp(0, 1), "depth": depth, "alpha": alpha,
            "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}


def render(model, frame, rays_uv, image_size=320, ray_chunk=512):
    xyz, color, scale, opacity = model.gaussians()
    device = xyz.device
    p, h, w = project(xyz, frame, device, image_size)
    uv = rays_uv.to(device).float()
    # Final image export can contain hundreds of thousands of rays. Chunking
    # keeps the [rays, gaussians] temporary tensors bounded on 12 GB GPUs.
    chunks = []
    for start in range(0, len(uv), max(1, int(ray_chunk))):
        chunks.append(_render_chunk(model, frame, uv[start:start + ray_chunk], xyz, color, scale, opacity, p, h, w))
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in chunks[0]}

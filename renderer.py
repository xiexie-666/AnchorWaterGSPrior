from __future__ import annotations

import torch


def project(xyz, frame, device, image_size=320):
    """Project normalized-world points with the COLMAP world-to-camera pose."""
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
    z = cam[:, 2]
    z_safe = z.clamp_min(1e-6)
    u = frame.camera.fx * scale * cam[:, 0] / z_safe + frame.camera.cx * scale
    v = frame.camera.fy * scale * cam[:, 1] / z_safe + frame.camera.cy * scale
    return torch.stack([u, v, z], -1), int(round(h0 * scale)), int(round(w0 * scale)), scale


def _render_chunk(model, frame, uv, xyz, color, scale, opacity, p, h, w, focal_scale):
    # Alpha compositing is essential: normalizing weights over all anchors
    # forces alpha=1 even when a ray has no nearby Gaussian.
    delta = uv[:, None, :] - p[None, :, :2]
    z = p[:, 2].clamp_min(1e-4)
    sigma_world = scale[:, :2].mean(-1).clamp_min(1e-4)
    # Keep projected support compact.  The world-space scale is optimized, but
    # an unconstrained footprint can cover tens of pixels in sparse models and
    # turns texture into large blurry blobs.  The baseline uses a 1.8 px upper
    # bound at the 320 px evaluation scale; coverage comes from more anchors.
    sigma = (sigma_world * focal_scale / z).clamp(0.55, 1.8)
    distance2 = delta.square().sum(-1)
    kernel = torch.exp(-0.5 * (distance2 / sigma[None].square()))
    kernel = torch.where(distance2 <= (3.0 * sigma[None]).square(), kernel, torch.zeros_like(kernel))
    a = (1.0 - torch.exp(-opacity[None] * kernel)).clamp(0.0, 0.999)
    one_minus = (1.0 - a).clamp_min(1e-6)
    before = torch.cat([torch.ones((len(uv), 1), device=uv.device, dtype=uv.dtype), one_minus[:, :-1]], 1)
    trans = torch.cumprod(before, dim=1)
    contrib = trans * a
    alpha = contrib.sum(-1).clamp(0.0, 1.0)
    rgb_object = contrib @ color
    depth = (contrib * z[None]).sum(-1) / alpha.clamp_min(1e-6)
    direction = torch.zeros((len(uv), 3), device=uv.device, dtype=uv.dtype)
    direction[:, 2] = -1.0
    medium_rgb, medium_bs, medium_attn = model.water(direction)
    medium = medium_rgb * (1.0 - torch.exp(-medium_attn * depth[:, None])) + medium_bs * 0.05
    rgb_clear = rgb_object / alpha[:, None].clamp_min(1e-6)
    rgb = rgb_clear * alpha[:, None] + medium * (1.0 - alpha[:, None])
    return {"rgb": rgb.clamp(0, 1), "rgb_object": rgb_object.clamp(0, 1), "rgb_clear": rgb_clear.clamp(0, 1),
            "rgb_medium": medium.clamp(0, 1), "depth": depth, "alpha": alpha,
            "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}


def render(model, frame, rays_uv, image_size=320, ray_chunk=512):
    xyz, color, scale, opacity = model.gaussians()
    device = xyz.device
    p, h, w, image_scale = project(xyz, frame, device, image_size)
    focal_scale = 0.5 * (frame.camera.fx + frame.camera.fy) * image_scale
    rough_sigma = (scale[:, :2].mean(-1).clamp_min(1e-4) * focal_scale / p[:, 2].clamp_min(1e-4)).clamp(0.55, 1.8)
    valid = torch.isfinite(p).all(-1) & (p[:, 2] > 1e-4)
    valid &= (p[:, 0] > -3 * rough_sigma) & (p[:, 0] < w + 3 * rough_sigma)
    valid &= (p[:, 1] > -3 * rough_sigma) & (p[:, 1] < h + 3 * rough_sigma)
    uv = rays_uv.to(device).float()
    if not bool(valid.any()):
        zeros = uv.new_zeros((len(uv), 3)); alpha = uv.new_zeros(len(uv)); depth = uv.new_ones(len(uv))
        direction = torch.zeros((len(uv), 3), device=device); direction[:, 2] = -1.0
        medium_rgb, medium_bs, medium_attn = model.water(direction)
        return {"rgb": medium_rgb.clamp(0, 1), "rgb_object": zeros, "rgb_clear": zeros,
                "rgb_medium": medium_rgb.clamp(0, 1), "depth": depth, "alpha": alpha,
                "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}
    p, color, scale, opacity = p[valid], color[valid], scale[valid], opacity[valid]
    order = torch.argsort(p[:, 2])
    p, color, scale, opacity = p[order], color[order], scale[order], opacity[order]
    chunks = []
    for start in range(0, len(uv), max(1, int(ray_chunk))):
        chunks.append(_render_chunk(model, frame, uv[start:start + ray_chunk], xyz[valid][order], color, scale,
                                    opacity, p, h, w, focal_scale))
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in chunks[0]}

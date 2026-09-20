from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-12 else float(-10 * np.log10(mse))


def ssim_simple(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float((2 * mu_a * mu_b + 1e-4) * (2 * cov + 1e-4) / ((mu_a * mu_a + mu_b * mu_b + 1e-4) * (va + vb + 1e-4)))


def evaluate_render_dir(render_dir: str, gt_dir: str):
    rows = []
    for p in sorted(Path(render_dir).glob("*.png")):
        q = Path(gt_dir) / p.name
        if not q.exists():
            continue
        a = np.asarray(Image.open(p).convert("RGB")) / 255.0
        b = np.asarray(Image.open(q).convert("RGB").resize((a.shape[1], a.shape[0]))) / 255.0
        rows.append((psnr(a, b), ssim_simple(a, b)))
    if not rows:
        return {}
    return {"PSNR": float(np.mean([x[0] for x in rows])), "SSIM": float(np.mean([x[1] for x in rows])), "views": len(rows)}


def save_results(path, metrics):
    Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def render_metrics(predictions, targets):
    """Aggregate image metrics from already rendered RGB arrays.

    ``predictions`` and ``targets`` are HxWx3 float arrays in [0, 1].  Keeping
    this path in memory makes the final evaluation use exactly the same image
    resize and camera selection as the training script.
    """
    rows = []
    for pred, target in zip(predictions, targets):
        pred = np.asarray(pred, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        rows.append((psnr(pred, target), ssim_simple(pred, target)))
    if not rows:
        return {"views": 0}
    return {
        "views": len(rows),
        "PSNR": float(np.mean([r[0] for r in rows])),
        "SSIM": float(np.clip(np.mean([r[1] for r in rows]), 0.0, 1.0)),
        "LPIPS": None,
        "LPIPS_note": "未计算：项目未强制下载预训练 LPIPS 权重。",
    }


def _read_xyz(path):
    """Read a PLY vertex cloud with a small ASCII fallback."""
    try:
        from plyfile import PlyData
        v = PlyData.read(str(path))["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    except Exception:
        lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        start = next(i for i, line in enumerate(lines) if line.strip() == "end_header") + 1
        out = []
        for line in lines[start:]:
            tok = line.split()
            if len(tok) >= 3:
                out.append([float(tok[0]), float(tok[1]), float(tok[2])])
        return np.asarray(out, dtype=np.float32)


def _normalize_cloud(x):
    center = np.median(x, axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(x - center, axis=1), 98)
    return (x - center) / max(float(scale), 1e-6)


def geometry_metrics(pred_xyz, gt_path, threshold=0.05, max_points=50000):
    """Compute symmetric Chamfer and F-score after per-cloud normalization.

    The normalization is intentional for COLMAP reconstructions whose global
    scale is arbitrary.  Therefore the reported values are in normalized scene
    coordinates and are not meters or centimeters.
    """
    gt_xyz = _read_xyz(gt_path)
    pred_xyz = np.asarray(pred_xyz, dtype=np.float32)
    if len(pred_xyz) > max_points:
        pred_xyz = pred_xyz[np.linspace(0, len(pred_xyz) - 1, max_points).astype(np.int64)]
    if len(gt_xyz) > max_points:
        gt_xyz = gt_xyz[np.linspace(0, len(gt_xyz) - 1, max_points).astype(np.int64)]
    pred_xyz, gt_xyz = _normalize_cloud(pred_xyz), _normalize_cloud(gt_xyz)
    try:
        from scipy.spatial import cKDTree
        d_pred = cKDTree(gt_xyz).query(pred_xyz, k=1)[0]
        d_gt = cKDTree(pred_xyz).query(gt_xyz, k=1)[0]
    except Exception as exc:
        return {"Chamfer": None, "F-score": None, "geometry_note": f"scipy 不可用: {exc}"}
    precision = float(np.mean(d_pred <= threshold))
    recall = float(np.mean(d_gt <= threshold))
    fscore = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "Chamfer": float((np.mean(d_pred) + np.mean(d_gt)) * 0.5),
        "F-score": float(fscore),
        "geometry_threshold": threshold,
        "geometry_units": "normalized_scene_coordinates",
        "geometry_gt": str(gt_path),
    }

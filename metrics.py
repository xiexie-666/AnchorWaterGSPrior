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

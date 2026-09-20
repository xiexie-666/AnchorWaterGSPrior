from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from anchor_water_gsprior.colmap import discover_scene
from anchor_water_gsprior.config import TrainConfig
from metrics import evaluate_render_dir, save_results
from anchor_water_gsprior.model import AnchorWaterModel
from anchor_water_gsprior.renderer import render
from anchor_water_gsprior.tsdf import TSDFPrior


def load_image(path, size):
    im = Image.open(path).convert("RGB")
    scale = min(1.0, float(size) / max(im.height, im.width))
    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(im).copy()).float().reshape(-1, 3) / 255.0, im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--phase1", type=int, default=15000)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--max-anchors", type=int, default=8192)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.output); (out / "checkpoints").mkdir(parents=True, exist_ok=True); (out / "renders").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir, sparse, frames, points, colors = discover_scene(args.data)
    if len(points) > args.max_anchors:
        idx = np.linspace(0, len(points) - 1, args.max_anchors).astype(np.int64); points, colors = points[idx], colors[idx]
    xyz = torch.from_numpy(points).to(device); rgb0 = torch.from_numpy(colors).to(device)
    model = AnchorWaterModel(xyz, rgb0).to(device)
    opt = torch.optim.Adam(model.optimizer_parameters(), lr=2e-3)
    prior = TSDFPrior()
    state = {"phase": 1, "anchor_frozen": False, "anchor_count": int(len(points)), "device": str(device)}
    logs = out / "metrics.jsonl"
    t0 = time.time()
    for step in range(1, args.iterations + 1):
        if step == args.phase1 + 1:
            model.freeze_anchor(); opt = torch.optim.Adam(model.optimizer_parameters(include_anchor=False), lr=1e-3)
            prior.update(model.anchor); state.update({"phase": 2, "anchor_frozen": True})
            assert not model._anchor.requires_grad and len(model.anchor) == state["anchor_count"]
        frame = frames[(step - 1) % len(frames)]
        target, (w, h) = load_image(image_dir / frame.name, args.image_size)
        n = min(2048, target.shape[0]); ids = torch.randperm(target.shape[0])[:n]
        # Rasterizer 使用左上角像素坐标，和 PIL flatten 顺序一致。
        yy, xx = torch.div(ids, w, rounding_mode="floor"), ids % w
        rays = torch.stack([xx.float(), yy.float()], -1).to(device); target = target[ids].to(device)
        pred = render(model, frame, rays, args.image_size)
        loss_photo = F.l1_loss(pred["rgb"], target)
        loss_flat = model.flat_loss()
        loss_prior = prior.loss(*[model.gaussians()[0], model.gaussians()[3]]) if state["phase"] == 2 else pred["rgb"].new_zeros(())
        loss = loss_photo + 1e-2 * loss_flat + 1e-3 * loss_prior
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % args.log_every == 0 or step == 1 or step == args.phase1 + 1:
            rec = {"step": step, "loss": float(loss), "photo": float(loss_photo), "flat": float(loss_flat), "tsdf": float(loss_prior), **state, "elapsed_s": time.time() - t0}
            with logs.open("a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec, ensure_ascii=False), flush=True)
        if step % args.save_every == 0 or step == args.iterations:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "state": state, "step": step}, out / "checkpoints" / f"step_{step:06d}.pt")
    # 低频导出少量视图，避免把完整训练变成 IO 瓶颈。
    model.eval()
    with torch.no_grad():
        for i, frame in enumerate(frames[: min(5, len(frames))]):
            target, (w, h) = load_image(image_dir / frame.name, args.image_size)
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            pred = render(model, frame, torch.stack([xx.flatten(), yy.flatten()], -1), args.image_size)
            Image.fromarray((pred["rgb"].reshape(h, w, 3).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).save(out / "renders" / f"{i:04d}.png")
            Image.fromarray((pred["rgb_clear"].reshape(h, w, 3).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).save(out / "renders" / f"{i:04d}_clear.png")
    with (out / "run_summary.json").open("w", encoding="utf-8") as f: json.dump({"data": str(args.data), "frames": len(frames), "anchors": len(model.anchor), **state}, f, indent=2)
    print(json.dumps({"status": "finished", "output": str(out), **state}, ensure_ascii=False))


if __name__ == "__main__":
    main()

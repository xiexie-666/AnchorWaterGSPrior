"""Small, dependency-light COLMAP reader used by the training prototype."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Frame:
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    camera: Camera

    def world_to_camera(self, xyz: np.ndarray) -> np.ndarray:
        q = self.qvec / (np.linalg.norm(self.qvec) + 1e-12)
        w, x, y, z = q
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float32)
        return xyz @ r.T + self.tvec[None]


def _camera_from_tokens(tokens: List[str]) -> Camera:
    model = tokens[1]
    width, height = int(tokens[2]), int(tokens[3])
    p = [float(x) for x in tokens[4:]]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        return Camera(width, height, p[0], p[0], p[1], p[2])
    return Camera(width, height, p[0], p[1], p[2], p[3])


def _read_cameras_txt(path: Path) -> Dict[int, Camera]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        out[int(tok[0])] = _camera_from_tokens(tok)
    return out


def _read_images_txt(path: Path, cameras: Dict[int, Camera]) -> List[Frame]:
    lines = [x for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x and not x.startswith("#")]
    out = []
    for i in range(0, len(lines), 2):
        tok = lines[i].split()
        if len(tok) < 10:
            continue
        out.append(Frame(tok[9], np.asarray([float(x) for x in tok[1:5]], np.float32),
                         np.asarray([float(x) for x in tok[5:8]], np.float32), cameras[int(tok[8])]))
    return out


def read_cameras_and_frames(sparse: Path) -> Tuple[Dict[int, Camera], List[Frame]]:
    sparse = Path(sparse)
    ct, it = sparse / "cameras.txt", sparse / "images.txt"
    if ct.exists() and it.exists():
        cams = _read_cameras_txt(ct)
        return cams, _read_images_txt(it, cams)
    cb, ib = sparse / "cameras.bin", sparse / "images.bin"
    if cb.exists() and ib.exists():
        cams = _read_cameras_bin(cb)
        return cams, _read_images_bin(ib, cams)
    raise FileNotFoundError(f"需要 COLMAP cameras.txt/images.txt 或先导出文本: {sparse}")


_CAMERA_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 8, 9: 4, 10: 12, 11: 12, 12: 12, 13: 12}


def _camera_from_bin(model_id, width, height, p):
    # COLMAP camera model IDs: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL,
    # OPENCV, OPENCV_FISHEYE, FULL_OPENCV, FOV, THIN_PRISM_FISHEYE, ...
    if model_id in (0, 2, 3):
        fx = fy = p[0]; cx, cy = p[1:3]
    else:
        fx, fy, cx, cy = p[:4]
    return Camera(int(width), int(height), float(fx), float(fy), float(cx), float(cy))


def _read_cameras_bin(path: Path):
    out = {}
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model_id = struct.unpack("<ii", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            nparam = _CAMERA_PARAMS.get(model_id, 4)
            p = struct.unpack("<" + "d" * nparam, f.read(8 * nparam))
            out[cid] = _camera_from_bin(model_id, width, height, p)
    return out


def _read_images_bin(path: Path, cameras: Dict[int, Camera]):
    out = []
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)
            q = np.asarray(struct.unpack("<4d", f.read(32)), np.float32)
            t = np.asarray(struct.unpack("<3d", f.read(24)), np.float32)
            cid = struct.unpack("<i", f.read(4))[0]
            name_bytes = bytearray()
            while True:
                b = f.read(1)
                if b == b"\x00" or not b:
                    break
                name_bytes.extend(b)
            name = name_bytes.decode("utf-8", errors="replace")
            npts = struct.unpack("<Q", f.read(8))[0]
            f.seek(npts * (8 + 8 + 8), 1)
            out.append(Frame(name, q, t, cameras[cid]))
    return out


def read_points(sparse: Path, max_points: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    sparse = Path(sparse)
    ply = sparse / "points3D.ply"
    if ply.exists():
        try:
            from plyfile import PlyData
            data = PlyData.read(str(ply))["vertex"]
            xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
            if all(k in data.data.dtype.names for k in ("red", "green", "blue")):
                rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32) / 255.0
            else:
                rgb = np.full_like(xyz, 0.6)
            return _subsample(xyz, rgb, max_points)
        except Exception:
            pass
    txt = sparse / "points3D.txt"
    if txt.exists():
        xyz, rgb = [], []
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            xyz.append([float(x) for x in tok[1:4]])
            rgb.append([float(x) / 255.0 for x in tok[4:7]])
        return _subsample(np.asarray(xyz, np.float32), np.asarray(rgb, np.float32), max_points)
    raise FileNotFoundError(f"未找到 points3D.ply/points3D.txt: {sparse}")


def _subsample(xyz: np.ndarray, rgb: np.ndarray, max_points: int):
    keep = np.isfinite(xyz).all(1)
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        xyz, rgb = xyz[idx], rgb[idx]
    center = np.median(xyz, axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(xyz - center, axis=1), 98)
    scale = max(float(scale), 1e-3)
    return ((xyz - center) / scale).astype(np.float32), rgb.astype(np.float32)


def discover_scene(root: str, prefer_white_balance: bool = True):
    root = Path(root)
    image_dir = root / ("images_wb" if prefer_white_balance and (root / "images_wb").exists() else "images")
    sparse = root / "sparse" / "0"
    if not sparse.exists():
        sparse = root / "sparse"
    _, frames = read_cameras_and_frames(sparse)
    points, colors = read_points(sparse)
    frames = [f for f in frames if (image_dir / f.name).exists()]
    if not frames:
        raise RuntimeError(f"没有与 COLMAP images.txt 对应的图像: {image_dir}")
    return image_dir, sparse, frames, points, colors

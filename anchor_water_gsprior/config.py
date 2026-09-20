from dataclasses import dataclass


@dataclass
class TrainConfig:
    iterations: int = 30000
    phase1_end: int = 15000
    image_size: int = 640
    max_anchors: int = 8192
    offset_count: int = 4
    lr: float = 2e-3
    water_lr: float = 1e-3
    flat_weight: float = 1e-2
    water_weight: float = 1.0
    tsdf_weight: float = 1e-3
    depth_weight: float = 1e-3
    densify_every: int = 1000
    tsdf_every: int = 1000
    log_every: int = 100
    save_every: int = 5000
    seed: int = 7

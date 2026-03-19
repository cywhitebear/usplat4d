import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import tyro
import yaml
from loguru import logger as guru

# Ensure we can import from shape-of-motion
import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), "../shape-of-motion"))

from flow3d.renderer import Renderer

torch.set_float32_matmul_precision("high")

@dataclass
class RenderUSplat4DConfig:
    work_dir: str
    """USplat4D output directory, e.g. /media/.../Outputs/usplat4d/iphone/backpack"""
    som_dir: Optional[str] = None
    """Original SoM output directory. Required if work_dir/cfg.yaml is lost or corrupted."""
    port: int = 8890
    """Port for the rendering server."""


def main(cfg: RenderUSplat4DConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    work_dir = cfg.work_dir
    os.makedirs(work_dir, exist_ok=True)
    
    # 1. Determine paths
    ckpt_path = osp.join(work_dir, "usplat4d", "final.ckpt")
    if not osp.exists(ckpt_path):
        guru.error(f"USplat4D checkpoint not found at {ckpt_path}. Did training finish?")
        return

    usplat_cfg_path = osp.join(work_dir, "cfg.yaml")
    
    # 2. Find som_dir
    som_dir = cfg.som_dir
    if not som_dir and osp.exists(usplat_cfg_path):
        try:
            with open(usplat_cfg_path, "r") as file:
                usplat_cfg = yaml.safe_load(file)
                som_dir = getattr(usplat_cfg, "get", lambda k: None)("som_dir") if isinstance(usplat_cfg, dict) else None
        except Exception as e:
            guru.warning(f"Could not read som_dir from {usplat_cfg_path}: {e}")
    
    if not som_dir:
        guru.error("Missing som_dir. The cfg.yaml is missing/corrupted and --som-dir was not provided.")
        guru.error("Please run again with: conda run -n som python render_usplat4d.py --work-dir ... --som-dir /path/to/som/output")
        return

    if not osp.exists(som_dir):
        guru.error(f"The specified som_dir does not exist: {som_dir}")
        return

    # 3. Read use_2dgs from SoM config
    som_cfg_path = osp.join(som_dir, "cfg.yaml")
    use_2dgs = False
    if osp.exists(som_cfg_path):
        try:
            with open(som_cfg_path, "r") as file:
                som_cfg = yaml.safe_load(file)
                if isinstance(som_cfg, dict):
                    use_2dgs = som_cfg.get("use_2dgs", False)
        except Exception:
            pass
    else:
        guru.warning(f"SoM config not found at {som_cfg_path}, assuming use_2dgs=False")

    # 4. Prepare work_dir structure for Renderer
    output_cfg = {}
    if osp.exists(usplat_cfg_path):
        try:
            with open(usplat_cfg_path, "r") as file:
                output_cfg = yaml.safe_load(file) or {}
                if not isinstance(output_cfg, dict):
                    output_cfg = {}
        except Exception:
            pass
    output_cfg["use_2dgs"] = use_2dgs
    if "som_dir" not in output_cfg:
        output_cfg["som_dir"] = som_dir

    with open(usplat_cfg_path, "w") as file:
        yaml.dump(output_cfg, file, default_flow_style=False)

    guru.info(f"Using checkpoint: {ckpt_path}")
    guru.info(f"Using som_dir: {som_dir}")
    guru.info(f"use_2dgs: {use_2dgs}")

    # 5. Initialize Renderer
    try:
        renderer = Renderer.init_from_checkpoint(
            ckpt_path,
            device,
            use_2dgs=use_2dgs,
            work_dir=work_dir,
            port=cfg.port,
        )
    except Exception as e:
        guru.error(f"Failed to initialize renderer: {e}")
        return

    guru.info(f"Starting rendering from global_step={renderer.global_step}")
    guru.info(f"Viewer server runs at http://localhost:{cfg.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        guru.info("Rendering server stopped.")


if __name__ == "__main__":
    main(tyro.cli(RenderUSplat4DConfig))

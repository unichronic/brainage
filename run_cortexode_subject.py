#!/usr/bin/env python3
"""Run both pretrained CortexODE hemispheres for one prepared subject.

The input contract is the CortexODE ADNI-style directory:
    <subject_dir>/mri/orig.mgz

This is a small batch-safe wrapper around the repository's model code.  It
loads the segmentation and both surface-deformation networks once, rather than
starting the repository's one-hemisphere CLI twice.  The output is a compact
JSON feature record; FreeSurfer meshes are only written when --surface-dir is
provided.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint


SCRIPT_DIR = Path(__file__).resolve().parent
CORTEXODE_DIR = SCRIPT_DIR.parent / "CortexODE"
sys.path.insert(0, str(CORTEXODE_DIR))
# CortexODE's topology-correction initializer uses the repository-relative
# path ``./util/critical186LUT.raw.gz`` at import time.
os.chdir(CORTEXODE_DIR)

from data.preprocess import process_surface_inverse, process_volume  # noqa: E402
from model.net import CortexODE, Unet  # noqa: E402
import eval as cortexode_eval  # noqa: E402


def load_models(model_dir: Path, device: torch.device):
    cortexode_eval.device = device
    segnet = Unet(c_in=1, c_out=3).to(device)
    segnet.load_state_dict(
        torch.load(model_dir / "model_seg_adni_pretrained.pt", map_location=device)
    )
    segnet.eval()

    models = {}
    for hemi in ("lh", "rh"):
        wm = CortexODE(dim_in=3, dim_h=128, kernel_size=5, n_scale=3).to(device)
        gm = CortexODE(dim_in=3, dim_h=128, kernel_size=5, n_scale=3).to(device)
        wm.load_state_dict(
            torch.load(model_dir / f"model_wm_adni_{hemi}_pretrained.pt", map_location=device)
        )
        gm.load_state_dict(
            torch.load(model_dir / f"model_gm_adni_{hemi}_pretrained.pt", map_location=device)
        )
        wm.eval()
        gm.eval()
        models[hemi] = (wm, gm)
    return segnet, models


def summarize_surface(v_white: np.ndarray, v_pial: np.ndarray) -> dict[str, float | int]:
    distances = np.linalg.norm(v_pial - v_white, axis=1)
    finite = np.isfinite(distances)
    if not np.any(finite):
        raise RuntimeError("all CortexODE vertex distances are non-finite")
    values = distances[finite]
    return {
        "nverts": int(len(distances)),
        "nfinite": int(np.sum(finite)),
        "mean_mm": float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "p01_mm": float(np.percentile(values, 1)),
        "p95_mm": float(np.percentile(values, 95)),
        "max_mm": float(np.max(values)),
    }


def run_subject(
    input_dir: Path,
    model_dir: Path,
    device: torch.device,
    surface_dir: Path | None,
    loaded_models=None,
):
    started = time.monotonic()
    brain_path = input_dir / "mri" / "orig.mgz"
    if not brain_path.is_file():
        raise FileNotFoundError(brain_path)

    if loaded_models is None:
        segnet, models = load_models(model_dir, device)
    else:
        segnet, models = loaded_models
    brain = nib.load(str(brain_path))
    brain_arr = (brain.get_fdata() / 255.0).astype(np.float32)
    brain_arr = process_volume(brain_arr, "adni")
    volume_in = torch.from_numpy(brain_arr).unsqueeze(0).to(device)
    t = torch.tensor([0.0, 1.0], device=device)

    with torch.no_grad():
        seg_out = segnet(volume_in)
        seg_pred = torch.argmax(seg_out, dim=1)[0]

    result: dict[str, object] = {"status": "ok", "input_dir": str(input_dir)}
    if surface_dir is not None:
        surface_dir.mkdir(parents=True, exist_ok=True)

    for hemi, label in (("lh", 1), ("rh", 2)):
        seg = (seg_pred == label).cpu().numpy()
        v_in_np, f_in_np = cortexode_eval.seg2surf(
            seg, "adni", sigma=0.5, alpha=16, level=0.8, n_smooth=2
        )
        v_in = torch.from_numpy(v_in_np).unsqueeze(0).to(device)
        f_in = torch.from_numpy(f_in_np).long().unsqueeze(0).to(device)
        wm_model, gm_model = models[hemi]

        with torch.no_grad():
            wm_model.set_data(v_in, volume_in)
            v_wm = odeint(
                wm_model,
                v_in,
                t=t,
                method="euler",
                options={"step_size": 0.1},
            )[-1]
            gm_in = v_wm.clone()
            for _ in range(2):
                gm_in = cortexode_eval.laplacian_smooth(gm_in, f_in, lambd=1.0)
                normals = cortexode_eval.compute_normal(gm_in, f_in)
                gm_in += 0.002 * normals
            gm_model.set_data(gm_in, volume_in)
            v_gm = odeint(
                gm_model,
                gm_in,
                t=t,
                method="euler",
                options={"step_size": 0.05},
            )[-1]

        v_wm_np, f_wm_np = process_surface_inverse(
            v_wm[0].cpu().numpy(), f_in[0].cpu().numpy(), "adni"
        )
        v_gm_np, f_gm_np = process_surface_inverse(
            v_gm[0].cpu().numpy(), f_in[0].cpu().numpy(), "adni"
        )
        result[hemi] = summarize_surface(v_wm_np, v_gm_np)

        if surface_dir is not None:
            nib.freesurfer.io.write_geometry(
                str(surface_dir / f"{hemi}.white"), v_wm_np, f_wm_np
            )
            nib.freesurfer.io.write_geometry(
                str(surface_dir / f"{hemi}.pial"), v_gm_np, f_gm_np
            )

    result["elapsed_sec"] = float(time.monotonic() - started)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=CORTEXODE_DIR / "ckpts" / "pretrained" / "adni"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--surface-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    result = run_subject(args.input_dir, args.model_dir, device, args.surface_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

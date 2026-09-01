#!/usr/bin/env python3
"""Extract CortexMorph regional thickness from one DL+DiReCT segmentation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
CORTEXMORPH_DIR = SCRIPT_DIR.parent / "cortexmorph"
sys.path.insert(0, str(CORTEXMORPH_DIR))

import apply_CortexMorph as cortexmorph  # noqa: E402
from nnunet.network_architecture.generic_UNet import Generic_UNet  # noqa: E402


def run_subject(input_dir: Path, case_id: str, output_csv: Path) -> None:
    started = time.monotonic()
    required = ["softmax_seg.nii.gz", "wmprobT.nii.gz", "gmprobT.nii.gz"]
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing CortexMorph inputs: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cortexmorph.DEVICE = device
    cortexmorph.PATCH_SIZE = (128, 128, 128)
    cortexmorph.TARGET_LABEL_NAMES = ["x", "y", "z"]
    cortexmorph.BASE_FEATURE_DEPTH = 24
    cortexmorph.POOL_MULTIPLIER = 1
    cortexmorph.orientation_fs = nib.orientations.axcodes2ornt(("L", "I", "A"))
    cortexmorph.lut = pd.read_csv(
        CORTEXMORPH_DIR / "freesurfer_cortex_lut.csv", sep="    ", engine="python"
    ).iloc[:, :2]

    cortexmorph.unet = Generic_UNet(
        input_channels=2,
        base_num_features=cortexmorph.BASE_FEATURE_DEPTH,
        num_classes=3,
        num_pool=3,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=cortexmorph.POOL_MULTIPLIER,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        deep_supervision=False,
        final_nonlin=nn.Identity(),
        seg_output_use_bias=True,
    )
    cortexmorph.unet.load_state_dict(
        torch.load(CORTEXMORPH_DIR / "cortexmorph_weights.pth.tar", map_location=device)
    )
    cortexmorph.unet = cortexmorph.unet.to(device)
    cortexmorph.unet.eval()

    parcellation = nib.load(str(input_dir / "softmax_seg.nii.gz"))
    wm_image = nib.load(str(input_dir / "wmprobT.nii.gz"))
    gm_image = nib.load(str(input_dir / "gmprobT.nii.gz"))
    stats = cortexmorph.get_regional_averages(case_id, parcellation, wm_image, gm_image)
    stats.insert(0, "elapsed_sec", float(time.monotonic() - started))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_csv)
    print(stats.to_json(orient="index"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    run_subject(args.input_dir, args.case_id, args.output_csv)


if __name__ == "__main__":
    main()

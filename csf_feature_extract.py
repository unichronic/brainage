import os
import sys
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

ATLAS_PATH = os.environ.get('HARVARD_OXFORD_ATLAS', '')
LEFT_VENTRICLE_LABEL = 3   # index 2 + 1
RIGHT_VENTRICLE_LABEL = 14  # index 13 + 1

_atlas_cache = {}


def get_resampled_atlas(target_img):
    key = target_img.shape
    if key not in _atlas_cache:
        if not ATLAS_PATH:
            raise RuntimeError(
                'set HARVARD_OXFORD_ATLAS to a Harvard-Oxford atlas NIfTI file'
            )
        atlas_img = nib.load(ATLAS_PATH)
        resampled = resample_to_img(atlas_img, target_img, interpolation='nearest', force_resample=True, copy_header=True)
        _atlas_cache[key] = resampled.get_fdata().astype(np.int16)
    return _atlas_cache[key]


def extract(mwc3_path):
    img = nib.load(mwc3_path)
    data = img.get_fdata()
    vox_vol = np.prod(img.header.get_zooms()[:3]) / 1000.0  # mL per voxel
    atlas = get_resampled_atlas(img)

    total_csf = float(data.sum() * vox_vol)
    left_vent = float(data[atlas == LEFT_VENTRICLE_LABEL].sum() * vox_vol)
    right_vent = float(data[atlas == RIGHT_VENTRICLE_LABEL].sum() * vox_vol)
    return {
        'total_csf_mL': total_csf,
        'left_ventricle_mL': left_vent,
        'right_ventricle_mL': right_vent,
        'total_ventricle_mL': left_vent + right_vent,
    }


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p, extract(p))

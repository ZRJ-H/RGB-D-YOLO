"""Read-only ablation of calibration hypotheses using saved fixed feature pairs.

These are diagnostics, not estimated or replacement calibrations. Prints JSON only.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path('E:/gpt/_release/foundationstereo_raw1_pair01_corrected_20260928_v1')
new = json.loads((ROOT/'config/stereo_corrected_1280x720_20260928.json').read_text(encoding='utf-8'))
old = json.loads((ROOT/'config/depth200m02243_recovered_20260928.json').read_text(encoding='utf-8'))
diag = json.loads((RELEASE/'rectification_diagnostic.json').read_text(encoding='utf-8'))
pl = np.array(diag['original_points_left'], dtype=np.float64)
pr = np.array(diag['original_points_right'], dtype=np.float64)


def arrays(cal):
    return tuple(np.array(cal[k], dtype=np.float64) for k in
                 ('cameraMatrixL', 'distCoeffL', 'cameraMatrixR', 'distCoeffR', 'R')) + (
                     np.array(cal.get('T_m', cal.get('T')), dtype=np.float64),)


def transformed(points, K, D, R, P):
    return cv2.undistortPointsIter(points.reshape(-1, 1, 2), K, D, R, P,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)).reshape(-1, 2)


def rectify(parameters, f=596.0, left=pl, right=pr, alpha=-1):
    kl, dl, kr, dr, rot, trans = parameters
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(kl, dl, kr, dr, (1280, 720), rot, trans,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha, newImageSize=(1280, 720))
    if f is not None:
        p1[0, 0] = p1[1, 1] = p2[0, 0] = p2[1, 1] = f
    return transformed(left, kl, dl, r1, p1), transformed(right, kr, dr, r2, p2), float(p1[0, 0])


def stats(dy):
    return {k: round(float(v), 6) for k, v in {
        'abs_p50': np.median(abs(dy)), 'abs_p90': np.percentile(abs(dy), 90),
        'signed_p50': np.median(dy), 'abs_max': abs(dy).max(),
        'signed_mean': dy.mean(), 'rmse': np.sqrt(np.mean(dy**2)),
        'fraction_abs_le_1': (abs(dy) <= 1).mean(),
        'fraction_abs_le_2': (abs(dy) <= 2).mean()}.items()}


def summarize(left, right, f=None):
    dy = left[:, 1] - right[:, 1]
    result = {'count': len(dy), 'focal': f, **stats(dy), 'regions': {}}
    for name, keep in [
        ('left_third', pl[:, 0] < 1280/3),
        ('center_third', (pl[:, 0] >= 1280/3) & (pl[:, 0] < 2560/3)),
        ('right_third', pl[:, 0] >= 2560/3),
        ('top_half', pl[:, 1] < 360), ('bottom_half', pl[:, 1] >= 360)]:
        result['regions'][name] = {'count': int(keep.sum()), **stats(dy[keep])}
    # Spatial slope is descriptive only; this does not adjust or refit any points.
    design = np.column_stack((np.ones(len(pl)), (pl[:, 0]-640)/640, (pl[:, 1]-360)/360))
    fit = np.linalg.lstsq(design, dy, rcond=None)[0]
    result['dy_linear_fit_intercept_per_half_width_per_half_height'] = fit.tolist()
    result['dy_linear_fit_residual_abs_p90'] = float(np.percentile(abs(dy-design@fit), 90))
    return result


kl, dl, kr, dr, r, t = arrays(new)
zero = np.zeros(5)
ident = np.eye(3)
tx = np.array([t[0], 0, 0])
params = {
    'new_as_supplied': (kl, dl, kr, dr, r, t),
    'new_D_zero': (kl, zero, kr, zero, r, t),
    'new_R_identity': (kl, dl, kr, dr, ident, t),
    'new_R_identity_D_zero': (kl, zero, kr, zero, ident, t),
    'new_T_horizontal': (kl, dl, kr, dr, r, tx),
    'new_R_transposed_T_unchanged': (kl, dl, kr, dr, r.T, t),
    'new_extrinsics_inverse': (kl, dl, kr, dr, r.T, -r.T@t),
    'new_extrinsics_inverse_D_zero': (kl, zero, kr, zero, r.T, -r.T@t),
    'new_KD_swapped_same_extrinsics': (kr, dr, kl, dl, r, t),
    'new_KD_swapped_inverse_extrinsics': (kr, dr, kl, dl, r.T, -r.T@t),
    'new_KD_swapped_inverse_extrinsics_D_zero': (kr, zero, kl, zero, r.T, -r.T@t),
    'new_intrinsics_only': (kl, dl, kr, dr, ident, tx),
    'new_intrinsics_only_D_zero': (kl, zero, kr, zero, ident, tx),
    'old_at_common_f596': arrays(old),
}
results = {'original': summarize(pl, pr)}
for name, ps in params.items():
    results[name] = summarize(*rectify(ps))
results['old_historical_focal'] = summarize(*rectify(arrays(old), f=None, alpha=1))
results['physical_camera_swap_control'] = summarize(*rectify(
    (kr, dr, kl, dl, r.T, -r.T@t), left=pr, right=pl))
assert abs(results['new_as_supplied']['abs_p50'] - diag['rectified_vertical_abs_pixels']['p50']) < 1e-5
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--detailed', action='store_true', help='Print complete regional distributions.')
args = parser.parse_args()
if not args.detailed:
    results = {name: {
        **{k: value[k] for k in ('count', 'focal', 'abs_p50', 'abs_p90', 'signed_p50')},
        'regions': {region: {k: row[k] for k in ('count', 'abs_p50', 'abs_p90', 'signed_p50')}
                    for region, row in value['regions'].items()},
        'dy_linear_fit_intercept_per_half_width_per_half_height':
            value['dy_linear_fit_intercept_per_half_width_per_half_height'],
    } for name, value in results.items()}
print(json.dumps({'notice': 'Fixed same 374 matches; exploratory hypotheses only, not replacement calibration.',
    'source': str(RELEASE/'rectification_diagnostic.json'), 'results': results}, indent=2))

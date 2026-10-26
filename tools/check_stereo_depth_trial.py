"""Single-pair, scale-fitted consistency diagnostic; not metric acceptance."""
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    trial = root / 'outputs/depth/orbbec_standard_trial_20260923'
    out = root / 'outputs/depth/orbbec_depth_check_pair01_20260923'
    if out.exists():
        raise ValueError('Refusing to overwrite existing result')
    report = json.loads((trial / 'report.json').read_text(encoding='utf-8'))
    row = next(r for r in report['results'] if r['number'] == 1)
    sources = list(Path(r'E:\dpth_data\sn_CPC93630004R\Depth').glob(f"*_d{row['device_timestamp']}_*"))
    assert len(sources) == 1
    raw = cv2.imread(str(sources[0]), cv2.IMREAD_UNCHANGED)
    disp = np.load(trial / 'left_01_doe6/disparity_pixels.npy', allow_pickle=False)
    # 临时假设深度与左IR同视场；按整数采样匹配半分辨率，不进行空间配准。
    assert raw.shape == (disp.shape[0]*2, disp.shape[1]*2)
    reference = raw[::2, ::2].astype(np.float32)
    valid = (reference > 0) & np.isfinite(disp) & (disp > 0)
    yy, xx = np.indices(disp.shape)
    overlap = xx - disp >= 0
    # 棋盘格空间块划分：一半只估比例，另一半计算一致性，避免同点拟合并验收。
    interior = (xx >= 32) & (xx < disp.shape[1]-32) & (yy >= 16) & (yy < disp.shape[0]-16)
    blocks = ((xx // 32 + yy // 32) % 2) == 0
    fit = valid & overlap & interior & blocks
    check = valid & overlap & interior & ~blocks
    factor = float(np.median(reference[fit] * disp[fit]))
    estimate = np.full(disp.shape, np.nan, dtype=np.float32)
    positive = np.isfinite(disp) & (disp > 0)
    estimate[positive] = factor / disp[positive]
    relative = np.full(disp.shape, np.nan, dtype=np.float32)
    relative[valid] = np.abs(estimate[valid]-reference[valid])/reference[valid]
    errors = relative[check]
    stats = {
        'status': 'diagnostic_only_metric_acceptance_pending',
        'pair': row, 'camera_depth_file': str(sources[0]),
        'camera_depth_unit': 'unknown_raw_unit',
        'alignment': 'unverified_same_pixel_assumption_no_registration',
        'reference_is_ground_truth': False,
        'conversion': 'estimate_raw = fitted_factor / disparity_pixels',
        'fitted_factor_raw_times_pixel': factor,
        'scale_source': 'median_camera_depth_times_disparity_on_alternate_32px_blocks',
        'comparison_sampling': 'camera_depth[::2, ::2]; no interpolation',
        'fit_pixels': int(fit.sum()), 'heldout_pixels': int(check.sum()),
        'camera_nonzero_fraction_full_resolution': float((raw>0).mean()),
        'mean_absolute_relative_difference': float(errors.mean()),
        'median_absolute_relative_difference': float(np.median(errors)),
        'p90_absolute_relative_difference': float(np.percentile(errors,90)),
        'fraction_within_5_percent': float((errors<=.05).mean()),
        'fraction_within_10_percent': float((errors<=.1).mean()),
        'limitations': ['No verified IR rectification', 'No verified left IR to depth registration',
                        'No verified depth unit or stereo baseline', 'Scale fitted from camera depth, not independent calibration',
                        'Single scene; spatial blocks are not independent scenes'],
    }
    out.mkdir(parents=True)
    np.save(out/'depth_scaled_to_camera_raw_units.npy', estimate)
    np.save(out/'absolute_relative_difference.npy', relative)
    np.save(out/'heldout_mask.npy', check)
    (out/'report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    lo, hi = np.percentile(reference[valid],[2,98])
    def colour(a, mask):
        gray = np.nan_to_num(np.clip((hi-a)/(hi-lo)*255,0,255)).astype(np.uint8)
        bgr = cv2.applyColorMap(gray,cv2.COLORMAP_TURBO)
        bgr[~mask] = 0
        return bgr
    left = cv2.imread(row['file'])
    left = cv2.resize(left,(disp.shape[1],disp.shape[0]),interpolation=cv2.INTER_AREA)
    panels = [left, colour(reference,reference>0), colour(estimate,positive)]
    err = cv2.applyColorMap(np.nan_to_num(np.clip(relative/.2*255,0,255)).astype(np.uint8),cv2.COLORMAP_TURBO)
    err[~check] = 0
    panels.append(err)
    titles = ['Left IR | pair 01 DOE=6', 'Camera depth | RAW units (not ground truth)',
              'Model depth | scale fitted to camera, NOT metres', 'Held-out difference | 0-20% | black = not evaluated']
    tiles = []
    for panel,title in zip(panels,titles):
        tile = cv2.copyMakeBorder(panel,36,0,0,0,cv2.BORDER_CONSTANT,value=(28,28,28))
        cv2.putText(tile,title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
    cv2.imwrite(str(out/'comparison.jpg'),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

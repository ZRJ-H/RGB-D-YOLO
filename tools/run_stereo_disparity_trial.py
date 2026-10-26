"""Run local standard FoundationStereo on IR pairs; no metric-depth claims."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import types

os.environ['XFORMERS_DISABLED'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
sys.dont_write_bytecode = True

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scale', type=float, default=0.5)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--all-pairs', action='store_true', help='Run all pairs, including the explicitly flagged nearest-timestamp pair')
    selection.add_argument('--numbers', nargs='+', type=int, help='Explicit source frame numbers to rerun')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    if not 0 < args.scale <= 1:
        raise ValueError('scale must be in (0, 1]')
    if args.numbers and (len(set(args.numbers)) != len(args.numbers) or any(n < 1 or n > 47 for n in args.numbers)):
        parser.error('numbers must be unique integers from 1 to 47')
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    from omegaconf import OmegaConf

    # Windows兼容层：用SDPA实现原FlashAttention的缩放与局部窗口语义。
    def attention(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **kwargs):
        if kwargs or causal:
            raise ValueError('Unsupported attention options')
        mask = None
        if window_size != (-1, -1):
            nq, nk = q.shape[1], k.shape[1]
            qi = torch.arange(nq, device=q.device)[:, None] + nk - nq
            ki = torch.arange(nk, device=q.device)[None, :]
            mask = torch.ones((nq, nk), dtype=torch.bool, device=q.device)
            if window_size[0] >= 0:
                mask &= ki >= qi - window_size[0]
            if window_size[1] >= 0:
                mask &= ki <= qi + window_size[1]
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, dropout_p=dropout_p, scale=softmax_scale).transpose(1, 2)

    shim = types.ModuleType('flash_attn')
    shim.flash_attn_func = attention
    shim.flash_attn_qkvpacked_func = lambda qkv, **kw: attention(*qkv.unbind(2), **kw)
    sys.modules['flash_attn'] = shim
    # 完整模型随后严格加载全部参数，无需另行下载骨干的初始预训练权重。
    original_create = timm.create_model
    def local_create(*a, **kw):
        kw['pretrained'] = False
        return original_create(*a, **kw)
    timm.create_model = local_create
    def local_hub(repo, model, *a, **kw):
        if repo != 'facebookresearch/dinov2':
            raise ValueError(f'Unexpected hub source: {repo}')
        from dinov2.hub import backbones
        kw['pretrained'] = False
        return getattr(backbones, model)(*a, **kw)
    torch.hub.load = local_hub

    repo = Path(r'D:\RC\lege\FoundationStereo')
    sys.path.insert(0, str(repo))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
    ckpt = repo / 'pretrained_models/23-51-11/model_best_bp2.pth'
    cfg = OmegaConf.load(ckpt.with_name('cfg.yaml'))
    cfg.vit_size = 'vitl'
    cfg.valid_iters = 32
    print('Loading standard checkpoint on', torch.cuda.get_device_name(0), flush=True)
    model = FoundationStereo(cfg)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    print('Strict checkpoint load passed', flush=True)
    del state
    model = model.cuda().eval()
    rows = json.loads((Path(__file__).resolve().parents[1] / 'outputs/depth/orbbec_left_review_20260923/doe_groups_v2.json').read_text(encoding='utf-8'))
    right_dir = Path(r'E:\dpth_data\sn_CPC93630004R\IR_RIGHT')
    rights = {re.search(r'_d(\d+)_', p.name).group(1): p for p in right_dir.glob('*.png')}
    chosen = [next(r for r in rows if r['doe'] == doe and r['size'] == [1280, 800] and r['device_timestamp'] in rights) for doe in (6, 0)]
    if args.all_pairs:
        chosen = rows
    elif args.numbers:
        by_number = {row['number']: row for row in rows}
        chosen = [by_number[number] for number in args.numbers]
    args.output.mkdir(parents=True)
    report = {'metric_depth': False, 'rectification': 'unverified_no_rectification_applied', 'checkpoint': str(ckpt), 'attention_backend': 'pytorch_sdpa_compatibility', 'iterations': 32, 'scale': args.scale, 'results': []}
    for row in chosen:
        stamp = row['device_timestamp']
        right_stamp = stamp if stamp in rights else min(rights, key=lambda k: abs(int(k)-int(stamp)))
        delta = int(right_stamp)-int(stamp)
        if abs(delta) > 1:
            raise ValueError('No close right timestamp; do not guess pairing')
        lp, rp = Path(row['file']), rights[right_stamp]
        left = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        assert left is not None and right is not None and left.shape == right.shape
        size = (round(left.shape[1]*args.scale), round(left.shape[0]*args.scale))
        left, right = [cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in (left, right)]
        inputs = [torch.from_numpy(np.repeat(im[..., None], 3, axis=2)).permute(2, 0, 1)[None].float().cuda() for im in (left, right)]
        padder = InputPadder(inputs[0].shape, divis_by=32, force_square=False)
        inputs = padder.pad(*inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            prediction = model(*inputs, iters=32, test_mode=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        disp = padder.unpad(prediction.float()).squeeze().cpu().numpy()
        valid = np.isfinite(disp) & (disp > 0)
        if not valid.any():
            raise RuntimeError('No finite positive disparity')
        dest = args.output / f"left_{row['number']:02d}_doe{row['doe']}"
        dest.mkdir()
        np.save(dest/'disparity_pixels.npy', disp)
        # 逆视差只描述相对远近，单位为1/像素，不代表米制距离。
        inverse = np.full(disp.shape, np.nan, dtype=np.float32)
        inverse[valid] = 1.0 / disp[valid]
        np.save(dest/'inverse_disparity_nonmetric.npy', inverse)
        xx = np.arange(disp.shape[1])[None, :]
        overlap = valid & (xx-disp >= 0) & (xx-disp <= disp.shape[1]-1)
        cv2.imwrite(str(dest/'positive_finite_mask.png'), valid.astype(np.uint8)*255)
        cv2.imwrite(str(dest/'horizontal_overlap_mask.png'), overlap.astype(np.uint8)*255)
        cv2.imwrite(str(dest/'left_input.png'), left)
        cv2.imwrite(str(dest/'right_input.png'), right)
        lo, hi = np.percentile(disp[valid], [2,98])
        gray = np.nan_to_num(np.clip((disp-lo)/max(hi-lo,1e-6)*255,0,255)).astype(np.uint8)
        colour = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        colour[~valid] = 0
        cv2.imwrite(str(dest/'disparity_colour.png'), colour)
        panel = np.vstack([cv2.cvtColor(left,cv2.COLOR_GRAY2BGR), cv2.cvtColor(right,cv2.COLOR_GRAY2BGR), colour])
        cv2.putText(panel, f"DOE={row['doe']} | DISPARITY ONLY | calibration unverified", (8,22), cv2.FONT_HERSHEY_SIMPLEX, .48, (0,255,255), 1)
        cv2.imwrite(str(dest/'preview.jpg'),panel)
        item = dict(row, right_file=str(rp), output=str(dest), input_size=list(size), inference_seconds=elapsed, positive_finite_ratio=float(valid.mean()), disparity_percentiles=np.percentile(disp[valid],[2,50,98]).tolist())
        item.update(right_device_timestamp=right_stamp, timestamp_delta_raw=delta,
                    pairing='exact_device_timestamp' if delta == 0 else 'nearest_timestamp_delta_1_unverified',
                    colour_percentile_range=[float(lo),float(hi)], masks_are_confidence=False)
        (dest/'result.json').write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding='utf-8')
        report['results'].append(item)
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(item,ensure_ascii=False),flush=True)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()

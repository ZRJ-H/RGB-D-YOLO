"""Return conditional depth on the original image grid; preserve source disparity."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


NEAR, FAR = 0.3, 5.0
STATUS = "conditional_on_rectified_input"
PROJECT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    require(image is not None, f"Cannot decode image: {path}")
    return image


def save_image(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    require(ok, f"Cannot encode image: {path}")
    encoded.tofile(path)
    check = read_image(path)
    require(check.shape == image.shape, f"Image write verification failed: {path}")


def local_href(path, base):
    try:
        return quote(Path(os.path.relpath(Path(path).resolve(), base)).as_posix(), safe="/:")
    except ValueError:  # Different Windows drives.
        return Path(path).resolve().as_uri()


def validate_calibration(calibration):
    device = calibration["device"]
    require(device["serial_number"] == "CPC93630004R", "Calibration serial does not match this trial")
    require(device.get("is_mirrored") is False, "Mirrored/unknown image geometry is unsupported")
    intrinsics = []
    for name in ("ir_left", "ir_right"):
        stream = calibration["streams"][name]
        intr = stream["intrinsic"]
        dist = stream["distortion"]
        require(stream["available"] and intr["available"] and dist["available"], f"Missing {name} calibration")
        require([stream["width"], stream["height"]] == [1280, 800], f"Unexpected {name} stream size")
        require([intr["width"], intr["height"]] == [1280, 800], f"Unexpected {name} intrinsic size")
        matrix = np.array(intr["camera_matrix"], dtype=float)
        expected = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
        require(matrix.shape == (3, 3) and np.isfinite(matrix).all() and np.allclose(matrix, expected, atol=1e-8, rtol=0), f"Inconsistent {name} K")
        require(intr["fx"] > 0 and intr["fy"] > 0, "Focal lengths must be positive")
        require(dist["model"] in ("BROWN_CONRADY", "BROWN_CONRADY_K6"), "Unknown distortion model")
        require(all(np.isfinite(dist[k]) and abs(dist[k]) <= 1e-12 for k in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")), "Nonzero/unknown IR distortion requires actual rectification")
        intrinsics.append(intr)
    require(np.allclose(intrinsics[0]["camera_matrix"], intrinsics[1]["camera_matrix"], atol=1e-8, rtol=0), "This script requires identical IR K")
    extr = calibration["extrinsics"]["ir_left_to_ir_right"]
    require(extr["available"] and extr["source"] == "IR_LEFT" and extr["target"] == "IR_RIGHT", "Wrong stereo extrinsic direction")
    require(extr["translation_unit"] == "mm", "Stereo translation must explicitly use mm")
    rotation, translation = np.asarray(extr["rotation"]), np.asarray(extr["translation"])
    require(rotation.shape == (3, 3) and np.allclose(rotation, np.eye(3), atol=1e-10, rtol=0), "Nonidentity stereo rotation requires actual rectification")
    require(translation.shape == (3,) and np.isfinite(translation).all() and translation[0] < 0 and np.allclose(translation[1:], 0, atol=1e-10, rtol=0), "Stereo baseline must be pure negative horizontal")
    require(np.isclose(extr["translation_norm_mm"], -translation[0], atol=1e-6, rtol=0), "Inconsistent baseline magnitude")
    return intrinsics, float(-translation[0] / 1000)


def font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def depth_colour(depth, valid):
    # Fixed shared range: near is warm/red, far is cool/blue. Values outside
    # the display range are saturated, never clipped in the float32 array.
    mapped = np.zeros(depth.shape, dtype=np.uint8)
    mapped[valid] = np.rint((1 - np.clip((depth[valid] - NEAR) / (FAR - NEAR), 0, 1)) * 255).astype(np.uint8)
    result = cv2.applyColorMap(mapped, cv2.COLORMAP_TURBO)
    result[~valid] = (30, 25, 20)
    return result


def restore_native_disparity(disparity, source_size):
    """Restore the full field of view and pixel units, without inventing edge depths.

    Nearest-neighbour sampling retains the lower-resolution estimates and holes.
    Disparity scales with image width; metric depth does not scale with resizing.
    The original inference overlap mask remains a conservative validity limit.
    """
    disparity = np.asarray(disparity)
    require(disparity.ndim == 2 and disparity.size > 0, "Expected a 2D disparity array")
    width, height = source_size
    require(width > 0 and height > 0 and int(width) == width and int(height) == height,
            "Native dimensions must be positive integers")
    height_in, width_in = disparity.shape
    require(width >= width_in and height >= height_in, "Native restoration cannot crop/downsample")
    xx_in = np.arange(width_in)[None, :]
    valid_in = (np.isfinite(disparity) & (disparity > 0)
                & (xx_in - disparity >= 0) & (xx_in - disparity <= width_in - 1))
    size = (int(width), int(height))
    native = cv2.resize(disparity.astype(np.float32), size, interpolation=cv2.INTER_NEAREST_EXACT)
    native *= width / width_in
    valid = cv2.resize(valid_in.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
    xx = np.arange(width)[None, :]
    valid &= np.isfinite(native) & (native > 0) & (xx - native >= 0) & (xx - native <= width - 1)
    return native, valid


def legend(width=640):
    canvas = Image.new("RGB", (width, 80), "#f5f6f4")
    draw = ImageDraw.Draw(canvas)
    pad, bar_width = 20, width - 160
    depth = np.linspace(NEAR, FAR, bar_width, dtype=np.float32)[None, :]
    bar = depth_colour(depth, np.ones_like(depth, dtype=bool))
    bar = cv2.resize(bar, (bar_width, 15), interpolation=cv2.INTER_NEAREST)
    canvas.paste(Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)), (pad, 12))
    for value in (NEAR, 1, 2, 3, 4, FAR):
        x = pad + (value - NEAR) / (FAR - NEAR) * (bar_width - 1)
        draw.line((x, 28, x, 32), fill="#41535a", width=1)
        draw.text((x, 34), f"{value:g} m", font=font(14), anchor="mt", fill="#263d43")
    draw.rectangle((width - 108, 12, width - 88, 27), fill="#14191e")
    draw.text((width - 80, 10), "invalid", font=font(14), fill="#263d43")
    draw.text((pad, 58), "Shared 0.3-5 m colour scale; out-of-range colours saturated", font=font(13), fill="#596b70")
    return canvas


def make_preview(left, colour, number, doe, skipped, uncertain, path):
    panel_width = 640
    height = round(left.shape[0] * panel_width / left.shape[1])
    left_rgb = cv2.cvtColor(cv2.resize(left, (panel_width, height)), cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(cv2.resize(colour, (panel_width, height)), cv2.COLOR_BGR2RGB)
    canvas = Image.new("RGB", (1312, height + 170), "#f5f6f4")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{number:02d}  /  DOE {doe}", font=font(24), fill="#14363a")
    label = "NO METRIC DEPTH: calibration size mismatch" if skipped else "PROVISIONAL: requires rectified historical input"
    draw.text((294, 15), label, font=font(20), fill="#975221")
    draw.text((12, 47), "LEFT IR / ORIGINAL GRID (preview reduced)", font=font(16), fill="#41535a")
    draw.text((660, 47), "RAW DISPARITY / per-frame colours" if skipped else "CONDITIONAL DEPTH / metres", font=font(16), fill="#41535a")
    canvas.paste(Image.fromarray(left_rgb), (12, 72))
    canvas.paste(Image.fromarray(right_rgb), (660, 72))
    if skipped:
        draw.text((660, height + 91), "848 x 480 source has no matching calibration.", font=font(17), fill="#975221")
    else:
        canvas.paste(legend(), (660, height + 82))
    footer = "PAIRING UNCERTAIN: right timestamp differs by 1 (unit unverified)" if uncertain else "Geometric validity mask only; this is not model confidence."
    draw.text((12, height + 96), footer, font=font(15), fill="#975221" if uncertain else "#596b70")
    save_image(path, cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))


def make_gallery(output, report):
    cards = []
    for row in report["results"]:
        folder = row["folder"]
        skipped = row["status"] != STATUS
        median = "暂不换算" if skipped else f'{row["depth_percentiles_m"][1]:.2f} m'
        flags = "尺寸不匹配 · 仅视差" if skipped else "历史校正状态待确认"
        if row["pairing_uncertain"]:
            flags += " · 配对时间差 1"
        downloads = f'<a href="{folder}/disparity_pixels_native.npy" download>原尺寸视差 .npy ↓</a>'
        if not skipped:
            downloads = f'<a href="{folder}/depth_m_provisional.npy" download>暂定深度 .npy ↓</a>' + downloads
        cards.append(f'''<article class="card" data-doe="{row['doe']}" data-flag="{int(skipped or row['pairing_uncertain'])}">
<div class="cardtop"><span class="number">{row['number']:02d}</span><span class="tag">DOE {row['doe']}</span><span class="median">{median}<small>{'米制跳过' if skipped else '有效区中位数'}</small></span></div>
<a class="preview" href="{folder}/preview.jpg" target="_blank"><img loading="lazy" src="{folder}/preview.jpg" alt="第 {row['number']} 组左红外和{'视差' if skipped else '暂定深度'}对照"></a>
<div class="cardbody"><p class="flags">{flags}</p><p class="meta">输出 {row['output_size'][0]} × {row['output_size'][1]}（与原图一致） · 内部推理 {row['input_size'][0]} × {row['input_size'][1]}{' · 有效 ' + format(row['valid_ratio'] * 100, '.1f') + '%' if not skipped else ''}</p>
<div class="downloads">{downloads}<a href="{folder}/result.json">记录 JSON ↗</a></div>
<a class="source" href="{html.escape(row['source_preview_href'], quote=True)}" target="_blank">左右图与原始视差预览 ↗</a></div></article>''')
    template = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>47 组双目深度试跑 · CPC93630004R</title>
<style>
:root{--ink:#163b3e;--muted:#627475;--paper:#f3f5f1;--line:#dce3dc;--teal:#0f6e65;--amber:#925625}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",sans-serif}a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}header,main,footer{max-width:1480px;margin:auto;padding:32px 42px}header{padding-top:50px;padding-bottom:22px}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--teal);font-weight:700}h1{font-size:40px;letter-spacing:-1.5px;margin:14px 0 12px;line-height:1.2}.intro{max-width:950px;line-height:1.8;color:var(--muted);margin:0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:16px;overflow:hidden;margin:28px 0 20px}.stat{background:#fff;padding:22px}.stat b{display:block;font-size:30px;font-weight:600}.stat span{font-size:13px;color:var(--muted)}.notice{background:#fff4df;border:1px solid #ecd9b5;border-left:5px solid #b47632;border-radius:8px;padding:17px 20px;font-size:14px;line-height:1.9;color:#70502f}.notice b{color:#613e20}.details{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:20px 0 0;flex-wrap:wrap}.legend{width:530px;max-width:100%}.links{display:flex;gap:18px;font-size:14px}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:0 0 22px}.filters{display:flex;gap:8px;flex-wrap:wrap}button{border:1px solid var(--line);border-radius:99px;background:white;color:var(--ink);padding:10px 17px;font:inherit;font-size:13px;cursor:pointer}button.active{background:var(--ink);color:#fff;border-color:var(--ink)}#count{font-size:13px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:23px}.card{background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 5px 15px #153b3e04}.cardtop{display:flex;align-items:center;gap:13px;padding:17px 18px}.number{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}.tag{background:#e9f3ee;color:var(--teal);padding:5px 8px;border-radius:5px;font-size:11px;font-weight:650}.median{margin-left:auto;text-align:right;font-size:21px;font-weight:600}.median small{display:block;color:var(--muted);font-size:10px;font-weight:400;margin-top:3px}.preview{display:block;background:#f5f6f4}.preview img{display:block;width:100%;height:auto}.cardbody{padding:0 18px 18px}.flags{font-size:12px;color:var(--amber);margin:12px 0 8px;line-height:1.5}.meta{font-size:11px;color:var(--muted);margin:0 0 15px}.downloads{display:flex;gap:11px;flex-wrap:wrap;font-size:11px;border-top:1px solid #edf0ec;padding-top:12px}.source{display:inline-block;font-size:10px;margin-top:12px;color:var(--muted)}footer{border-top:1px solid var(--line);font-size:12px;color:var(--muted);line-height:1.9;margin-top:20px}main{padding-top:12px}.hidden{display:none}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){header,main,footer{padding-left:18px;padding-right:18px}header{padding-top:28px}h1{font-size:30px}.stats{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.toolbar{align-items:flex-start}.stat{padding:16px}.stat b{font-size:26px}.links{flex-wrap:wrap}}
</style></head><body><header><div class="eyebrow">FOUNDATIONSTEREO / LOCAL REVIEW / 2026.09.24</div><h1>47 组双目深度试跑</h1><p class="intro">相机 CPC93630004R · 标准 FoundationStereo · 半分辨率 · 32 次迭代。左侧为红外输入，右侧为按新标定参数换算的暂定深度；点击图片放大，切换 DOE 分组浏览。</p>
<div class="stats"><div class="stat"><b>47</b><span>完成视差推理</span></div><div class="stat"><b>46</b><span>生成暂定米制数组</span></div><div class="stat"><b>95.23 mm</b><span>左 / 右 IR 基线</span></div><div class="stat"><b>0.3 — 5 m</b><span>统一颜色范围</span></div></div>
<div class="notice"><b>这些米制结果仍有前提：历史 IR 图像已经完成双目校正。</b><br>用户此前说明原图未校正；本次 JSON 呈现零畸变、相同内参和水平基线的理想校正几何，不能证明历史采集状态，因此未重映射原图，也未完成米制精度验收。<b>24 号</b>原图为 848 × 480，与标定尺寸不符，仅展示视差；<b>7 号</b>左右时间戳差 1，配对仍待确认。有效掩膜只排除非正、非有限和水平越界视差，不代表模型置信度。</div>
<div class="details"><img class="legend" src="legend.png" alt="深度统一色条：近处红色、远处蓝色，0.3 至 5 米；无效区黑色"><div class="links"><a href="report.json">完整报告 ↗</a><a href="README.md">使用说明 ↗</a><a href="calibration.json">标定副本 ↗</a></div></div></header>
<main><div class="toolbar"><div class="filters"><button class="active" data-filter="all">全部 47 组</button><button data-filter="6">DOE 6 · 23</button><button data-filter="0">DOE 0 · 24</button><button data-filter="flag">特殊帧 · 2</button></div><span id="count">显示 47 组</span></div><div class="grid">__CARDS__</div></main>
<footer>计算式：Z = fx × B / [d − (cxL − cxR)]；640 × 400 输入的 fx = __FX__ px，B = __BASELINE__ m。数组保留真实计算值，图像颜色在 0.3–5 m 之外饱和；NaN 表示无效像素。<br>中位数只描述该画面的暂定距离分布，不能据此比较 DOE 精度。未读取或换算旧设备深度 PNG，未生成训练标签、点云或下游几何。文件均保存在本地。</footer>
<script>const cards=[...document.querySelectorAll('.card')];document.querySelectorAll('button[data-filter]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));let n=0;cards.forEach(c=>{const f=b.dataset.filter;const show=f==='all'||(f==='flag'?c.dataset.flag==='1':c.dataset.doe===f);c.classList.toggle('hidden',!show);n+=show});document.getElementById('count').textContent='显示 '+n+' 组'}));</script></body></html>'''
    template = template.replace("半分辨率 · 32 次迭代", "内部半分辨率推理 · 原尺寸输出 · 32 次迭代")
    template = template.replace("左侧为红外输入，右侧为按新标定参数换算的暂定深度；",
                                "深度数组、视差和掩膜与原图宽高一致。最近邻恢复尺寸不增加细节；预览为缩小排版。左侧为红外原图，右侧为暂定深度；")
    template = template.replace("640 × 400 输入的 fx", "1280 × 800 输出的 fx")
    template = template.replace("文件均保存在本地。", "输出保留原始像素坐标和完整视场。深度到RGB配准尚未执行：历史RGB为1280×720、标定为1280×800，且IR_LEFT与DEPTH坐标系关系待确认。")
    total = len(report["results"])
    flagged = sum(row["status"] != STATUS or row["pairing_uncertain"] for row in report["results"])
    template = template.replace("47 组", f"{total} 组").replace("<b>47</b>", f"<b>{total}</b>")
    template = template.replace("<b>46</b>", f'<b>{report["metric_pairs"]}</b>')
    template = template.replace("DOE 6 · 23", f'DOE 6 · {sum(row["doe"] == 6 for row in report["results"])}')
    template = template.replace("DOE 0 · 24", f'DOE 0 · {sum(row["doe"] == 0 for row in report["results"])}')
    template = template.replace("特殊帧 · 2", f"特殊帧 · {flagged}")
    if 24 not in report["skipped_metric_pairs"]:
        template = template.replace("<b>24 号</b>原图为 848 × 480，与标定尺寸不符，仅展示视差；", "")
    if not any(row["pairing_uncertain"] for row in report["results"]):
        template = template.replace("<b>7 号</b>左右时间戳差 1，配对仍待确认。", "")
    rendered = template.replace("__CARDS__", "\n".join(cards)).replace("__FX__", f'{report["fx_native_resolution"]:.6f}').replace("__BASELINE__", f'{report["baseline_m"]:.10f}')
    (output / "index.html").write_text(rendered, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--trial", type=Path, required=True, help="Completed selected-pair or full-batch disparity trial directory")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args()
    trial, output, calibration_path = args.trial.resolve(), args.output.resolve(), args.calibration.resolve()
    require(not output.exists(), "Use a new output directory; existing outputs are preserved")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    intrinsics, baseline = validate_calibration(calibration)
    source_report_path = trial / "report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    rows = sorted(source_report["results"], key=lambda row: row["number"])
    numbers = [row["number"] for row in rows]
    require(numbers and len(set(numbers)) == len(numbers) and all(1 <= number <= 47 for number in numbers),
            "Require unique source frame numbers from 1 to 47")
    require(source_report["scale"] == 0.5, "This review expects the agreed half-resolution trial")
    # Preflight before creating any output: verify actual originals, network
    # inputs and raw disparity, not just dimensions recorded in metadata.
    prepared = []
    for row in rows:
        number, doe = row["number"], row["doe"]
        require(doe == (6 if number <= 23 else 0), f"Unexpected DOE at frame {number}")
        source = trial / f"left_{number:02d}_doe{doe}"
        for image_path in (row["file"], row["right_file"]):
            require("sn_CPC93630004R" in Path(image_path).parts, f"Source serial path mismatch: {image_path}")
        left_original, right_original = [read_image(row[key], cv2.IMREAD_GRAYSCALE) for key in ("file", "right_file")]
        require(left_original.shape == right_original.shape, f"Left/right source shape mismatch at {number}")
        actual_size = list(reversed(left_original.shape))
        require(actual_size == row["size"], f"Source size metadata mismatch at {number}")
        expected_size = [848, 480] if number == 24 else [1280, 800]
        require(actual_size == expected_size, f"Unexpected source dimensions at {number}")
        left, right = [read_image(source / name, cv2.IMREAD_GRAYSCALE) for name in ("left_input.png", "right_input.png")]
        disparity_path = source / "disparity_pixels.npy"
        disparity = np.load(disparity_path, allow_pickle=False)
        require(disparity.ndim == 2 and disparity.shape == left.shape == right.shape, f"Network/disparity shape mismatch at {number}")
        require(list(reversed(disparity.shape)) == row["input_size"] == [round(value * 0.5) for value in actual_size], f"Scale mismatch at {number}")
        for original, saved in ((left_original, left), (right_original, right)):
            resized = cv2.resize(original, tuple(row["input_size"]), interpolation=cv2.INTER_AREA)
            require(np.array_equal(resized, saved), f"Saved inference input does not match original at {number}")
        delta = int(row["right_device_timestamp"]) - int(row["device_timestamp"])
        require(delta == row["timestamp_delta_raw"] and abs(delta) <= 1, f"Invalid timestamp pairing at {number}")
        require((delta != 0) == (number == 7), f"Unexpected tentative pairing at {number}")
        require((source / "preview.jpg").is_file(), f"Missing source preview at {number}")
        prepared.append((row, source, left_original, disparity, actual_size, delta))
    output.mkdir(parents=True)
    shutil.copy2(calibration_path, output / "calibration.json")
    shutil.copy2(source_report_path, output / "source_trial_report.json")
    require(sha256(calibration_path) == sha256(output / "calibration.json"), "Calibration copy verification failed")
    save_image(output / "legend.png", cv2.cvtColor(np.asarray(legend()), cv2.COLOR_RGB2BGR))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": STATUS,
        "metric_depth_verified": False, "historical_ir_rectification_verified": False,
        "calibration_geometry": "identical_K_zero_IR_distortion_identity_rotation_negative_horizontal_baseline",
        "remapping_applied": False,
        "remapping_reason": "Supplied ideal rectified geometry implies identity mapping; it cannot rectify historical raw IR with unknown original distortion/extrinsics.",
        "historical_input_warning": "用户此前说明IR未校正；本标定呈现理想校正几何，不能证明历史图像校正状态。米制输出仅在历史输入已校正且标定适用时成立。",
        "source_trial": str(trial), "source_trial_report_sha256": sha256(source_report_path),
        "checkpoint": source_report["checkpoint"], "attention_backend": source_report["attention_backend"],
        "iterations": source_report["iterations"], "scale": source_report["scale"],
        "output_grid": "original_left_ir_image_grid",
        "output_size_matches_source": True, "native_resolution_inference": False,
        "restoration": "nearest_exact_disparity_resampling_times_width_ratio; metric_depth_not_scaled",
        "field_of_view_changed": False, "crop_applied": False, "rotation_applied": False,
        "registration_applied": False, "left_ir_to_device_depth_frame_verified": False,
        "depth_intrinsics_original": calibration["streams"]["depth"]["intrinsic"],
        "depth_to_color_original": calibration["extrinsics"]["depth_to_color"],
        "calibration_source": str(calibration_path), "calibration_copy": "calibration.json",
        "calibration_sha256": sha256(calibration_path), "postprocess_script_sha256": sha256(__file__),
        "calibration_serial": calibration["device"]["serial_number"], "calibration_size": [1280, 800],
        "baseline_m": baseline, "fx_half_resolution": intrinsics[0]["fx"] * 0.5,
        "fx_native_resolution": intrinsics[0]["fx"],
        "formula": "Z_m = fx_scaled * baseline_m / (disparity_pixels - (cx_left - cx_right)_scaled)",
        "validity_definition": "nearest_exact(inference finite-positive horizontal-overlap mask) AND native horizontal-overlap mask",
        "masks_are_confidence": False, "depth_dtype": "float32", "invalid_depth_value": "NaN",
        "shared_colour_range_m": [NEAR, FAR], "depth_values_clipped": False,
        "device_depth_pngs_read": False, "json_depth_unit_mm_applied_to_old_pngs": False,
        "input_pairs": len(rows), "metric_pairs": 0, "skipped_metric_pairs": [], "results": [],
    }
    for row, source, left, disparity, actual_size, delta in prepared:
        folder = source.name
        dest = output / folder
        dest.mkdir()
        raw_path = source / "disparity_pixels.npy"
        shutil.copy2(row["file"], dest / "left_native.png")
        disparity_native, valid = restore_native_disparity(disparity, actual_size)
        require(disparity_native.shape == left.shape, "Output must exactly match original image shape")
        np.save(dest / "disparity_pixels_native.npy", disparity_native)
        save_image(dest / "validity_mask.png", valid.astype(np.uint8) * 255)
        skipped = actual_size != report["calibration_size"]
        item = {
            "number": row["number"], "doe": row["doe"], "folder": folder,
            "status": "skipped_calibration_resolution_mismatch" if skipped else STATUS,
            "metric_depth_verified": False, "source_size": actual_size, "input_size": row["input_size"],
            "output_size": actual_size, "output_grid": "original_left_ir_image_grid",
            "output_size_matches_source": True, "native_resolution_inference": False,
            "disparity_native_file": "disparity_pixels_native.npy",
            "restoration_interpolation": "INTER_NEAREST_EXACT",
            "disparity_pixel_scale": actual_size[0] / disparity.shape[1],
            "metric_depth_value_scale": 1.0,
            "output_to_inference_pixel_centres": "u_in=(u_out+0.5)*W_in/W_out-0.5; v_in=(v_out+0.5)*H_in/H_out-0.5",
            "source_left": row["file"], "source_right": row["right_file"],
            "source_left_sha256": sha256(row["file"]), "source_right_sha256": sha256(row["right_file"]),
            "raw_disparity_path": str(raw_path), "raw_disparity_sha256": sha256(raw_path),
            "source_result_path": str(source / "result.json"), "source_preview_path": str(source / "preview.jpg"),
            "raw_disparity_href": local_href(raw_path, output),
            "source_preview_href": local_href(source / "preview.jpg", output),
            "device_timestamp": row["device_timestamp"], "right_device_timestamp": row["right_device_timestamp"],
            "timestamp_delta_raw": delta, "timestamp_unit": "unverified", "pairing_uncertain": delta != 0,
            "calibration_sha256": report["calibration_sha256"], "masks_are_confidence": False,
            "historical_ir_rectification_verified": False, "inference_seconds": row["inference_seconds"],
        }
        if skipped:
            item["skip_reason"] = "848x480 originals do not match the supplied 1280x800 calibration; no metric conversion assumed."
            report["skipped_metric_pairs"].append(row["number"])
            colour = cv2.resize(read_image(source / "disparity_colour.png", cv2.IMREAD_COLOR),
                                tuple(actual_size), interpolation=cv2.INTER_NEAREST_EXACT)
            save_image(dest / "disparity_colour_native.png", colour)
        else:
            fx = intrinsics[0]["fx"]
            offset = intrinsics[0]["cx"] - intrinsics[1]["cx"]
            require(valid.any(), f"No geometrically eligible pixels at {row['number']}")
            depth = np.full(disparity_native.shape, np.nan, dtype=np.float32)
            depth[valid] = fx * baseline / (disparity_native[valid] - offset)
            require(np.isfinite(depth[valid]).all() and (depth[valid] > 0).all(), "Invalid metric arithmetic")
            require(depth.shape == tuple(reversed(actual_size)), "Never return resized depth")
            np.save(dest / "depth_m_provisional.npy", depth)
            saved = np.load(dest / "depth_m_provisional.npy", allow_pickle=False)
            require(saved.dtype == np.float32 and np.array_equal(saved, depth, equal_nan=True), "Depth array write verification failed")
            save_image(dest / "validity_mask.png", valid.astype(np.uint8) * 255)
            colour = depth_colour(depth, valid)
            save_image(dest / "depth_colour.png", colour)
            item.update(fx_scaled=fx, principal_point_offset_scaled=offset, baseline_m=baseline,
                        intrinsics_on_output_grid=intrinsics[0],
                        depth_file="depth_m_provisional.npy", validity_mask="validity_mask.png",
                        valid_ratio=float(valid.mean()), depth_percentiles_m=np.percentile(depth[valid], [2, 50, 98]).tolist(),
                        below_colour_range_ratio=float((depth[valid] < NEAR).mean()),
                        above_colour_range_ratio=float((depth[valid] > FAR).mean()))
            report["metric_pairs"] += 1
        make_preview(left, colour, row["number"], row["doe"], skipped, delta != 0, dest / "preview.jpg")
        write_json(dest / "result.json", item)
        report["results"].append(item)
        print(f"{row['number']:02d} DOE={row['doe']}: {item['status']}", flush=True)
    expected_skips = [row["number"] for row in rows if row["size"] != report["calibration_size"]]
    require(report["metric_pairs"] == len(rows) - len(expected_skips) and report["skipped_metric_pairs"] == expected_skips,
            "Unexpected final output counts")
    write_json(output / "report.json", report)
    make_gallery(output, report)
    readme = f"""# 原尺寸双目深度输出

本次共{len(rows)}组，编号{numbers}；{report['metric_pairs']}组生成条件性米制深度，跳过米制编号{report['skipped_metric_pairs']}。打开index.html查看。

- depth_m_provisional.npy为float32光轴Z，单位米，NaN无效；1280×800，与原IR及标定尺寸一致。
- 内部推理640×400；最近邻恢复原图网格，视差乘2，米制深度值不变。保持原始方向、坐标原点、完整视场；不增加独立深度细节。
- disparity_pixels_native.npy使用原图像素单位；validity_mask.png为继承推理有效区并检查原图水平重叠的掩膜，不是置信度。
- depth_colour.png保持原尺寸，统一0.3～5米色标；preview.jpg仅为缩小排版图。
- 元数据、校准副本与SHA256在report.json及每帧result.json。已有输出保留。

当前米制输出以历史IR已正确校正、标定适用为前提，metric_depth_verified=false；未完成真实物体尺寸精度验收。图像恢复后使用原图内参，三维外参R/t不随尺寸改变。未做RGB配准：历史RGB1280×720与RGB标定1280×800尚未对应，IR_LEFT与DEPTH坐标关系也待确认。NPY已是米，不要再乘标定中的depth_unit_mm。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    write_json(output / "verification.json", {
        "status": "output_structure_and_arithmetic_verified_not_metric_accuracy",
        "metric_depth_verified": False, "frames": len(rows), "metric_arrays": report["metric_pairs"],
        "skipped_frames": report["skipped_metric_pairs"],
        "output_size_matches_source": True, "native_resolution_inference": False,
        "checks": ["original images decoded and sizes verified", "saved inference inputs exactly match half-scale originals", "left/right/disparity shapes verified", "native depth/disparity/mask dimensions match original", "native disparity pixel units rescaled; metric values not scaled", "timestamp pairing and DOE labels verified", "calibration ideal geometry verified", "calibration copy SHA-256 verified", "every output image decoded after write", "every depth npy reloaded and compared including NaNs"],
    })
    print(f"DONE: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

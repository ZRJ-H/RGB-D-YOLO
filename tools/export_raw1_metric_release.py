"""Export the 15 rectified Raw1 disparities with a user-assumed mm or m T.

Standalone dependencies: numpy, opencv-python, Pillow. Never reruns the network.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_image(path):
    a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    require(a is not None, f'Cannot decode {path}')
    return a


def png(path, data):
    ok, encoded = cv2.imencode('.png', data)
    require(ok, 'PNG encode failed')
    encoded.tofile(path)
    np.testing.assert_array_equal(read_image(path), data)


def preview(left, right, colour, number, median, fraction, path, unit, near, far):
    canvas = Image.new('RGB', (1280, 370), '#f4f6f2')
    draw = ImageDraw.Draw(canvas)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default()
    draw.text((12, 8), f'Pair {number:02d} | ASSUMED baseline 32.4841 {unit} | rectified grid 1280 x 720', fill='#173b3e', font=font)
    for i, (a, title) in enumerate(zip((left, right, colour), ('RECTIFIED LEFT', 'RECTIFIED RIGHT', 'CONDITIONAL DEPTH (m)'))):
        rgb = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB if a.ndim == 2 else cv2.COLOR_BGR2RGB)
        draw.text((12 + 424*i, 39), title, fill='#173b3e', font=font)
        canvas.paste(Image.fromarray(cv2.resize(rgb, (416, 234), interpolation=cv2.INTER_AREA)), (8+424*i, 66))
    draw.text((12, 306), f'Colour: warm {near:g} m -> cool {far:g} m; dark = invalid | median Z {median:.3f} m | valid {fraction:.1%}', fill='#173b3e', font=font)
    draw.text((12, 338), 'Unit assumed by user; calibration applicability / physical accuracy remain unverified.', fill='#985721', font=font)
    canvas.save(path, quality=92)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--calibration-text', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--translation-unit', choices=('mm', 'm'), default='mm')
    ap.add_argument('--compare-mm', type=Path, help='Verified mm-unit release for all-pixel scale comparison')
    args = ap.parse_args()
    unit = args.translation_unit
    unit_cn = '毫米' if unit == 'mm' else '米'
    scale_to_m = .001 if unit == 'mm' else 1.
    near, far = ((.3, 5.) if unit == 'mm' else (300., 5000.))
    colour_name = 'depth_colour_0p3_5m.png' if unit == 'mm' else 'depth_colour_300_5000m.png'
    cal_name = f'calibration_assumed_{unit}.json'
    if args.compare_mm:
        require(unit == 'm', '--compare-mm requires --translation-unit m')
        refcal = read_json(args.compare_mm / 'calibration/calibration_assumed_mm.json')
        require(refcal['translation_unit'] == 'mm' and abs(refcal['baseline_m']-.0324841)<1e-10, 'Wrong mm comparison baseline')
    src, out = args.input.resolve(), args.output.resolve()
    rect = src / 'rectified'
    require(Path('E:/gpt/_release').resolve() in out.parents, 'Release must be on E:/gpt/_release')
    archive = out.with_suffix('.zip')
    require(not out.exists() and not archive.exists(), 'Preserve existing outputs; use a new name')
    require(shutil.disk_usage(out.parent).free > 2_000_000_000, 'Need at least 2 GB free on E')
    oldcal = read_json(rect / 'calibration.json')
    text_bytes = args.calibration_text.read_bytes()
    decoded = text_bytes.decode('gbk')
    for key in ('cameraMatrixL', 'distCoeffL', 'cameraMatrixR', 'distCoeffR', 'R', 'T'):
        match = re.search(r'Mat\s+' + key + r'\s*=.*?<<\s*(.*?)\);', decoded, re.S)
        require(match is not None, f'Missing calibration {key}')
        numbers = np.array([float(v.strip()) for v in match.group(1).split(',')])
        np.testing.assert_array_equal(numbers, np.array(oldcal[key]).ravel())
    oldrect = read_json(rect / 'rectification.json')
    inference = read_json(rect / 'report.json')
    require(inference['status'] == 'completed' and inference['scale'] == 1, 'Require completed native inference')
    require(sorted(r['number'] for r in inference['results']) == list(range(1, 16)), 'Require all 15 pairs')
    T_m = np.asarray(oldcal['T'], np.float64) * scale_to_m
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        np.array(oldcal['cameraMatrixL']), np.array(oldcal['distCoeffL']),
        np.array(oldcal['cameraMatrixR']), np.array(oldcal['distCoeffR']),
        (1280, 720), np.array(oldcal['R']), T_m,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1, newImageSize=(1280, 720))
    for key, a in (('R1', R1), ('R2', R2)):
        np.testing.assert_allclose(a, oldrect[key], rtol=0, atol=1e-10)
    for key, a in (('P1', P1), ('P2', P2)):
        np.testing.assert_allclose(a[:, :3], np.array(oldrect[key])[:, :3], rtol=0, atol=1e-8)
    baseline = float(-P2[0, 3] / P2[0, 0])
    require(abs(baseline - 32.4841 * scale_to_m) < 1e-10, 'Unexpected baseline')
    fx, fy, cx, cy = P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]
    dcx = cx - P2[0, 2]
    oldmanifest = read_json(src / 'SHA256.json')
    out.mkdir()
    (out / 'provenance').mkdir()
    (out / 'calibration').mkdir()
    copied = []

    def copy(relative, target):
        source = src / relative
        require(digest(source) == oldmanifest[relative]['sha256'], f'Source changed: {relative}')
        dest = out / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        require(digest(dest) == digest(source), 'Copy hash mismatch')
        copied.append({'source': relative, 'target': target, 'sha256': digest(dest)})

    for p in sorted((src / 'source').glob('*.png')):
        copy('source/' + p.name, 'source/' + p.name)
    require(len(copied) == 45, 'Expected 45 original PNG')
    for name in ('report.json', 'rectification.json', 'calibration.json'):
        copy('rectified/' + name, 'provenance/historical_' + name)
    copy('rectified/rectification_diagnostic.json', 'calibration/rectification_diagnostic.json')
    for name in ('rectification.npz', 'left_source_support_mask.png', 'right_source_support_mask.png'):
        copy('rectified/' + name, 'calibration/' + name)
    (out / 'calibration/original_calibration.txt').write_bytes(text_bytes)
    shutil.copyfile(Path(__file__), out / 'provenance/export_raw1_metric_release.py')
    cal = {k: oldcal[k] for k in ('image_size', 'cameraMatrixL', 'cameraMatrixR', 'distCoeffL', 'distCoeffR', 'R', 'T')}
    cal.update(translation_unit=unit, unit_basis='user_requested_assumption_not_independently_verified',
               T_m=T_m.tolist(), baseline_m=baseline, original_file_sha256=digest(args.calibration_text),
               source_filename=args.calibration_text.name, physical_accuracy_validated=False)
    write_json(out / 'calibration' / cal_name, cal)
    write_json(out / 'calibration/rectification_metric.json', {
        **{k: a.tolist() for k, a in (('R1', R1), ('R2', R2), ('P1', P1), ('P2', P2), ('Q', Q))},
        'translation_unit': 'm', 'unit_basis': cal['unit_basis'], 'output_size': [1280, 720],
        'grid': 'rectified_left_camera', 'original_pixel_grid_preserved': False,
        'old_maps_and_disparity_reusable': True, 'physical_accuracy_validated': False,
        'valid_roi_left': list(roi1), 'valid_roi_right': list(roi2)})
    ls = read_image(rect / 'left_source_support_mask.png') > 0
    rs = read_image(rect / 'right_source_support_mask.png') > 0
    v, u = np.indices(ls.shape)
    verts_dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    rows, checks = [], []
    for n in range(1, 16):
        folder = f'pair_{n:02d}'
        dest = out / folder
        dest.mkdir()
        for name in ('left_input.png', 'right_input.png', 'disparity_pixels.npy'):
            copy(f'rectified/{folder}/{name}', f'{folder}/{name}')
        disparity = np.load(dest / 'disparity_pixels.npy', allow_pickle=False)
        require(disparity.shape == (720, 1280) and disparity.dtype == np.float32, 'Disparity grid/type mismatch')
        left, right = [read_image(dest / name) for name in ('left_input.png', 'right_input.png')]
        xr = u - disparity
        in_bounds = np.isfinite(disparity) & (disparity > 0) & (xr >= 0) & (xr <= 1279)
        safe_xr = np.clip(np.nan_to_num(xr, nan=0, posinf=0, neginf=0), 0, 1279)
        lo, hi = np.floor(safe_xr).astype(int), np.ceil(safe_xr).astype(int)
        valid = in_bounds & ls & rs[v, lo] & rs[v, hi] & (disparity - dcx > 0)
        z = np.full(disparity.shape, np.nan, np.float32)
        z[valid] = (fx * baseline / (disparity[valid].astype(np.float64) - dcx)).astype(np.float32)
        xyz = np.full((*disparity.shape, 3), np.nan, np.float32)
        xyz[..., 0][valid] = ((u[valid]-cx) * z[valid].astype(np.float64) / fx).astype(np.float32)
        xyz[..., 1][valid] = ((v[valid]-cy) * z[valid].astype(np.float64) / fy).astype(np.float32)
        xyz[..., 2] = z
        np.save(dest / 'depth_m_provisional.npy', z)
        np.save(dest / 'xyz_m_provisional.npy', xyz)
        png(dest / 'geometric_validity_mask.png', valid.astype(np.uint8)*255)
        mapped = np.zeros(z.shape, np.uint8)
        mapped[valid] = np.rint((1-np.clip((z[valid]-near)/(far-near), 0, 1))*255).astype(np.uint8)
        colour = cv2.applyColorMap(mapped, cv2.COLORMAP_TURBO)
        colour[~valid] = (30, 25, 20)
        png(dest / colour_name, colour)
        vertices = np.empty(int(valid.sum()), verts_dtype)
        for i, k in enumerate(('x', 'y', 'z')):
            vertices[k] = xyz[..., i][valid]
        for k in ('red', 'green', 'blue'):
            vertices[k] = left[valid]
        header = ('ply\nformat binary_little_endian 1.0\n'
                  'comment rectified LEFT optical coordinates, metres, X right Y down Z forward\n'
                  f'comment T assumed {unit} by user; physical accuracy unverified\n'
                  f'element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n'
                  'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii')
        ply = dest / 'cloud_rectified_left_gray_provisional.ply'
        with ply.open('xb') as f:
            f.write(header)
            vertices.tofile(f)
        # Independent Q reprojection checks all valid pixels and persisted PLY data.
        z2 = np.load(dest / 'depth_m_provisional.npy', allow_pickle=False)
        xyz2 = np.load(dest / 'xyz_m_provisional.npy', allow_pickle=False)
        qpoints = cv2.reprojectImageTo3D(disparity, Q)
        np.testing.assert_allclose(xyz2[valid], qpoints[valid], rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(xyz2[..., 2], z2)
        np.testing.assert_array_equal(np.isfinite(z2), valid)
        require(np.isnan(xyz2[~valid]).all(), 'Invalid XYZ must be NaN')
        with ply.open('rb') as f:
            require(f.read(len(header)) == header, 'PLY header mismatch')
            reread = np.fromfile(f, verts_dtype)
        np.testing.assert_array_equal(reread, vertices)
        reproj = max(np.max(np.abs(xyz2[..., 0][valid].astype(float)*fx/z2[valid]+cx-u[valid])),
                     np.max(np.abs(xyz2[..., 1][valid].astype(float)*fy/z2[valid]+cy-v[valid])))
        require(reproj < .001, 'Reprojection numerical check failed')
        row = dict(number=n, width=1280, height=720, valid_points=int(valid.sum()),
                   valid_fraction=float(valid.mean()), depth_p02_m=float(np.percentile(z[valid], 2)),
                   depth_median_m=float(np.median(z[valid])), depth_p98_m=float(np.percentile(z[valid], 98)))
        rows.append(row)
        write_json(dest / 'result.json', {**row, 'unit_basis': cal['unit_basis'], 'grid': 'rectified_left_camera',
                   'baseline_m': baseline, 'physical_accuracy_validated': False})
        checks.append(dict(number=n, Q_all_pixels_passed=True, PLY_all_vertices_passed=True,
                           nan_mask_passed=True, max_reprojection_error_pixels=float(reproj)))
        if args.compare_mm:
            reference = args.compare_mm / folder
            np.testing.assert_array_equal(disparity, np.load(reference/'disparity_pixels.npy', allow_pickle=False))
            np.testing.assert_array_equal(read_image(dest/'geometric_validity_mask.png'), read_image(reference/'geometric_validity_mask.png'))
            dz = np.load(reference/'depth_m_provisional.npy', allow_pickle=False)
            dxyz = np.load(reference/'xyz_m_provisional.npy', allow_pickle=False)
            np.testing.assert_allclose(z2, dz.astype(float)*1000, rtol=3e-7, atol=1e-7, equal_nan=True)
            np.testing.assert_allclose(xyz2, dxyz.astype(float)*1000, rtol=3e-7, atol=1e-7, equal_nan=True)
            checks[-1].update(mm_reference_disparity_and_masks_identical=True,
                             depth_and_xyz_1000x_mm_with_float32_rounding=True,
                             max_depth_scale_rounding_difference_m=float(np.max(np.abs(z2[valid]-dz[valid].astype(float)*1000))))
        preview(left, right, colour, n, row['depth_median_m'], row['valid_fraction'], dest/'preview.jpg', unit, near, far)
        print(f'Exported and verified {n}/15; median Z {row["depth_median_m"]:.4f} m', flush=True)
    with (out/'frames.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    overview = Image.new('RGB', (1280, 1850), '#f4f6f2')
    for i in range(15):
        with Image.open(out/f'pair_{i+1:02d}/preview.jpg') as im:
            overview.paste(im.resize((640, 185)), ((i%2)*640, (i//2)*230))
    overview = overview.crop((0, 0, 1280, 1795))
    overview.save(out/'overview_all15.jpg', quality=92)
    report = f'''# 15组双目图米制深度与点云报告（T按{unit_cn}）

## 本次结果与单位

全部15组已完成深度、逐像素XYZ与灰度PLY导出。用户指定本标定的平移参数暂按{unit_cn}解释：`T=(-32.4841,0,0) {unit}`，基线 **{baseline:.7f} m**。这是用户采用的单位假设，不是厂家单位证明或实物精度验收。

{'这是独立的米基线对照版本：相对于T按毫米的版本，基线、深度和XYZ尺度均为1000倍；视差、像素网格、有效点位置不变。不能因已输出米制文件就认定该假设符合真实相机。' if unit == 'm' else ''}

本次复用已实际运行的15组校正后标准FoundationStereo视差，**本轮没有重复网络推理**。单位选择不改变左右校正映射或视差，仅决定深度和三维坐标的尺度。原始未校正视差保留在上一交付中，不用于本次米制反投影。

## 输入、模型过程与输出

输入为15张2560×720左右拼接图，拆成左右1280×720，转灰度。按附件左右K、D、R、T做OpenCV双目校正（CALIB_ZERO_DISPARITY、alpha=1、保持完整1280×720画布）；灰度复制三通道，上下各补8行，标准FoundationStereo用多尺度卷积和DINOv2特征建立水平视差代价体，经聚合及GRU迭代32次后上采样、去补边，输出720×1280 float32视差。

已有网络运行采用原生分辨率、FP16自动混合精度与SDPA兼容层；权重SHA256为 `{inference['checkpoint_sha256']}`。没有保存网络内部特征或32次中间预测。随包[历史网络运行记录](provenance/historical_report.json)记录的是当时单位尚未明确的状态；本次单位与几何结果以本报告及[当前标定](calibration/{cal_name})为准。

原45张PNG保存在source目录，包括Color、设备Depth_Raw和Depth_Dispaly；设备Raw编码和比例仍未知，未当成米制真值。附件原文见[标定TXT](calibration/original_calibration.txt)，当前P1/P2/Q见[米制校正参数](calibration/rectification_metric.json)。

## 几何换算与坐标

```text
d = u_left - u_right，单位像素
Z = fx × B / (d - (cx_left - cx_right))
X = (u - cx_left) × Z / fx
Y = (v - cy_left) × Z / fy
```

本次校正后 `fx=fy={fx:.12f}`，`cx={cx:.12f}`，`cy={cy:.12f}`，左右主点相同，故 **Z={fx*baseline:.12f}/d 米**。depth[v,u]保存光轴Z；xyz[v,u,:]保存[X,Y,Z]，单位均为米，无效位置均NaN。PLY为同一XYZ的有效点，颜色为校正后左灰度。

输出网格为**校正后左图1280×720**，不是原拼接左半幅的像素网格。坐标原点为左光心，轴随R1旋转至校正后左相机坐标系：X右、Y下、Z前。需要原左相机坐标时可用`R1.T @ XYZ_rectified`，但这不会自动完成原始像素网格重采样。本次未做RGB配准或物体尺寸测量。

有效掩膜要求：视差正且有限、右图对应横坐标在范围内、左像素有完整校正源图支持、右对应位置的floor/ceil两列也均有源图支持。这排除了校正黑边；不是网络置信度，没有验证遮挡与左右一致性。深度颜色统一{near:g}～{far:g}米，近暖远冷，超范围仅显示截断，数组不截断。{'本米基线版本的色标也放大了1000倍，所以图形颜色近似相同不代表数值相同。' if unit == 'm' else ''}

## 全15组输出

每组包括视差、校正后左右图、depth_m_provisional.npy、xyz_m_provisional.npy、PLY、有效掩膜和预览。统计取整幅有效像素，不是货物尺寸或准确率。

|组号|有效点|Z中位数（米）|预览|深度|XYZ|点云|
|---:|---:|---:|---|---|---|---|
'''
    for row in rows:
        n=row['number']; p=f'pair_{n:02d}'
        report += f'|{n:02d}|{row["valid_points"]}|{row["depth_median_m"]:.4f}|[查看]({p}/preview.jpg)|[NPY]({p}/depth_m_provisional.npy)|[NPY]({p}/xyz_m_provisional.npy)|[PLY]({p}/cloud_rectified_left_gray_provisional.ply)|\n'
    report += '''
![全15组总览](overview_all15.jpg)

## 验证与限制

已逐组验证输出形状/类型、无效NaN、XYZ的Z与深度相同、独立Q矩阵反投影、像素数值回投、PLY全部顶点与灰度读回。见[核验记录](verification.json)及[统计CSV](frames.csv)。这些验证实现，不证明实物精度。

此前固定5840个稀疏匹配点的垂直残差绝对值P50/P90由校正前0.834/1.383像素变为校正后0.897/2.380像素，8/15组中位数变差。改变平移单位不会改善此现象，标定适用性仍须核对。参见[校正诊断](calibration/rectification_diagnostic.json)。结果均标为provisional，应在确认实际基线和适用标定后以已知距离/尺寸独立验收。

## 复现与分享

本包包含全部15组几何结果、实际校正后输入、45张源PNG、参数/映射/诊断与本次导出源码。无需模型权重即可查看NPY与PLY。解压整个包后打开本REPORT.md；保持目录结构即可显示图片和访问链接。网络权重和运行环境未随包提供。

导出脚本为[export_raw1_metric_release.py](provenance/export_raw1_metric_release.py)，依赖Python、NumPy、OpenCV、Pillow；以之前完整视差交付目录作为--input，原始标定TXT作为--calibration-text，E盘下新目录作为--output。--translation-unit mm或m明确选择T单位，默认mm；--compare-mm可指定已完成的毫米版以逐像素核验1000倍尺度。输出目录已存在时拒绝覆盖。
'''
    if args.compare_mm:
        report += '\n本版已逐组与毫米版核对：视差和有效掩膜一致，深度/XYZ为毫米版的1000倍（允许float32舍入误差）。详见核验记录。\n'
    (out/'REPORT.md').write_text(report, encoding='utf-8')
    (out/'START_HERE.md').write_text(f'解压后打开 [REPORT.md](REPORT.md)。T按{unit_cn}是假设；基线{baseline:g}米，深度/XYZ/PLY单位为米，校正适用性及实物精度未验证。\n', encoding='utf-8')
    links = re.findall(r'\]\(([^)]+)\)', report)
    write_json(out/'verification.json', {'status':'passed', 'pairs':15, 'new_network_calls':0,
        'reused_inference_pairs':15, 'unit_basis':cal['unit_basis'], 'baseline_m':baseline,
        'translation_unit_assumption':unit, 'colour_range_m':[near,far],
        'mm_comparison_source':str(args.compare_mm) if args.compare_mm else None,
        'physical_accuracy_validated':False, 'source_copies':copied, 'checks':checks, 'report_links':links})
    require(all((out/p).is_file() for p in links), 'Broken report link')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    manifest={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files}
    write_json(out/'SHA256.json', manifest)
    files.append(out/'SHA256.json')
    partial=out.with_suffix('.partial.zip')
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for p in files:
            zf.write(p, out.name+'/'+p.relative_to(out).as_posix(),
                     compress_type=zipfile.ZIP_STORED if p.suffix in ('.png','.jpg') else zipfile.ZIP_DEFLATED)
    print('Verifying archive hashes and CRC', flush=True)
    with zipfile.ZipFile(partial) as zf:
        require(len(zf.namelist())==len(set(zf.namelist()))==len(manifest)+1, 'ZIP count mismatch')
        for name, expected in manifest.items():
            data=zf.read(out.name+'/'+name)
            require(hashlib.sha256(data).hexdigest()==expected['sha256'] and len(data)==expected['bytes'], 'ZIP hash mismatch')
        require(json.loads(zf.read(out.name+'/SHA256.json'))==manifest, 'Archived manifest mismatch')
    partial.rename(archive)
    receipt={'status':'passed','archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),
             'files':len(manifest)+1,'depth_count':15,'xyz_count':15,'ply_count':15,'total_points':sum(r['valid_points'] for r in rows)}
    write_json(out.with_suffix('.archive_verification.json'), receipt)
    archive.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+archive.name+'\n', encoding='ascii')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

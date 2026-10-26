"""Rerun one original Raw1 stereo pair with corrected calibration and metre point clouds."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

sys.dont_write_bytecode = True
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from foundationstereo_runtime import load_model, sha256
from prepare_raw1_rectified import paired_diagnostic, read_image, save_image, transform_points


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rectification(cal):
    size = tuple(cal['image_size'])
    KL, KR, R = [np.array(cal[k], np.float64) for k in ('cameraMatrixL', 'cameraMatrixR', 'R')]
    DL, DR = [np.array(cal[k], np.float64) for k in ('distCoeffL', 'distCoeffR')]
    T = np.array(cal['T_m'], np.float64)
    np.testing.assert_allclose(T, np.array(cal['T_mm']) / 1000, rtol=0, atol=1e-15)
    np.testing.assert_allclose(R.T @ R, np.eye(3), rtol=0, atol=2e-6)
    require(size == (1280, 720) and T[0] < 0, 'Expected 1280x720 horizontal stereo calibration')
    R1, R2, autoP1, autoP2, _, _, _ = cv2.stereoRectify(
        KL, DL, KR, DR, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=size)
    # Preserve complete supplied translation, including small y/z components.
    tx = float((R2 @ T)[0])
    np.testing.assert_allclose((R2 @ T)[1:], 0, atol=1e-12)
    baseline = -tx
    np.testing.assert_allclose(baseline, np.linalg.norm(T), rtol=0, atol=1e-12)
    f, cx, cy = cal['rectified_focal_px'], float(autoP1[0, 2]), float(autoP1[1, 2])
    np.testing.assert_allclose(autoP1[:, :3], autoP2[:, :3], atol=1e-10)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)
    P1, P2 = np.column_stack((K, [0., 0., 0.])), np.column_stack((K, [f*tx, 0., 0.]))
    Q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1/tx, 0]], np.float64)
    maps = [cv2.initUndistortRectifyMap(k, d, r, p, size, cv2.CV_32FC1)
            for k, d, r, p in ((KL, DL, R1, P1), (KR, DR, R2, P2))]
    support, checks = [], []
    for (mx, my), k, d, r, p in zip(maps, (KL, KR), (DL, DR), (R1, R2), (P1, P2)):
        require(np.isfinite(mx).all() and np.isfinite(my).all(), 'Nonfinite remap')
        mask = (mx >= 0) & (mx < 1279) & (my >= 0) & (my < 719)
        support.append(mask)
        yy, xx = np.mgrid[0:720:19, 0:1280:19]
        take = mask[yy, xx]
        xy = np.column_stack((xx[take], yy[take]))
        source = np.column_stack((mx[yy[take], xx[take]], my[yy[take], xx[take]]))
        error = np.linalg.norm(transform_points(source, k, d, r, p)-xy, axis=1)
        require(error.max() < .01, 'Map roundtrip > .01 pixel')
        checks.append({'source_support_fraction': float(mask.mean()), 'map_roundtrip_max_px': float(error.max())})
    metadata = {name: a.tolist() for name, a in (('R1', R1), ('R2', R2), ('P1', P1), ('P2', P2), ('Q', Q))}
    metadata.update(focal_px=f, baseline_m=baseline, nominal_baseline_m=cal['nominal_baseline_m'],
                    rectified_translation_m=(R2@T).tolist(), output_size=list(size), unit='m',
                    grid='rectified_left_optical', principal_point_policy='stereoRectify CALIB_ZERO_DISPARITY',
                    focal_policy='explicit user focal 596 for both axes and views; P/Q/maps rebuilt consistently',
                    support_checks=checks, physical_accuracy_validated=False)
    return maps, support, metadata


def cloud_files(out, xyz, rgb, valid):
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices = np.empty(int(valid.sum()), dtype)
    for i, field in enumerate(('x', 'y', 'z')):
        vertices[field] = xyz[..., i][valid]
    for i, field in enumerate(('red', 'green', 'blue')):
        vertices[field] = rgb[..., i][valid]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment units metres; rectified left optical frame; X right Y down Z forward\n'
              f'element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii')
    with (out/'cloud_rgb_m.ply').open('xb') as stream:
        stream.write(header)
        vertices.tofile(stream)
    with (out/'cloud_rgb_m.ply').open('rb') as stream:
        require(stream.read(len(header)) == header, 'PLY header mismatch')
        np.testing.assert_array_equal(np.fromfile(stream, dtype), vertices)
    # PCL-compatible packed RGB field, written as a float bit-pattern.
    pcdtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
    pcd = np.empty(len(vertices), pcdtype)
    for field in ('x', 'y', 'z'):
        pcd[field] = vertices[field]
    packed = (vertices['red'].astype(np.uint32)<<16) | (vertices['green'].astype(np.uint32)<<8) | vertices['blue'].astype(np.uint32)
    pcd['rgb'] = packed.astype('<u4').view('<f4')
    pcdheader = ('# .PCD v0.7 - Point Cloud Data file format\n# units metres, rectified left optical frame\n'
                 'VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
                 f'WIDTH {len(pcd)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pcd)}\nDATA binary\n').encode('ascii')
    with (out/'cloud_rgb_m.pcd').open('xb') as stream:
        stream.write(pcdheader)
        pcd.tofile(stream)
    with (out/'cloud_rgb_m.pcd').open('rb') as stream:
        require(stream.read(len(pcdheader)) == pcdheader, 'PCD header mismatch')
        require(stream.read() == pcd.tobytes(), 'PCD data mismatch')


def previews(out, left, right, disparity, depth, valid, number):
    low, high = np.percentile(depth[valid], [2, 98])
    disp_low, disp_high = np.percentile(disparity[valid], [2, 98])
    colours = []
    for values, lo, hi, inverse, name in ((disparity, disp_low, disp_high, False, 'disparity_colour.png'),
                                        (depth, low, high, True, 'depth_colour.png')):
        t = np.nan_to_num(np.clip((values-lo)/max(hi-lo, 1e-6), 0, 1), nan=0)
        if inverse:
            t = 1-t
        colour = cv2.applyColorMap(np.rint(t*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colour[~valid] = (24, 24, 24)
        save_image(out/name, colour)
        colours.append(colour)
    canvas = Image.new('RGB', (1304, 854), '#f4f6f2')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    draw.text((12, 9), f'Pair {number:02d} | corrected calibration | f = 596 px | nominal baseline = 0.18025 m', fill='#173b3e', font=font)
    labels = ['RECTIFIED LEFT (COLOUR)', 'RECTIFIED RIGHT',
              f'DISPARITY: blue {disp_low:.1f} -> red {disp_high:.1f} px',
              f'DEPTH: red {low:.2f} -> blue {high:.2f} m']
    for i, (value, label) in enumerate(zip((left, right, *colours), labels)):
        x, y = 12+(i%2)*646, 48+(i//2)*391
        draw.text((x, y), label, fill='#173b3e', font=font)
        rgb = cv2.cvtColor(value, cv2.COLOR_GRAY2RGB if value.ndim==2 else cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(cv2.resize(rgb, (634, 357), interpolation=cv2.INTER_AREA)), (x,y+27))
    draw.text((12, 829), 'Native 1280 x 720 arrays | dark = invalid | physical accuracy requires a known-size reference.', fill='#684a27', font=font)
    canvas.save(out/'preview.jpg', quality=94)
    pair = np.hstack((left, cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)))
    for y in range(0, 720, 40):
        cv2.line(pair, (0,y), (2559,y), (50,220,50), 1)
    save_image(out/'rectified_epipolar_lines.png', pair)
    return [float(low), float(high)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-zip', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--number', type=int, default=1)
    args = parser.parse_args()
    out = args.output.resolve()
    require(Path('E:/gpt/_release').resolve() in out.parents, 'All release artifacts must be under E:/gpt/_release')
    require(not out.exists() and not out.with_suffix('.zip').exists(), 'Use a distinct output name')
    require(1 <= args.number <= 15, 'Expected Raw1 pair number 1..15')
    cal = json.loads(args.calibration.read_text(encoding='utf-8'))
    maps, support, rect = rectification(cal)
    name = f'Image_Color{args.number}.png'
    with zipfile.ZipFile(args.source_zip) as archive:
        data = archive.read(name)
    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    require(original.shape == (720,2560,3) and original.dtype == np.uint8, 'Source is not 2 x 1280x720')
    left_bgr, right_bgr = original[:,:1280].copy(), original[:,1280:].copy()
    originals = [cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) for a in (left_bgr,right_bgr)]
    left_colour = cv2.remap(left_bgr,*maps[0],cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    inputs = [cv2.remap(a,*m,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for a,m in zip(originals,maps)]
    diag = paired_diagnostic(args.number,*originals,
                            np.array(cal['cameraMatrixL']),np.array(cal['distCoeffL']),
                            np.array(cal['cameraMatrixR']),np.array(cal['distCoeffR']),
                            *[np.array(rect[k]) for k in ('R1','R2','P1','P2')])
    out.mkdir(parents=True)
    (out/'source').mkdir()
    (out/'provenance').mkdir()
    (out/'source'/name).write_bytes(data)
    save_image(out/'source/left_original_colour.png', left_bgr)
    save_image(out/'source/right_original.png',right_bgr)
    for filename, value in zip(('left_rectified.png','right_rectified.png'), inputs):
        save_image(out/filename,value)
    save_image(out/'left_rectified_colour.png',left_colour)
    for side, mask in zip(('left','right'),support):
        save_image(out/f'{side}_source_support_mask.png',mask.astype(np.uint8)*255)
    np.savez_compressed(out/'rectification_maps.npz',left_map_x=maps[0][0],left_map_y=maps[0][1],
                        right_map_x=maps[1][0],right_map_y=maps[1][1])
    write_json(out/'calibration.json',cal)
    write_json(out/'rectification.json',rect)
    write_json(out/'rectification_diagnostic.json',diag)
    source = {'zip':str(args.source_zip.resolve()),'zip_sha256':sha256(args.source_zip),'member':name,
              'member_sha256':hashlib.sha256(data).hexdigest(),'pair_number':args.number,
              'split_column':1280,'left_half_as_left':True,'selection':'first pair of previous Raw1 data, requested single-pair sample'}
    write_json(out/'source_manifest.json',source)
    print('Prepared corrected pair; loading network',flush=True)
    model, InputPadder, torch, metadata = load_model()
    report = {'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),**metadata,
              'scale':1,'iterations':32,'max_disp':416,'network_calls':0,'image_size':[1280,720],
              'input_preprocessing':'BGR to gray, calibrated remap, repeat gray in three channels; native scale; pad to 1280x736 then unpad',
              'source':source,'command_arguments':vars(args)|{'source_zip':str(args.source_zip),'calibration':str(args.calibration),'output':str(args.output)}}
    write_json(out/'inference.json',report)
    tensors = [torch.from_numpy(np.repeat(a[...,None],3,axis=2)).permute(2,0,1)[None].float().cuda() for a in inputs]
    padder = InputPadder(tensors[0].shape,divis_by=32,force_square=False)
    tensors = padder.pad(*tensors)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        prediction = model(*tensors,iters=32,test_mode=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    disparity = padder.unpad(prediction.float()).squeeze().cpu().numpy()
    require(disparity.shape==(720,1280) and disparity.dtype==np.float32,'Wrong disparity shape/type')
    np.save(out/'disparity_pixels.npy',disparity)
    report.update(status='completed',network_calls=1,inference_seconds=elapsed,completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'inference.json',report)
    print(f'Inference done: {elapsed:.3f}s; exporting geometry',flush=True)
    v,u = np.indices(disparity.shape)
    xr = u-disparity
    safe = np.clip(np.nan_to_num(xr,nan=0,posinf=0,neginf=0),0,1279)
    lo,hi = np.floor(safe).astype(int),np.ceil(safe).astype(int)
    valid = np.isfinite(disparity)&(disparity>0)&(xr>=0)&(xr<=1279)&support[0]&support[1][v,lo]&support[1][v,hi]
    require(valid.any(),'No valid points')
    P = np.array(rect['P1'])
    f,cx,cy,B = P[0,0],P[0,2],P[1,2],rect['baseline_m']
    depth = np.full(disparity.shape,np.nan,np.float32)
    depth[valid] = (f*B/disparity[valid].astype(float)).astype(np.float32)
    xyz = np.full((*disparity.shape,3),np.nan,np.float32)
    xyz[...,0][valid] = ((u[valid]-cx)*depth[valid].astype(float)/f).astype(np.float32)
    xyz[...,1][valid] = ((v[valid]-cy)*depth[valid].astype(float)/f).astype(np.float32)
    xyz[...,2] = depth
    np.save(out/'depth_m.npy',depth)
    np.save(out/'xyz_m.npy',xyz)
    save_image(out/'valid_mask.png',valid.astype(np.uint8)*255)
    cloud_files(out,xyz,cv2.cvtColor(left_colour,cv2.COLOR_BGR2RGB),valid)
    limits = previews(out,left_colour,inputs[1],disparity,depth,valid,args.number)
    persisted_xyz = np.load(out/'xyz_m.npy',allow_pickle=False)
    qpoints = cv2.reprojectImageTo3D(disparity,np.array(rect['Q']))
    np.testing.assert_allclose(persisted_xyz[valid],qpoints[valid],rtol=2e-6,atol=2e-6)
    np.testing.assert_array_equal(persisted_xyz[...,2],np.load(out/'depth_m.npy',allow_pickle=False))
    require(np.isnan(persisted_xyz[~valid]).all(),'Invalid XYZ must be NaN')
    np.testing.assert_array_equal(np.isfinite(depth),valid)
    reproject = max(np.max(np.abs(xyz[...,0][valid].astype(float)*f/depth[valid]+cx-u[valid])),
                    np.max(np.abs(xyz[...,1][valid].astype(float)*f/depth[valid]+cy-v[valid])))
    require(reproject<.001,'Reprojection error > .001px')
    result = {'status':'completed','pair_number':args.number,'valid_points':int(valid.sum()),'valid_fraction':float(valid.mean()),
              'depth_percentiles_m':dict(zip(('p02','p50','p98'),map(float,np.percentile(depth[valid],[2,50,98])))),
              'focal_px':f,'baseline_m':B,'nominal_baseline_m':cal['nominal_baseline_m'],'f_times_baseline_px_m':f*B,
              'unit':'m','coordinate_system':'rectified left optical: X right, Y down, Z forward',
              'validity_definition':'positive finite disparity; left remap full source support; right correspondence in bounds and both horizontal interpolation neighbours have full remap source support',
              'mask_is_confidence':False,'physical_accuracy_validated':False,'rgb_source':'same left image remapped with same left map',
              'point_cloud_filtering':'geometric mask only; no smoothing, voxel reduction or outlier removal',
              'depth_colour_limits_m':limits,'files':{'depth':'depth_m.npy','xyz':'xyz_m.npy','ply':'cloud_rgb_m.ply','pcd':'cloud_rgb_m.pcd'}}
    write_json(out/'result.json',result)
    write_json(out/'verification.json',{'status':'passed','Q_all_valid_pixels':True,'PLY_all_vertices_and_colours':True,
              'PCD_all_vertices_and_colours':True,'NaN_mask':True,'XYZ_Z_equals_depth':True,'reprojection_max_pixels':float(reproject),
              'source_member_sha256':sha256(out/'source'/name),'physical_accuracy_validated':False})
    for filename in ('run_corrected_stereo_sample.py','foundationstereo_runtime.py','prepare_raw1_rectified.py'):
        shutil.copyfile(Path(__file__).with_name(filename),out/'provenance'/filename)
    shutil.copyfile(Path(metadata['checkpoint']).with_name('cfg.yaml'),out/'provenance/cfg.yaml')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

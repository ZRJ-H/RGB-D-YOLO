"""Verify and package a single native standard-FoundationStereo rerun."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.export_native_stereo_batch import sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    out, old, archive = args.output.resolve(), args.previous.resolve(), args.archive.resolve()
    assert not any((out / name).exists() for name in ("REPORT.md", "comparison.json", "SHA256.json", "provenance"))
    assert not archive.exists() and out not in archive.parents
    for path in (archive.with_suffix(".zip.sha256"), archive.with_suffix(".verification.json")):
        assert not path.exists(), path
    source = read(out / "inference/report.json")
    assert len(source["results"]) == 1 and source["scale"] == 1 and source["iterations"] == 32
    row = source["results"][0]
    folder = f"left_{row['number']:02d}_doe{row['doe']}"
    native, geometry = out / "inference" / folder, out / "geometry" / folder
    old_native, old_geometry = old / "orbbec_native47" / folder, old / "geometry" / folder
    result, old_result = read(geometry / "result.json"), read(old_geometry / "result.json")
    assert result["verification"]["status"] == "passed"
    assert read(out / "geometry/verification.json")["verified_pairs"] == 1
    assert row["timestamp_delta_raw"] == 0
    assert result["source_hashes_sha256"]["original_left"] == old_result["source_hashes_sha256"]["original_left"]
    assert result["source_hashes_sha256"]["original_right"] == old_result["source_hashes_sha256"]["original_right"]
    comparisons = {}
    for key, current, previous in (
        ("disparity", native / "disparity_pixels.npy", old_native / "disparity_pixels.npy"),
        ("depth", geometry / "depth_m_provisional.npy", old_geometry / "depth_m_provisional.npy"),
        ("xyz", geometry / "xyz_m_provisional.npy", old_geometry / "xyz_m_provisional.npy"),
    ):
        a, b = np.load(current, allow_pickle=False), np.load(previous, allow_pickle=False)
        assert a.shape == b.shape and a.dtype == b.dtype
        common = np.isfinite(a) & np.isfinite(b)
        changed = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        diff = np.abs(a[common].astype(np.float64) - b[common].astype(np.float64))
        comparisons[key] = dict(shape=list(a.shape), arrays_equal=not bool(changed.any()),
                                changed_elements=int(changed.sum()), max_abs_difference=float(diff.max()),
                                mean_abs_difference=float(diff.mean()),
                                nan_masks_equal=bool(np.array_equal(np.isnan(a), np.isnan(b))),
                                new_sha256=sha256(current), previous_sha256=sha256(previous))
    ply_name = "cloud_ir_gray_provisional.ply"
    comparisons["ply"] = dict(new_sha256=sha256(geometry / ply_name), previous_sha256=sha256(old_geometry / ply_name))
    comparisons["ply"]["files_identical"] = comparisons["ply"]["new_sha256"] == comparisons["ply"]["previous_sha256"]
    comparison = dict(created_utc=datetime.now(timezone.utc).isoformat(), previous_directory=str(old),
                      same_source_images=True, comparisons=comparisons,
                      physical_accuracy_verified=False, problem_resolved=False)
    write(out / "comparison.json", comparison)

    provenance = out / "provenance"
    provenance.mkdir()
    checkpoint = Path(source["checkpoint"])
    for script in ("run_stereo_disparity_trial.py", "export_native_stereo_batch.py", "build_stereo_rerun_review.py"):
        shutil.copy2(PROJECT / "tools" / script, provenance / script)
    shutil.copy2(checkpoint.with_name("cfg.yaml"), provenance / "cfg.yaml")
    checkpoint_sha = sha256(checkpoint)
    previous_provenance = read(old / "provenance.json")
    previous_checkpoint = next(x for x in previous_provenance["files"] if x["path"] == str(checkpoint))
    assert checkpoint_sha == previous_checkpoint["sha256"]
    write(provenance / "run.json", dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
          checkpoint_bytes=checkpoint.stat().st_size, same_checkpoint_as_previous=True,
          model="standard FoundationStereo 23-51-11", strict_load=True, scale=1,
          iterations=32, max_disp=416, batch_size=1, precision="FP16 autocast",
          attention_backend=source["attention_backend"], executable=sys.executable,
          command=f'python -B -u tools/run_stereo_disparity_trial.py --numbers {row["number"]} --scale 1 --output "{out / "inference"}"'))
    shutil.copy2(old_native / "result.json", provenance / "previous_inference_result.json")
    shutil.copy2(old_geometry / "result.json", provenance / "previous_geometry_result.json")
    depth = np.load(geometry / "depth_m_provisional.npy")
    xyz = np.load(geometry / "xyz_m_provisional.npy")
    point_lines = []
    for u, v in ((640, 400), (800, 400), (1100, 700)):
        point = xyz[v, u]
        point_lines.append(f"| ({u}, {v}) | {float(depth[v,u]):.6f} | " + ", ".join(f"{float(x):.6f}" for x in point) + " |")
    same = all(comparisons[k]["arrays_equal"] for k in ("disparity", "depth", "xyz"))
    conclusion = ("本次确实重新调用了标准版模型。新旧视差、深度、XYZ逐元素一致，PLY文件字节也一致。重跑完成，但你指出的深度值或尺寸问题没有因重跑而消失。"
                  if same and comparisons["ply"]["files_identical"] else
                  "本次确实重新调用了标准版模型。新旧数值差异见comparison.json；尚不能认定实际深度或尺寸问题已经修复。")
    report = f"""# {folder} 标准版重跑

{conclusion}

![左右输入与本次深度](geometry/{folder}/preview.jpg)

## 本次运行

- 输入：第{row['number']}组DOE={row['doe']}，左右均为1280×800，设备时间戳均为`{row['device_timestamp']}`。原始文件哈希与此前记录一致。
- 模型：标准FoundationStereo `23-51-11/model_best_bp2.pth`，完整权重strict加载；权重SHA256 `{checkpoint_sha}`。
- 参数：原生1280×800，scale=1，32次迭代，max_disp=416，batch=1，FP16 autocast，Windows SDPA兼容层。
- 本次网络调用耗时{row['inference_seconds']:.3f}秒，不含权重加载和导出，也不是独立性能测试。
- 导出视差1280×800、深度1280×800、XYZ为800×1280×3、点云{result['pointcloud_points']:,}点。原图、旧结果和权重均保留。

## 可直接使用的文件

- [视差NPY](inference/{folder}/disparity_pixels.npy)
- [深度NPY，单位米](geometry/{folder}/depth_m_provisional.npy)
- [逐像素XYZ，单位米](geometry/{folder}/xyz_m_provisional.npy)
- [全场景灰度点云PLY，单位米](geometry/{folder}/cloud_ir_gray_provisional.ply)
- [左IR](inference/{folder}/left_input.png)、[右IR](inference/{folder}/right_input.png)、[深度伪彩色](geometry/{folder}/depth_colour_0p3_5m.png)
- [新旧数值对比](comparison.json)、[公式与点云核验](geometry/verification.json)、[本次运行日志](inference.log)

## 深度与尺寸核对

本次沿用已提供的标定：fx=fy=620.138794px，cx=637.5，cy=406，左右基线B=0.0952326965m。
`Z_m=59.05748955687802/disparity_px`，`X=(u-cx)Z/fx`，`Y=(v-cy)Z/fy`。
NPY和PLY已经是米，不应再次乘设备Depth PNG的`depth_unit_mm`；Z是光轴深度，不是光心到点的欧氏距离。
测两点三维距离使用`||XYZ[v2,u2]-XYZ[v1,u1]||`；两点的Z差仅表示光轴方向差，不能当作任意物体边长。

以下为原左IR网格的零起始像素示例，不自动代表箱体角点或完整尺寸：

| (u,v) | Z / m | XYZ / m |
|---|---:|---|
{chr(10).join(point_lines)}

逐像素深度公式、无效NaN、XYZ-Z、回投和PLY全部顶点/灰度读回通过。有效掩膜仅检查数值及水平重叠，不是准确性或遮挡置信度。
历史IR校正状态及标定适用关系仍未通过独立实物验收；本次没有绑定货物真值、重新测长宽高或宣称精度修复。
本次按要求交付这一组的标准版完整重跑结果；现有信息不足以判断实际读数偏差的具体原因，不擅自修改标定或人为调整数值。

## 保存与复现

完整文件使用相对路径，解压后打开本报告。`provenance/`含运行参数及主要脚本快照；它不是独立软件安装包，完整模型和环境仍依赖本机工程。
`SHA256.json`记录包内文件哈希，压缩包旁的`.verification.json`记录CRC和SHA256读回核验。此次核验不等于物理精度验收。
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", report)
    assert all((out / link).is_file() for link in links)
    write(out / "review_verification.json", dict(status="passed", native_pairs=1, report_links_checked=len(links),
          source_images_match_previous=True, checkpoint_matches_previous=True, all_numeric_outputs_equal_previous=same,
          geometry_verification="geometry/verification.json", physical_accuracy_verified=False, problem_resolved=False))
    manifest = {p.relative_to(out).as_posix(): dict(bytes=p.stat().st_size, sha256=sha256(p))
                for p in sorted(out.rglob("*")) if p.is_file()}
    write(out / "SHA256.json", manifest)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as package:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                package.write(path, (Path(out.name) / path.relative_to(out)).as_posix())
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert len(package.namelist()) == len(manifest) + 1 == len(set(package.namelist()))
        assert read(out / "SHA256.json") == json.loads(package.read(f"{out.name}/SHA256.json"))
        for name, record in manifest.items():
            data = package.read(f"{out.name}/{name}")
            assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
    archive_sha = sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    receipt = dict(status="passed", archive=str(archive), archive_bytes=archive.stat().st_size,
                   archive_sha256=archive_sha, hashed_content_files=len(manifest), archived_files=len(manifest)+1,
                   all_crc_and_sha256_verified=True)
    write(archive.with_suffix(".verification.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

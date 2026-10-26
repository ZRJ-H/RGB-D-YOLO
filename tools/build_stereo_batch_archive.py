"""Create a portable ZIP and verify every archived byte against its manifest."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

RELEASE_ROOT = Path(r"E:\gpt\_release")

README = """# 全47组双目推理结果包

先完整解压压缩包，再打开 [REPORT.md](REPORT.md)。报告包含模型输入、处理流程、输出定义、全部47组结果表及图片。用支持Markdown的阅读器打开即可；流程图需阅读器支持Mermaid，正文表格同时说明各步骤。

- `orbbec_native47/`：47组原生视差、左右输入图、掩膜及预览。
- `geometry/`：46组原生深度、逐像素XYZ和全场景IR灰度PLY，47组预览、4张总览、CSV及JSON。
- `provenance/`：运行脚本和配置的来源快照，用于追溯。
- `input_manifest.csv`：原始图片路径、时间戳差和哈希。
- `report_verification.json`、`geometry/verification.json`：已执行的文件/公式核验。
- `SHA256.json`：包内其他所有文件的SHA-256和字节数，不包含自身。

## 使用说明

报告中的预览、总览及数据链接使用相对路径，保持解压后的目录结构即可查看。NPY可用NumPy读取；PLY可用CloudCompare等支持PLY的软件打开。NPY是精确数值数据，PNG/JPG是预览。

深度/XYZ/PLY单位已经是米，不要再次乘设备Depth PNG的单位比例。XYZ为左IR光学坐标系，X向右、Y向下、Z向前；无效像素是NaN，点云只保存有效位置，颜色为左IR灰度，未做RGB配准。

第24组848×480缺匹配标定，只有视差和预览，没有米制深度/XYZ/PLY；该组已完成网络推理。第7组左右时间戳差1，仍保留暂定配对标记。米制几何依赖历史IR校正状态及标定适用关系，本包不是全批货物尺寸精度验收结果。

## 复现边界

本包是报告与结果交付，包含本次实际送入模型的94张左右IR输入图。未包含3.3GB模型权重、完整FoundationStereo工程或Python/GPU环境，也未包含同次采集但未参与模型推理的Color和设备Depth原图。

JSON、CSV、日志及源码核对笔记中的原始Windows绝对路径是历史来源记录，移机后不一定存在；这些外部路径不是查看本包报告、图片和数据的依赖。源码核对笔记中的本机网络源码链接需要原工程，源码快照不等于独立可运行的软件安装包。

需要重新运行时，在原项目或配置好等效依赖的环境中按报告末尾命令操作，并指定新输出目录。结果包内的模型权重SHA-256可用于核对所用权重。
"""


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--archive", type=Path,
                        help=r"Output ZIP (default: E:\gpt\_release\<source-directory-name>.zip)")
    args = parser.parse_args()
    source = args.source.resolve()
    archive = (args.archive or RELEASE_ROOT / f"{source.name}.zip").resolve()
    partial = archive.with_suffix(".partial.zip")
    receipt = archive.with_suffix(".verification.json")
    sidecar = archive.with_suffix(".zip.sha256")
    if any(p.exists() for p in (archive, partial, receipt, sidecar)):
        raise FileExistsError("Use a new archive name; existing deliveries are preserved")
    if source == archive or source in archive.parents:
        raise ValueError("Archive must be outside the source directory")
    assert source.is_dir()
    files = sorted(p for p in source.rglob("*") if p.is_file())
    assert all(not p.is_symlink() and source in p.resolve().parents for p in files)
    assert not (source / "START_HERE.md").exists() and not (source / "SHA256.json").exists()
    for name, count_key, expected in (("report_verification.json", "pairs", 47), ("geometry/verification.json", "metric_pairs", 46)):
        record = json.loads((source / name).read_text(encoding="utf-8"))
        assert record["status"] == "passed" and record[count_key] == expected
    archive.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(archive.parent).free < source_bytes * 1.1:
        raise OSError("Insufficient free space for conservative archive estimate")
    root = source.name
    manifest = {}
    print(f"Packaging {len(files)} existing files, {source_bytes:,} bytes", flush=True)
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as output:
        for index, path in enumerate(files, 1):
            rel = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo.from_file(path, f"{root}/{rel}")
            info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in (".png", ".jpg", ".jpeg") else zipfile.ZIP_DEFLATED
            info._compresslevel = 3
            h = hashlib.sha256()
            count = 0
            before = path.stat()
            with path.open("rb") as inp, output.open(info, "w", force_zip64=True) as dest:
                for block in iter(lambda: inp.read(4 * 1024 * 1024), b""):
                    h.update(block)
                    count += len(block)
                    dest.write(block)
            after = path.stat()
            assert before.st_size == after.st_size == count and before.st_mtime_ns == after.st_mtime_ns
            manifest[rel] = {"bytes": count, "sha256": h.hexdigest()}
            if index % 100 == 0 or index == len(files):
                print(f"Packed {index}/{len(files)}", flush=True)
        readme = README.encode("utf-8")
        output.writestr(f"{root}/START_HERE.md", readme)
        manifest["START_HERE.md"] = {"bytes": len(readme), "sha256": hashlib.sha256(readme).hexdigest()}
        output.writestr(f"{root}/SHA256.json", dump(manifest))
    print("Verifying every archived file (decompression, CRC, SHA-256)", flush=True)
    with zipfile.ZipFile(partial) as package:
        assert len(package.namelist()) == len(set(package.namelist())) == len(manifest) + 1
        archived_manifest = json.loads(package.read(f"{root}/SHA256.json"))
        assert archived_manifest == manifest
        assert set(package.namelist()) == {f"{root}/{p}" for p in manifest} | {f"{root}/SHA256.json"}
        for index, (rel, record) in enumerate(manifest.items(), 1):
            h = hashlib.sha256()
            count = 0
            with package.open(f"{root}/{rel}") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(block)
                    count += len(block)
            assert count == record["bytes"] and h.hexdigest() == record["sha256"], rel
            if index % 200 == 0 or index == len(manifest):
                print(f"Verified {index}/{len(manifest)}", flush=True)
    h = hashlib.sha256()
    with partial.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    partial.rename(archive)
    sidecar.write_text(f"{h.hexdigest()}  {archive.name}\n", encoding="ascii")
    evidence = {
        "status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive), "archive_bytes": archive.stat().st_size,
        "archive_sha256": h.hexdigest(), "source_directory": str(source),
        "source_files": len(files), "archived_files": len(manifest) + 1,
        "hashed_content_files": len(manifest), "source_bytes": source_bytes,
        "all_entries_crc_and_sha256_verified": True, "duplicate_entries": False,
        "original_outputs_modified": False,
        "inference_pairs": 47, "metric_depth_xyz_ply_pairs": 46,
        "reading_instructions": f"{root}/START_HERE.md", "report": f"{root}/REPORT.md",
    }
    receipt.write_bytes(dump(evidence))
    print(json.dumps(evidence, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

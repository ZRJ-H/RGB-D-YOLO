from __future__ import annotations

from pathlib import Path

from garbage_auto_annotation.provenance import build_source_record, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"garbage-data")

    assert sha256_file(sample) == (
        "02d365f6605cfd244538f37dd9da274bb180e782fcf2795aca671f66cd24d668"
    )


def test_source_record_contains_size_and_digest(tmp_path: Path) -> None:
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"x")

    record = build_source_record(
        name="demo",
        doi="10.example/demo",
        url="https://example.test/demo",
        files=[sample],
        license_name="CC-BY-4.0",
    )

    assert record["name"] == "demo"
    assert record["doi"] == "10.example/demo"
    assert record["url"] == "https://example.test/demo"
    assert record["license"] == "CC-BY-4.0"
    assert record["files"][0]["name"] == "sample.bin"
    assert record["files"][0]["bytes"] == 1
    assert len(record["files"][0]["sha256"]) == 64

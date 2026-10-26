"""Fetch the pinned public base in verified, resumable HTTP ranges on a data disk."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import requests

REPO = "ultralytics/ultralytics"
REGISTRY = "https://registry-1.docker.io/v2/" + REPO
TOKEN_URL = "https://auth.docker.io/token"
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def token():
    response = session().get(TOKEN_URL, params={"service": "registry.docker.io", "scope": f"repository:{REPO}:pull"}, timeout=45)
    response.raise_for_status()
    return response.json()["token"]


def fetch_metadata(reference, directory):
    auth = token()
    response = session().get(f"{REGISTRY}/manifests/{reference}", headers={"Authorization": "Bearer " + auth,
        "Accept": "application/vnd.docker.distribution.manifest.v2+json"}, timeout=60)
    response.raise_for_status()
    if "sha256:" + hashlib.sha256(response.content).hexdigest() != reference:
        raise RuntimeError("Base manifest digest mismatch")
    manifest = response.json()
    (directory / "base-manifest.json").write_bytes(response.content)
    digest = manifest["config"]["digest"]
    response = session().get(f"{REGISTRY}/blobs/{digest}", headers={"Authorization": "Bearer " + auth}, timeout=60)
    response.raise_for_status()
    if "sha256:" + hashlib.sha256(response.content).hexdigest() != digest:
        raise RuntimeError("Base config digest mismatch")
    (directory / "base-config.json").write_bytes(response.content)
    return manifest, auth


def fetch_range(task):
    digest, size, start, end, directory, auth = task
    target = directory / f"{digest.split(':')[1]}.{start}-{end}.part"
    count = end - start + 1
    if target.exists() and target.stat().st_size == count:
        return target
    for attempt in range(5):
        try:
            if attempt:
                auth = token()
            headers = {"Authorization": "Bearer " + auth, "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
            with session().get(f"{REGISTRY}/blobs/{digest}", headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Invalid HTTP Content-Range")
                elif not (response.status_code == 200 and start == 0 and count == size):
                    raise RuntimeError("Registry did not honor the byte range")
                written = 0
                with target.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        written += len(chunk)
                        if written > count:
                            raise RuntimeError("Range response exceeded expected length")
                        stream.write(chunk)
                if written != count:
                    raise RuntimeError("Incomplete range response")
            return target
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.base_config.read_text(encoding="utf-8"))
    manifest, auth = fetch_metadata(metadata["digest"], args.output)
    if args.probe:
        descriptor = max(manifest["layers"], key=lambda item: item["size"])
        # Probe both ends, with independent sessions, before relying on Range.
        for start in (0, descriptor["size"] - 4096):
            fetch_range((descriptor["digest"], descriptor["size"], start, start + 4095, args.output, auth))
        print("Verified registry byte-range support", flush=True)
        return
    block = 16 * 1024 * 1024
    tasks = []
    total = sum(layer["size"] for layer in manifest["layers"])
    for layer in manifest["layers"]:
        for start in range(0, layer["size"], block):
            tasks.append((layer["digest"], layer["size"], start, min(start + block, layer["size"]) - 1, args.output, auth))
    print(json.dumps({"layers": len(manifest["layers"]), "compressed_bytes": total, "ranges": len(tasks)}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch_range, task) for task in tasks}
        while pending:
            done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            received = sum(p.stat().st_size for p in args.output.glob("*.part"))
            print(json.dumps({"received_bytes": received, "total_bytes": total, "pending_ranges": len(pending)}), flush=True)
    for layer in manifest["layers"]:
        digest = layer["digest"].split(":")[1]
        target = args.output / (digest + ".tar.gz")
        if target.exists() and sha(target) == digest:
            continue
        with target.open("wb") as stream:
            for start in range(0, layer["size"], block):
                end = min(start + block, layer["size"]) - 1
                with (args.output / f"{digest}.{start}-{end}.part").open("rb") as part:
                    shutil.copyfileobj(part, stream, 4 * 1024 * 1024)
        if sha(target) != digest:
            raise RuntimeError("Compressed layer digest mismatch: " + digest)
    (args.output / "download-complete.json").write_text(json.dumps({"manifest_digest": metadata["digest"],
        "verified_layers": len(manifest["layers"]), "compressed_bytes": total}, indent=2), encoding="utf-8")
    print("All compressed layers verified", flush=True)


if __name__ == "__main__":
    main()

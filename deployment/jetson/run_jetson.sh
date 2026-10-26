#!/usr/bin/env bash
set -euo pipefail
image='yolo-depth:0.1.0-preview-jp6.2.1-arm64'
if [[ $(uname -m) != aarch64 ]]; then
  echo 'This launcher requires Jetson ARM64.' >&2
  exit 2
fi
if [[ ! -r /etc/nv_tegra_release ]] || ! grep -q 'R36 .*REVISION: 4.4' /etc/nv_tegra_release; then
  echo 'Target host: JetPack 6.2.1 / L4T 36.4.4. Verify the installed JetPack before proceeding.' >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash run_jetson.sh NEW_OUTPUT_DIRECTORY [doctor|self-test|perception arguments...]' >&2
  exit 2
fi
output=$1
shift
mkdir -- "$output"
output=$(realpath -- "$output")
if [[ $# -eq 0 ]]; then
  set -- self-test /outputs/acceptance
fi
exec docker run --rm --network none --runtime nvidia --shm-size 1g \
  --mount "type=bind,source=$output,target=/outputs" \
  "$image" "$@"

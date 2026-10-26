from __future__ import annotations

import argparse
import json
from pathlib import Path

from garbage_auto_annotation.factory_seed import write_factory_focus_view


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a domain-weighted training view from factory seed and negative images."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    print(json.dumps(write_factory_focus_view(args.dataset_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

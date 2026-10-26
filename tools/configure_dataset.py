"""Create a portable local YOLO YAML without changing the archived dataset."""
import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', default='dataset.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = yaml.safe_load((root / args.source).read_text(encoding='utf-8'))
    data['path'] = str(root)
    for split in ('train', 'val', 'test'):
        value = data.get(split)
        if not isinstance(value, str):
            continue
        # Historical train lists contain machine-absolute paths. Resolve each
        # entry against this dataset using its images/ suffix, never the old PC.
        source = root / value
        if source.suffix == '.txt' and source.is_file():
            entries = []
            for line in source.read_text(encoding='utf-8').splitlines():
                line = line.strip().replace('\\', '/')
                if not line:
                    continue
                marker = 'images/'
                if marker not in line:
                    raise ValueError(f'Cannot relocate image-list entry: {line}')
                image = (root / (marker + line.split(marker, 1)[1])).resolve()
                if not image.is_relative_to(root) or not image.is_file():
                    raise ValueError(f'Missing or unsafe image path: {image}')
                entries.append(image.as_posix())
            target = args.output.resolve().with_name(args.output.stem + '_' + split + '.txt')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('x', encoding='utf-8') as stream:
                stream.write('\n'.join(entries) + '\n')
            data[split] = str(target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
    print(args.output.resolve())


if __name__ == '__main__':
    main()

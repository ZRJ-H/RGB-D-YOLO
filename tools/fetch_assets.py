"""Download SHA256-verified release bundles and restore their original files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def restore(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read('files.json'))
        for item in manifest:
            target = (destination / item['path']).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Unsafe bundle path')
            if target.exists():
                if digest(target) == item['sha256']:
                    continue
                raise FileExistsError(f'Refusing to overwrite {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.partial')
            with bundle.open('objects/' + item['sha256']) as src, temporary.open('wb') as out:
                shutil.copyfileobj(src, out, 8 * 1024 * 1024)
            if digest(temporary) != item['sha256']:
                raise ValueError(f'Checksum mismatch: {target}')
            temporary.rename(target)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('groups', nargs='+', help='models, reviewed, raw, public or all')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'assets/manifest.json').read_text(encoding='utf-8'))
    selected = set(catalog['groups']) if 'all' in args.groups else set(args.groups)
    if selected - set(catalog['groups']):
        parser.error('Unknown group')
    args.cache.mkdir(parents=True, exist_ok=True)
    for name in sorted(selected):
        spec = catalog['groups'][name]
        parts = []
        for part in spec['parts']:
            target = args.cache / part['name']
            if not target.is_file() or digest(target) != part['sha256']:
                temp = target.with_suffix(target.suffix + '.partial')
                print('Downloading', part['name'], flush=True)
                urllib.request.urlretrieve(catalog['base_url'] + '/' + part['name'], temp)
                if digest(temp) != part['sha256']:
                    raise ValueError('Download checksum mismatch')
                temp.replace(target)
            parts.append(target)
        archive = args.cache / (name + '.bundle.zip')
        with archive.open('wb') as out:
            for part in parts:
                with part.open('rb') as src:
                    shutil.copyfileobj(src, out, 8 * 1024 * 1024)
        if digest(archive) != spec['sha256']:
            raise ValueError('Assembled archive checksum mismatch')
        restore(archive, args.destination)
        print('Restored', name, flush=True)


if __name__ == '__main__':
    main()

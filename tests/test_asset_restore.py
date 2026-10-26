import hashlib
import json
import zipfile

import pytest

from tools.fetch_assets import restore


def bundle(path, name, data=b'example'):
    sha = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('files.json', json.dumps([{'path': name, 'sha256': sha}]))
        archive.writestr('objects/' + sha, data)


def test_restore_verified_bytes_and_repeat(tmp_path):
    source = tmp_path / 'bundle.zip'
    bundle(source, 'models/example.bin')
    restore(source, tmp_path / 'out')
    restore(source, tmp_path / 'out')
    assert (tmp_path / 'out/models/example.bin').read_bytes() == b'example'


def test_restore_rejects_path_traversal(tmp_path):
    source = tmp_path / 'bundle.zip'
    bundle(source, '../escape.bin')
    with pytest.raises(ValueError, match='Unsafe'):
        restore(source, tmp_path / 'out')
    assert not (tmp_path / 'escape.bin').exists()


def test_restore_preserves_existing_different_file(tmp_path):
    source = tmp_path / 'bundle.zip'
    bundle(source, 'example.bin')
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'example.bin').write_bytes(b'keep me')
    with pytest.raises(FileExistsError):
        restore(source, out)
    assert (out / 'example.bin').read_bytes() == b'keep me'


def test_restore_rejects_corrupt_object(tmp_path):
    source = tmp_path / 'bundle.zip'
    sha = hashlib.sha256(b'expected').hexdigest()
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('files.json', json.dumps([{'path': 'example.bin', 'sha256': sha}]))
        archive.writestr('objects/' + sha, b'wrong')
    with pytest.raises(ValueError, match='Checksum'):
        restore(source, tmp_path / 'out')
    assert not (tmp_path / 'out/example.bin').exists()

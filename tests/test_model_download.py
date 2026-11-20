import hashlib
from pathlib import Path
import runpy
import zipfile

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/download_model.py'))


def artifact(tmp_path, extra=False):
    content = b'unit fixture model bytes'
    archive = tmp_path / 'model.zip'
    with zipfile.ZipFile(archive, 'w') as file:
        file.writestr('model/weights.bin', content)
        if extra:
            file.writestr('../unexpected.txt', b'outside')
    manifest = {'archive_root': 'model', 'archive_bytes': archive.stat().st_size,
                'archive_sha256': MODULE['sha256'](archive),
                'files': {'weights.bin': hashlib.sha256(content).hexdigest()}}
    return archive, manifest


def test_install_verifies_bytes_and_is_idempotent(tmp_path):
    archive, manifest = artifact(tmp_path)
    destination = tmp_path / 'installed'
    MODULE['install_archive'](archive, destination, manifest)
    assert MODULE['verify_installed'](destination, manifest)
    MODULE['install_archive'](archive, destination, manifest)
    assert (destination / 'weights.bin').read_bytes() == b'unit fixture model bytes'


def test_damaged_archive_does_not_create_model(tmp_path):
    archive, manifest = artifact(tmp_path)
    archive.write_bytes(archive.read_bytes()[:-2])
    destination = tmp_path / 'installed'
    with pytest.raises(ValueError, match='SHA-256'):
        MODULE['install_archive'](archive, destination, manifest)
    assert not destination.exists()


def test_unexpected_archive_members_never_escape_destination(tmp_path):
    archive, manifest = artifact(tmp_path, extra=True)
    destination = tmp_path / 'installed'
    with pytest.raises(ValueError, match='состав'):
        MODULE['install_archive'](archive, destination, manifest)
    assert not (tmp_path / 'unexpected.txt').exists()
    assert not destination.exists()


def test_existing_model_files_are_preserved(tmp_path):
    archive, manifest = artifact(tmp_path)
    destination = tmp_path / 'installed'
    destination.mkdir()
    (destination / 'custom.txt').write_text('user data', encoding='utf-8')
    with pytest.raises(FileExistsError):
        MODULE['install_archive'](archive, destination, manifest)
    assert (destination / 'custom.txt').read_text(encoding='utf-8') == 'user data'

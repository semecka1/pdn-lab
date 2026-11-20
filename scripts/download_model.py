"""Загрузка проверенного архива готовой модели из GitHub Release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def verify_installed(destination: Path, manifest: dict) -> bool:
    return all((destination / name).is_file() and sha256(destination / name) == digest
               for name, digest in manifest['files'].items())


def install_archive(archive: Path, destination: Path, manifest: dict) -> None:
    """Проверить архив и установить в новую папку, сохранив существующие файлы."""
    destination = destination.resolve()
    if verify_installed(destination, manifest):
        return
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Папка {destination} уже содержит другую или неполную модель. Выберите новую --output-dir.')
    if archive.stat().st_size != manifest['archive_bytes'] or sha256(archive) != manifest['archive_sha256']:
        raise ValueError('Размер или SHA-256 архива модели не совпал с models/release.json.')
    names = list(manifest['files'])
    if not all(Path(name).name == name and name not in ('.', '..') for name in names):
        raise ValueError('Некорректные имена файлов в манифесте.')
    expected = {manifest['archive_root'] + '/' + name for name in names}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ('.' + destination.name + '-' + uuid.uuid4().hex)
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive) as source:
            if set(source.namelist()) != expected or len(source.namelist()) != len(expected):
                raise ValueError('Неожиданный состав архива модели.')
            for name in names:
                # Извлекаются только явно перечисленные файлы, без extractall.
                with source.open(manifest['archive_root'] + '/' + name) as incoming:
                    with (staging / name).open('xb') as output:
                        shutil.copyfileobj(incoming, output, length=1024 * 1024)
                if sha256(staging / name) != manifest['files'][name]:
                    raise ValueError(f'Контрольная сумма файла {name} не совпала.')
        if destination.exists():
            destination.rmdir()  # Только пустая папка; чужие файлы не удаляются.
        staging.rename(destination)
    finally:
        if staging.exists():
            assert staging.resolve().parent == destination.parent.resolve()
            assert staging.resolve() != destination and staging.name.startswith('.' + destination.name + '-')
            shutil.rmtree(staging)


def download(manifest: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (manifest['archive_sha256'] + '.zip')
    if archive.exists() and sha256(archive) == manifest['archive_sha256']:
        return archive
    partial = archive.with_suffix('.download')
    request = urllib.request.Request(manifest['url'], headers={'User-Agent': 'pdn-lab1/3.0.0'})
    last = time.monotonic()
    written = 0
    print(f"Загрузка модели: {manifest['archive_bytes'] / 1024**2:.0f} МиБ.", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            written += len(chunk)
            output.write(chunk)
            if time.monotonic() - last >= 10:
                print(f"Загружено {written / 1024**2:.0f} / {manifest['archive_bytes'] / 1024**2:.0f} МиБ", flush=True)
                last = time.monotonic()
    if written != manifest['archive_bytes'] or sha256(partial) != manifest['archive_sha256']:
        raise ValueError('Загруженный архив не прошёл проверку размера/SHA-256; модель не установлена.')
    partial.replace(archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models/rubert-pdn-v2')
    parser.add_argument('--archive', type=Path, help='Локальный архив вместо загрузки из сети')
    parser.add_argument('--check-only', action='store_true', help='Только проверить уже установленную модель')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'models/release.json').read_text(encoding='utf-8'))
    if verify_installed(args.output_dir, manifest):
        print('Готовая модель установлена; контрольные суммы всех файлов совпадают.')
        return
    if args.check_only:
        raise SystemExit('Готовая модель отсутствует или отличается от опубликованной версии.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Выходная папка уже содержит файлы. Укажите новую --output-dir.')
    archive = args.archive or download(manifest, ROOT / '.cache/model-download')
    install_archive(archive, args.output_dir, manifest)
    print(f'Модель установлена: {args.output_dir.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error))

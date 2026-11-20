from pathlib import Path
import shutil
import uuid

import pytest


@pytest.fixture
def tmp_path():
    # mkdir(mode=0o700) в стандартной Windows-фикстуре может исключать
    # дополнительный sandbox SID. Наследуем разрешения рабочей папки.
    root = Path(__file__).resolve().parents[1] / ".cache" / "tests"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    resolved = path.resolve()
    if resolved.is_relative_to(root.resolve()) and resolved != root.resolve():
        shutil.rmtree(resolved)

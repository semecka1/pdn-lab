import pytest

from pdn_lab.detector import Detector
from pdn_lab.model import NameModel, default_model_directory, default_data_directory
from pdn_lab.schema import ATTRIBUTES, PROJECT_ROOT


def test_saved_model_end_to_end_and_long_tail():
    directory = PROJECT_ROOT / "models" / "rubert-pdn"
    if not (directory / "training_manifest.json").exists():
        pytest.skip("Интеграционная проверка запускается после обучения модели")
    pytest.importorskip("torch")
    detector = Detector(NameModel.load(directory))
    assert detector.detect("Иван получил кредит") == [["NAME", 0, 4]]
    demo = (PROJECT_ROOT / "examples" / "demo.txt").read_text(encoding="utf-8")
    entities = detector.detect(demo)
    assert {entity[0] for entity in entities} == set(ATTRIBUTES)
    assert len(entities) == 8
    tail = "Иванов Иван Иванович. СНИЛС 112-233-445 95."
    text = "В этом разделе приведены общие сведения о порядке работы системы. " * 150 + tail
    found = detector.detect(text)
    expected = {("SURNAME", "Иванов"), ("NAME", "Иван"), ("PATRONYMIC", "Иванович"), ("SNILS", "112-233-445 95")}
    actual_tail = {(label, text[start:end]) for label, start, end in found if start >= len(text) - len(tail)}
    assert actual_tail == expected


def test_finetuned_model_email_conflict_and_patronymic():
    directory = PROJECT_ROOT / 'models' / 'rubert-pdn-v2'
    if not (directory / 'training_manifest.json').exists():
        pytest.skip('Интеграционная проверка требует завершённого дообучения v2')
    pytest.importorskip('torch')
    if (PROJECT_ROOT / 'models' / 'active.json').exists():
        assert default_model_directory() == directory
        assert default_data_directory() == PROJECT_ROOT / 'data' / 'hard-v2'
    detector = Detector(NameModel.load(directory))
    text = "Карта: 4111111111111111@example.org; Отчество: Иванович; Имя: Иван; Фамилия: Иванов."
    actual = {(label, text[start:end]) for label, start, end in detector.detect(text)}
    assert actual == {('EMAIL','4111111111111111@example.org'),('PATRONYMIC','Иванович'),('NAME','Иван'),('SURNAME','Иванов')}
    text = "При проверке указан только Иванович. Выпущено 30 винтов; давление 101325 Па."
    assert [(label,text[start:end]) for label,start,end in detector.detect(text)] == [('PATRONYMIC','Иванович')]


def test_saved_model_semantic_context_v3():
    directory = default_model_directory()
    if not (directory / 'training_manifest.json').exists():
        pytest.skip('Интеграционная проверка требует сохранённой модели')
    pytest.importorskip('torch')
    detector = Detector(NameModel.load(directory))
    texts = [
        'Лев живёт в саванне. вера в победу и надежда на успех помогают команде.',
        'В названии модели упомянут Петров.',
        'Писатель САЛТЫКОВ-ЩЕДРИН создал произведение.',
        'Товарный знак «Петров» зарегистрирован; контакт: Иванов Иван Иванович.',
    ]
    results = detector.detect_many(texts)
    assert results[0] == results[1] == []
    assert [(e.label,texts[2][e.start:e.end]) for e in results[2]] == [('SURNAME','САЛТЫКОВ-ЩЕДРИН')]
    assert {(e.label,texts[3][e.start:e.end]) for e in results[3]} == {('SURNAME','Иванов'),('NAME','Иван'),('PATRONYMIC','Иванович')}

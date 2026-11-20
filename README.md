# Распознавание и обезличивание персональных данных

Python-программа находит персональные данные и заменяет их вымышленными значениями. Профиль `lab` охватывает восемь типов из ЛР‑1, профиль `course` — десять типов чётного варианта курсовой. Профиль `all` объединяет все тринадцать. Версия 4 использует прежние веса RuBERT v2 и добавляет алгоритмы ЛР‑2. [Методы обезличивания и воспроизведение КМ‑3](docs/anonymization.md)

Команда `predict` возвращает список троек `[тип, начало, конец]`. Команда `anonymize` возвращает объект с полями `text` и `entities`. Координаты в обоих случаях относятся к исходной строке Python, начинаются с нуля и имеют исключённую правую границу.

После установки зависимостей и модели выполните из корня проекта

```powershell
.\.venv\Scripts\python.exe -m pdn_lab anonymize --text "Саша подписала заявление. Ответ пришёл от Саши на sasha@example.org."
.\.venv\Scripts\python.exe -m pdn_lab anonymize --profile course --file examples/course.txt
```

Если от ЛР‑1 уже получены координаты, функция `anon("Иван получил кредит", [["NAME", 0, 4]])` из пакета `pdn_lab` вернёт только заменённый текст. Для этого режима достаточно базовой установки проекта. Новая обработка выбирает замену заново.

## Быстрый запуск на Windows

Нужны **Git, Python 3.12** и доступ в интернет для установки зависимостей и однократного скачивания модели. Поддерживается Python 3.11–3.13. Команды выполняются в **PowerShell**, из папки проекта. Активация виртуального окружения не требуется.

### 1. Клонировать проект

```powershell
git clone https://github.com/casinocool/pdn-lab1.git
cd pdn-lab1
```

### 2. Установить зависимости

Вариант для любого компьютера, включая компьютеры без NVIDIA:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[model,dev]"
```

Если команда `py` недоступна, замените первую строку на `python -m venv .venv`, предварительно проверив версию через `python --version`.

Для NVIDIA с драйвером, совместимым с CUDA 12.8, вместо строки установки CPU-сборки используйте:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Также можно воспользоваться `setup.ps1`: `./setup.ps1 -Python "C:\путь\к\python.exe" -CpuOnly` либо без `-CpuOnly` для NVIDIA. Скрипт не меняет политику выполнения PowerShell; при запрете запуска используйте ручные команды выше.

### 3. Скачать готовую модель

```powershell
.\.venv\Scripts\python.exe scripts/download_model.py
```

Скрипт получает архив из [Release v3.0.0](https://github.com/casinocool/pdn-lab1/releases/tag/v3.0.0), проверяет SHA-256 архива и каждого файла, затем устанавливает модель в `models/rubert-pdn-v2`. Архив содержит около 0,7 ГБ данных; для установки оставьте примерно 2 ГБ свободного места. Повторная команда проверяет уже установленную модель. Папка с другой или неполной моделью не перезаписывается.

При ручном скачивании архива можно установить его без повторной загрузки:

```powershell
.\.venv\Scripts\python.exe scripts/download_model.py --archive "C:\Downloads\rubert-pdn-v2.zip"
.\.venv\Scripts\python.exe scripts/download_model.py --check-only
```

### 4. Запустить распознавание

```powershell
# Один текст
.\.venv\Scripts\python.exe -m pdn_lab predict --text "Иван получил кредит"

# Готовый пример со всеми восемью атрибутами
.\.venv\Scripts\python.exe -m pdn_lab predict --file examples/hard_v2.txt

# Животное Лев, названия, составная фамилия и реальный человек
.\.venv\Scripts\python.exe -m pdn_lab predict --file examples/context_v3.txt

# Вводить свои тексты построчно
.\.venv\Scripts\python.exe -m pdn_lab predict --interactive
```

Для первого текста ожидается:

```json
[["NAME", 0, 4]]
```

Для готовых файлов сохранены эталоны: [восемь атрибутов](examples/hard_v2.expected.json) и [контекст имён](examples/context_v3.expected.json). В интерактивном режиме завершение на Windows — `Ctrl+Z`, затем `Enter`. Ввод обрабатывается одной строкой; многострочный текст передавайте через `--file`.

По умолчанию устройство выбирается автоматически. `--device cpu` принудительно включает CPU, `--device cuda` — NVIDIA. После установки модели распознавание работает локально без сети; тексты пользователя во внешние сервисы не отправляются.

## Запуск в VS Code

1. Откройте **папку клонированного репозитория** в VS Code.
2. Установите рекомендованные расширения **Python** и **Python Debugger** от Microsoft.
3. Выполните шаги установки зависимостей и скачивания модели выше во встроенном терминале.
4. Откройте «Запуск и отладка» (`Ctrl+Shift+D`), выберите конфигурацию и нажмите `F5`.

Готовые конфигурации:

- **ПДн: интерактивный ввод** — ввод своих текстов в терминале.
- **ПДн: готовый пример** — обработка `examples/demo.txt`.
- **ПДн: контекст имён (v3)** — проверка животных, названий и составных фамилий.
- **ПДн: проверка качества** — оценка тестовой выборки; сначала подготовьте данные по следующему разделу.

Интерпретатор `.venv/Scripts/python.exe` уже указан в настройках проекта. Для перезапуска после изменения кода/модели: `Shift+F5`, затем `F5`.

## Подготовка данных и проверка качества

Для обычного распознавания обучающие данные не нужны. Для оценки готовой модели восстановите исходный и расширенный наборы:

```powershell
.\.venv\Scripts\python.exe -m pdn_lab prepare
.\.venv\Scripts\python.exe scripts/prepare_hard_data.py
.\.venv\Scripts\python.exe -m pdn_lab evaluate --output reports/test.json --save-predictions
```

Первый шаг загружает [factRuEval-2016](https://github.com/dialogue-evaluation/factRuEval-2016) зафиксированной ревизии и создаёт синтетические примеры. Контрольные суммы проверяются. Второй шаг добавляет сложные примеры и сохраняет исходный test без изменений: **6574 train**, **1286 validation**, **1095 test**, отдельные **1000 контрольных текстов**. Существующие наборы не перезаписываются. Подготовка не требует локальных отчётов предыдущих запусков.

Основная метрика — weighted F1 по BIO-меткам восьми атрибутов без `O` в усреднении. Токены `O` участвуют в подсчёте ошибок. Дополнительно вычисляются строгие метрики целых сущностей: совпасть должны тип и обе границы. Подробности — в [методике проверки](docs/testing.md) и [подготовке к защите](docs/defense.md).

### Стресс-тест

```powershell
# Восстановить уточнённые 42377 размеченных сценариев
.\.venv\Scripts\python.exe scripts/audit_stress_cases.py

# Проверить готовую модель и сохранить новый протокол
.\.venv\Scripts\python.exe scripts/stress_test.py --cases data/stress-audited-v2/cases.jsonl --model-dir models/rubert-pdn-v2 --output-dir reports/stress

# Отдельные контрольные тексты
.\.venv\Scripts\python.exe scripts/stress_test.py --cases data/hard-v2/processed/holdout.jsonl --model-dir models/rubert-pdn-v2 --output-dir reports/control

# Ручные примеры для проверки контекста ФИО
.\.venv\Scripts\python.exe scripts/stress_test.py --cases data/context-v3/cases.jsonl --model-dir models/rubert-pdn-v2 --output-dir reports/context
```

Отчёты создаются локально в `reports/` и исключены из Git. Для повторного запуска укажите новую выходную папку. Старые скрипты `compare_*` и `verify_v2_artifacts.py` предназначены для сохранённых локальных экспериментов и требуют их `reports/`/`runs/`; для чистого клона используйте самостоятельные команды выше.

## Обучение модели с нуля

Этот раздел нужен для повторения обучения. Готовый запуск выше использует уже обученные веса из Release.

```powershell
.\.venv\Scripts\python.exe -m pdn_lab prepare
.\.venv\Scripts\python.exe -m pdn_lab train --data-dir data --output-dir models/rubert-pdn-retrained --epochs 4 --device cuda
.\.venv\Scripts\python.exe scripts/prepare_hard_data.py
.\.venv\Scripts\python.exe -m pdn_lab train --data-dir data/hard-v2 --continue-from models/rubert-pdn-retrained --output-dir models/rubert-pdn-v2-retrained --learning-rate 1e-5 --epochs 3 --device cuda
.\.venv\Scripts\python.exe -m pdn_lab evaluate --data-dir data/hard-v2 --model-dir models/rubert-pdn-v2-retrained --output reports/retrained-test.json
```

Если данные уже подготовлены, пропустите повторные команды подготовки. Для CPU замените `--device cuda` на `--device cpu`; обучение будет медленнее. При нехватке видеопамяти используйте `--batch-size 2 --accumulation 4`.

Основа — [DeepPavlov/rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased), ревизия `4036cab694767a299f2b9e6492909664d9414229`. Дообучаются все веса. Лучшая эпоха выбирается только по validation, test для выбора эпохи не используется. Для проверки заново обученной модели передавайте `--model-dir models/rubert-pdn-v2-retrained` либо поменяйте `model_dir` в `models/active.json`.

## Тесты

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Интеграционные тесты с реальными весами выполняются после скачивания модели; без неё они помечаются как пропущенные. Остальные проверки используют поддерживаемые исходные фикстуры и не требуют результатов прошлых запусков.

## Атрибуты и ограничения

| Метка | Атрибут | Метод |
|---|---|---|
| `SURNAME` | Фамилия | RuBERT + контекст и восстановление составных фамилий |
| `NAME` | Имя | RuBERT + ограниченные фильтры животного/объектного контекста |
| `PATRONYMIC` | Отчество | RuBERT + явные подписи полей |
| `EMAIL` | E-mail | Формат, `email-validator`, приоритет целого адреса перед вложенными номерами |
| `INN` | ИНН | 10/12 цифр и контрольные разряды |
| `PHONE` | Телефон | Российский формат; дополнительные проверки через `phonenumbers` |
| `CARD` | Банковская карта | 13–19 цифр и алгоритм Луна |
| `SNILS` | СНИЛС | 11 цифр и контрольное число |

Поддержаны типичные пробелы, дефисы и Unicode-разделители номеров. Для СНИЛС исторические основы до `001001998` включительно не принимаются как подтверждённые контрольным числом. Российский телефон из 10 цифр требует явной подписи; иностранный номер — префикса `+` и проверки библиотеки. E-mail проверяется без DNS, поддержаны Unicode/IDN и quoted local-part; существование ящика не проверяется.

Контрольная сумма номера не подтверждает его выдачу человеку. Полный e-mail защищён от вложенного цифрового совпадения. Явные артикулы и номера заказа исключаются. Омонимы, OCR, опечатки, редкие имена и новые форматы документов могут давать ошибки. Контекстные правила уточняют найденные моделью ФИО и не восстанавливают все полностью пропущенные имена. Конечный синтетический стресс-набор не является оценкой всех возможных текстов.

## Структура проекта

- `src/pdn_lab/` — распознавание, данные, обучение, метрики и CLI.
- `scripts/` — установка готовой модели, расширение данных и стресс-проверки.
- `tests/` — исходные тесты и поддерживаемые фикстуры.
- `examples/` — тексты и ожидаемые результаты.
- `data/context-v3/` — вручную размеченные контекстные примеры.
- `models/active.json` — выбор модели и данных; `models/release.json` — адрес архива и SHA-256.
- `docs/` — объяснение методов и порядок проверки.

Веса, подготовленные датасеты, виртуальное окружение, кэши и локальные протоколы не хранятся в Git. Веса распространяются отдельным Release; исходные датасеты и сложные примеры восстанавливаются командами проекта. `requirements-lock.txt` содержит версии проверенного Windows-окружения с CUDA 12.8; для CPU используйте установку выше.

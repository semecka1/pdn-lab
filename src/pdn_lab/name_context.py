"""Уточнение уже найденных ФИО по полям и узким семантическим контекстам."""
import re

from .schema import Entity, NAME_ATTRIBUTES

KEYS = {"фамилия": "SURNAME", "surname": "SURNAME", "last_name": "SURNAME", "имя": "NAME", "name": "NAME", "first_name": "NAME", "отчество": "PATRONYMIC", "patronymic": "PATRONYMIC", "middle_name": "PATRONYMIC"}
KEY_PATTERN = "|".join(sorted(KEYS, key=len, reverse=True))
FIELD = re.compile(rf"(?<!\w)[\"']?(?P<key>{KEY_PATTERN})[\"']?\s*[:=]\s*[\"']?(?P<value>[^\W\d_]+(?:[\-’'][^\W\d_]+)*\.?)", re.I)
XML_FIELD = re.compile(rf"<(?P<key>{KEY_PATTERN})>\s*(?P<value>[^\W\d_]+(?:[\-’'][^\W\d_]+)*\.?)\s*</", re.I)
QUANTITY = re.compile(r"(?<!\w)[+-]?[0-9]+(?:[.,][0-9]+)?\s*(?P<unit>МПа|кПа|Па|КБ|МБ|ГБ|кг|мг|км|мм|см|руб|коробок|винтов|шайбы|шайб|деталей|изделий)(?!\w)")
IDENTIFIER_ROLE = re.compile(r"(?:название\s+(?:переменной|модели)|имя\s+(?:переменной|файла)|переменная|идентификатор)\s*[:=]?\s*$", re.I)

WORD = r"[^\W\d_]+(?:[-‑’'][^\W\d_]+)*"
TITLE_ROLE = (
    r"товарн(?:ый|ого|ым)\s+знак(?:а|ом)?|(?:торгов(?:ая|ой)\s+)?марк(?:а|и|у)|"
    r"бренд(?:а|ом)?|компани(?:я|и|ю)|фирм(?:а|ы|у)|магазин(?:а|е)?|"
    r"модел(?:ь|и|ью|ю)|проект\s+называется|ООО|ОАО|ЗАО|ПАО|АО|"
    r"(?:название|названии|наименование|имя|(?:поле\s+)?name)\s+"
    r"(?:модели|компании|фирмы|товара|изделия|проекта|файла|папки|переменной)"
)
QUOTED_TITLE = re.compile(
    rf"(?<!\w)(?:{TITLE_ROLE})\s*[:=—-]?\s*"
    r"(?P<value>«[^»\r\n]{1,160}»|\"[^\"\r\n]{1,160}\"|“[^”\r\n]{1,160}”|„[^“\r\n]{1,160}“|'[^'\r\n]{1,160}')",
    re.I,
)
IDENTIFIER_VALUE = re.compile(
    rf"(?<!\w)(?:(?:в\s+)?(?:названи[еиюя]|имя)\s+"
    rf"(?:модели|файла|переменной|папки|компании|фирмы|товара|изделия|проекта)|"
    rf"модель(?=\s*[:=])|(?:товарн(?:ый|ого)\s+знак(?:а)?|бренд))\s*[:=—-]?\s*"
    rf"(?:(?:упомянут[ао]?|указан[ао]?|значится|используется)\s+)?(?P<value>{WORD})",
    re.I,
)
FILE_NAME = re.compile(rf"(?<!\w){WORD}\.(?:txt|csv|xml|json|py|pdf|docx|xlsx|log)(?!\w)", re.I)
COMPOUND = re.compile(r"(?<![\w’'‑-])[^\W\d_]+(?:[-‑][^\W\d_]+)+(?![\w’'‑-])")
FUNCTION_PARTS = frozenset({'на', 'над', 'под', 'при', 'по', 'из', 'в', 'с'})
OFFICE_PREFIXES = frozenset({'премьер', 'вице', 'экс', 'генерал'})
PERSON_ROLE = re.compile(
    r"(?:писател[ьяю]|поэт(?:а|у)?|автор(?:а|у)?|сотрудник(?:а|у)?|инженер(?:а|у)?|"
    r"докладчик(?:а|у)?|клиент(?:а|у)?|гражданин(?:а|у)?|журналист(?:а|у)?|"
    r"профессор(?:а|у)?|врач(?:а|у)?|исследовател[ьяю])\s*[:—-]?\s*$", re.I,
)
INITIALS_AFTER = re.compile(r"^\s+[А-ЯЁA-Z]\.\s*(?:[А-ЯЁA-Z]\.)?(?!\w)")
INITIALS_BEFORE = re.compile(r"(?<!\w)[А-ЯЁA-Z]\.\s*(?:[А-ЯЁA-Z]\.)?\s*$")
LION_ANIMAL = re.compile(
    r"^\s*(?:[—–-]\s*|это\s+)?(?:(?:крупный|дикий|африканский|взрослый|крупное|дикое|хищное)\s+){0,3}"
    r"(?:хищник\s+(?:из\s+)?семейства\s+кошачьих|животное(?!\w)|крупная\s+кошка(?!\w))|"
    r"^\s+(?:жив[её]т|обитает|спит)\s+в\s+(?:африканской\s+)?(?:саванн[ае]|прайде|вольере)(?!\w)|"
    r"^\s+рычит\s+в\s+клетке(?!\w)",
    re.I,
)
ANIMAL_ROLE = re.compile(r"(?:вид|название\s+животного|животное)\s*[:—-]\s*$", re.I)
WILD_HABITAT = re.compile(r"\bв\s+(?:африканской\s+)?саванне\s*$", re.I)
PREY = re.compile(r"^\s+охотится\s+на\s+антилоп", re.I)


def _field_values(text: str):
    for pattern in (FIELD, XML_FIELD):
        for match in pattern.finditer(text):
            start, end = match.span('value')
            if text[start:end].endswith('.') and end - start > 2:
                end -= 1
            yield KEYS[match.group('key').lower()], start, end


def _neighbouring_person(text: str, entity: Entity, entities: list[Entity]) -> bool:
    """Непосредственное соседство ФИО; контекст не переносится через предложение."""
    for other in entities:
        if other == entity or other.label not in NAME_ATTRIBUTES:
            continue
        if other.end <= entity.start:
            between = text[other.end:entity.start]
        elif entity.end <= other.start:
            between = text[entity.end:other.start]
        else:
            continue
        if len(between) <= 5 and not between.strip(' \t,'):
            return True
    return False


def _expand_surnames(text: str, entities: list[Entity], fields) -> list[Entity]:
    result = list(entities)
    for match in COMPOUND.finditer(text):
        parts = re.split('[-‑]', match.group())
        if not all(part[0].isupper() and (part.istitle() or part.isupper()) for part in parts):
            continue
        if any(part.casefold() in FUNCTION_PARTS for part in parts) or parts[0].casefold() in OFFICE_PREFIXES:
            continue
        start, end = match.span()
        matching = [e for e in result if e.label in NAME_ATTRIBUTES and start <= e.start < e.end <= end]
        if not any(e.label == 'SURNAME' for e in matching):
            continue
        full = Entity('SURNAME', start, end)
        labelled = any(label == 'SURNAME' and a <= start and end <= b for label,a,b in fields)
        supported = labelled or _neighbouring_person(text, full, result)
        supported = supported or bool(PERSON_ROLE.search(text[max(0,start-50):start]))
        supported = supported or bool(INITIALS_BEFORE.search(text[max(0,start-15):start]))
        supported = supported or bool(INITIALS_AFTER.search(text[end:end+15]))
        if supported:
            result = [e for e in result if e not in matching]
            result.append(full)
    return result


def refine_name_fields(text: str, entities: list[Entity]) -> list[Entity]:
    fields = list(_field_values(text))
    result = list(entities)
    for label, start, end in fields:
        matching = [e for e in result if e.label in NAME_ATTRIBUTES and start <= e.start < e.end <= end]
        if matching:
            result = [e for e in result if e not in matching]
            result.append(Entity(label, start, end))
    result = _expand_surnames(text, result, fields)
    quantities = [match.span('unit') for match in QUANTITY.finditer(text)]
    titles = [match.span('value') for pattern in (QUOTED_TITLE, IDENTIFIER_VALUE) for match in pattern.finditer(text)]
    titles.extend(match.span() for match in FILE_NAME.finditer(text))
    accepted = []
    for entity in result:
        if any(start <= entity.start and entity.end <= end for start,end in quantities):
            continue
        # «Фамилия: ...» внутри названия товара остаётся частью названия.
        if any(start <= entity.start and entity.end <= end for start,end in titles):
            continue
        protected = any(start <= entity.start and entity.end <= end for _,start,end in fields)
        if not protected:
            if IDENTIFIER_ROLE.search(text[max(0,entity.start-80):entity.start]):
                continue
            if entity.label == 'NAME' and text[entity.start:entity.end].casefold() == 'лев':
                animal = LION_ANIMAL.search(text[entity.end:entity.end+90])
                animal = animal or ANIMAL_ROLE.search(text[max(0,entity.start-50):entity.start])
                if text[entity.start:entity.end] == 'лев':
                    animal = animal or (WILD_HABITAT.search(text[max(0,entity.start-40):entity.start]) and PREY.search(text[entity.end:entity.end+40]))
                if not _neighbouring_person(text, entity, result) and animal:
                    continue
        accepted.append(entity)
    return sorted(set(accepted), key=lambda e: (e.start, e.end))

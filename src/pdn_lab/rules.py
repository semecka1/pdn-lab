from __future__ import annotations

from functools import lru_cache
import re
from email_validator import EmailNotValidError, validate_email
import phonenumbers
from .schema import Entity
from .validators import SEPARATOR, digits, valid_card, valid_inn, valid_phone, valid_snils

ATOM = r"[\w!#$%&'*+/=?^`{|}~\-]+"
DOMAIN_LABEL = r"[^\W_](?:[\w\-]*[^\W_])?"
EMAIL = re.compile(rf'(?<![\w.])(?:"(?:[^"\\\r\n]|\\.)+"|{ATOM}(?:\.{ATOM})*)@(?:{DOMAIN_LABEL}\.)+{DOMAIN_LABEL}(?![\w\-])', re.I)
INN = re.compile(r"(?<!\w)(?:[0-9]{12}|[0-9]{10})(?!\w)")
SNILS = re.compile(rf"(?<!\w)[0-9]{{3}}{SEPARATOR}*[0-9]{{3}}{SEPARATOR}*[0-9]{{3}}{SEPARATOR}*[0-9]{{2}}(?!\w)")
CARD_PATTERNS = [
    re.compile(r"(?<!\w)[0-9]{13,19}(?!\w)"),
    re.compile(rf"(?<!\w)[0-9]{{4}}(?:{SEPARATOR}+[0-9]{{4}}){{2}}{SEPARATOR}+[0-9]{{1,4}}(?!\w)"),
    re.compile(rf"(?<!\w)[0-9]{{4}}(?:{SEPARATOR}+[0-9]{{4}}){{3}}{SEPARATOR}+[0-9]{{1,3}}(?!\w)"),
    re.compile(rf"(?<!\w)[0-9]{{4}}{SEPARATOR}+[0-9]{{6}}{SEPARATOR}+[0-9]{{5}}(?!\w)"),
]
PHONE = re.compile(rf"(?<!\w)(?:\+7|[78]){SEPARATOR}*(?:\([0-9]{{3}}\)|[0-9]{{3}}){SEPARATOR}*[0-9]{{3}}{SEPARATOR}*[0-9]{{2}}{SEPARATOR}*[0-9]{{2}}(?!\w)")
LOCAL_PHONE = re.compile(rf"(?<!\w)(?:\([0-9]{{3}}\)|[0-9]{{3}}){SEPARATOR}*[0-9]{{3}}{SEPARATOR}*[0-9]{{2}}{SEPARATOR}*[0-9]{{2}}(?!\w)")
CONTEXT = {'INN': r'(?:инн|налоговый номер)', 'SNILS': r'(?:снилс|страховой номер)', 'CARD': r'(?:карта|карты|карту|card)', 'PHONE': r'(?:телефон|телефона|тел\.?|мобильный|phone)'}
NON_PERSONAL = re.compile(r"(?:артикул(?: товара)?|номер заказа|код (?:товара|детали)|идентификатор(?:\s+[^\W\d_]+){0,5})\s*[:№=—-]?\s*$", re.I)


def overlaps(left: Entity, right: Entity) -> bool:
    return left.start < right.end and right.start < left.end


@lru_cache(maxsize=8192)
def _valid_email(value: str) -> bool:
    try:
        validate_email(value, check_deliverability=False, allow_quoted_local=True, allow_smtputf8=True)
        return True
    except EmailNotValidError:
        return False


def find_emails(text: str) -> list[Entity]:
    result = []
    for match in EMAIL.finditer(text):
        start, end = match.span()
        at = text.rfind('@', start, end)
        # Позиции URL-параметров сохраняются в исходной строке.
        url = re.search(r"https?://[^\s<>\"']*$", text[:at], flags=re.I)
        if url:
            question = text.find('?', url.start(), at)
            if question >= 0:
                boundary = max(question, text.rfind('&', question, at))
                equals = text.find('=', boundary + 1, at)
                if equals >= 0:
                    start = max(start, equals + 1)
        caption = re.match(r'(?:e-?mail|email|почта)=', text[start:at], flags=re.I)
        if caption:
            start += caption.end()
        # Сохраняем апостроф в o'connor, исключаем парную внешнюю кавычку.
        if start < end and text[start] == "'" and text[end:end+1] == "'":
            start += 1
        if _valid_email(text[start:end]):
            result.append(Entity('EMAIL', start, end))
    return result


def _context(text: str, start: int) -> tuple[set[str], bool]:
    prefix = text[max(0, start - 100):start]
    # Неподписанная строка заголовков TSV не является подписью первой ячейки.
    if '\n' in prefix and not prefix.rstrip().endswith((':', '=', '№')):
        prefix = prefix.rsplit('\n', 1)[-1]
    explicit = {kind for kind, pattern in CONTEXT.items() if re.search(pattern + r'\s*[:№=—-]?\s*$', prefix, re.I)}
    return explicit, bool(NON_PERSONAL.search(prefix)) and not explicit


def find_structured(text: str) -> list[Entity]:
    # Полный валидный e-mail защищён от вложенных цифровых кандидатов.
    emails = find_emails(text)
    candidates = []
    specifications = [('SNILS', SNILS, valid_snils, 50), ('INN', INN, valid_inn, 40)]
    specifications += [('CARD', pattern, valid_card, 30) for pattern in CARD_PATTERNS]
    specifications += [('PHONE', PHONE, valid_phone, 20)]
    for label, pattern, validator, priority in specifications:
        for match in pattern.finditer(text):
            entity = Entity(label, match.start(), match.end())
            if any(overlaps(entity, email) for email in emails) or not validator(match.group()):
                continue
            explicit, non_personal = _context(text, entity.start)
            if non_personal or (explicit and label not in explicit):
                continue
            candidates.append((priority + (100 if label in explicit else 0), entity))
    for match in LOCAL_PHONE.finditer(text):
        explicit, non_personal = _context(text, match.start())
        number = digits(match.group())
        entity = Entity('PHONE', match.start(), match.end())
        if 'PHONE' in explicit and not non_personal and valid_phone('7' + number) and not any(overlaps(entity, email) for email in emails):
            candidates.append((120, entity))
    for match in phonenumbers.PhoneNumberMatcher(text, 'RU'):
        if not match.raw_string.startswith('+') or match.number.country_code == 7:
            continue
        entity = Entity('PHONE', match.start, match.end)
        explicit, non_personal = _context(text, entity.start)
        if not non_personal and (not explicit or 'PHONE' in explicit) and not any(overlaps(entity, email) for email in emails):
            candidates.append((20 + (100 if 'PHONE' in explicit else 0), entity))
    result = list(emails)
    for _, entity in sorted(candidates, key=lambda pair: (-pair[0], -(pair[1].end-pair[1].start), pair[1].start)):
        if not any(overlaps(entity, accepted) for accepted in result):
            result.append(entity)
    return sorted(result, key=lambda e: e.start)

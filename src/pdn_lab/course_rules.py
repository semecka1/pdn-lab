"""Additional structured entities for the even-numbered coursework variant."""
import ipaddress
import re

from .kladr import catalog
from .rules import CONTEXT, NON_PERSONAL, overlaps
from .schema import Entity

SPACE = r"[ \t\u00a0\u202f]"
PASSPORT_BODY = rf"(?:[0-9]{{2}}{SPACE}?[0-9]{{2}}){SPACE}*(?:№{SPACE}*)?[0-9]{{6}}"
PASSPORT = re.compile(rf"(?<!\w){PASSPORT_BODY}(?!\w)")
BIRTH_BODY = rf"(?P<roman>[IVXLCDM]+)[ \t\-‑–]*[А-ЯЁ]{{2}}{SPACE}*(?:№{SPACE}*)?[0-9]{{6}}"
BIRTH = re.compile(rf"(?<!\w){BIRTH_BODY}(?!\w)", re.I)
ROMAN = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", re.I)
LONG_NUMBER = re.compile(r"(?<!\w)(?:[0-9]{19}|[0-9]{17}|[0-9]{15}|[0-9]{13})(?!\w)")
IPV4 = re.compile(r"(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?!\w|\.[0-9])")
IPV6 = re.compile(r"(?<![\w:])(?=[0-9a-f:]*:)[0-9a-f:]*:[0-9a-f:]+(?:\.[0-9]{1,3}){0,3}(?!\w|:[0-9a-f]|\.[0-9])", re.I)
FIELD_NAMES = {
    **CONTEXT, "PASSPORT": (r"паспорт(?:а|е|ом|у)?(?:[ \t]+(?:гражданина[ \t]+)?РФ)?"
                            r"(?:[ \t]+(?:указан|значится|записан))?(?:[ \t]+(?:серия|номер))?"),
    "OGRNIP": r"огрнип", "KLADR": r"(?:код\s+)?кладр", "IP": r"ip(?:[ -]адрес)?",
    "BIRTH_CERT": r"свидетельств[оа]\s+о\s+рождении",
}
FIELDS = {kind: re.compile(rf"(?<!\w)(?:{expr})\s*[:№=—-]?\s*$", re.I) for kind, expr in FIELD_NAMES.items()}


def valid_ogrnip(value):
    return bool(re.fullmatch(r"3[0-9]{14}", value) and int(value[-1]) == int(value[:-1]) % 13 % 10)


def valid_birth(value):
    match = BIRTH.fullmatch(value)
    return bool(match and ROMAN.fullmatch(match["roman"]))


def merge_course_fields(text, existing):
    candidates = []
    for pattern, label in ((BIRTH, "BIRTH_CERT"), (PASSPORT, "PASSPORT"), (LONG_NUMBER, None),
                           (IPV6, "IP"), (IPV4, "IP")):
        for match in pattern.finditer(text):
            value = match.group()
            kind = label
            if kind is None:
                if valid_ogrnip(value):
                    kind = "OGRNIP"
                elif len(value) in (13, 17, 19) and value in catalog()[0]:
                    kind = "KLADR"
                else:
                    continue
            elif kind == "BIRTH_CERT" and not valid_birth(value):
                continue
            elif kind == "IP":
                try:
                    ipaddress.ip_address(value)
                except ValueError:
                    continue
            prefix = text[max(0, match.start() - 70):match.start()]
            hints = {key for key, field in FIELDS.items() if field.search(prefix)}
            if (hints and kind not in hints) or NON_PERSONAL.search(prefix):
                continue
            if kind == "PASSPORT" and value.isdigit() and kind not in hints:
                continue
            candidates.append(Entity(kind, match.start(), match.end()))
    # A labelled but invalid coursework field cannot become a card or an INN
    # merely because its digits coincidentally pass another checksum.
    accepted = [e for e in existing if e.label == "EMAIL"]
    for e in sorted(candidates, key=lambda item: -(item.end - item.start)):
        if not any(overlaps(e, other) for other in accepted):
            accepted.append(e)
    for e in existing:
        prefix = text[max(0, e.start - 70):e.start]
        if any(FIELDS[k].search(prefix) for k in FIELDS.keys() - CONTEXT.keys()):
            continue
        if not any(overlaps(e, other) for other in accepted):
            accepted.append(e)
    return sorted(accepted, key=lambda e: e.start)

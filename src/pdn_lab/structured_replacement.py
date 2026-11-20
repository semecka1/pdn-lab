"""Canonical values and format-preserving replacements for detected fields."""
from ipaddress import ip_address, ip_network
import re

from email_validator import EmailNotValidError, validate_email
import phonenumbers

from .course_rules import PASSPORT, valid_birth, valid_ogrnip
from .kladr import catalog, level
from .rules import find_emails
from .validators import digits, make_inn, snils_check, valid_card, valid_inn, valid_phone, valid_snils


def _phone_ok(value):
    if not re.fullmatch(r"\+?[0-9(][0-9() .\-–‑\t\r\n\u00a0\u202f]*(?:(?:ext\.?|доб\.?)\s*[0-9]+)?", value, re.I):
        return False
    if valid_phone(value):
        return True
    number = digits(value)
    if not value.startswith("+"):
        return len(number) == 10 and valid_phone("7" + value)
    try:
        return phonenumbers.is_valid_number(phonenumbers.parse(value, None))
    except phonenumbers.NumberParseException:
        return False


def canonical(label, value):
    if label == "EMAIL":
        if not any(e.start == 0 and e.end == len(value) for e in find_emails(value)):
            raise ValueError("Некорректный e-mail")
        local, _, domain = value.rpartition("@")
        return local + "@" + domain.casefold()
    if label == "IP":
        return str(ip_address(value))
    if label == "KLADR":
        if value not in catalog()[0]:
            raise ValueError("Код отсутствует в локальном выпуске КЛАДР")
        return value
    if label == "BIRTH_CERT":
        if not valid_birth(value):
            raise ValueError("Некорректная серия или номер свидетельства")
        return "".join(c.upper() for c in value if c.isalnum())
    validators = {"INN": valid_inn, "SNILS": valid_snils, "CARD": valid_card, "PHONE": _phone_ok,
                  "PASSPORT": PASSPORT.fullmatch, "OGRNIP": valid_ogrnip}
    if label not in validators or not validators[label](value):
        raise ValueError(f"Значение не соответствует типу {label}")
    return digits(value)


def display(label, replacement, original):
    if label == "IP":
        return replacement
    if label == "EMAIL":
        return replacement.rsplit("@", 1)[0] + "@" + original.rsplit("@", 1)[1]
    if label == "BIRTH_CERT":
        return original[:-6] + replacement[-6:]
    stream = iter(replacement)
    return "".join(next(stream) if "0" <= c <= "9" else c for c in original)


def _catalog_choice(code, blocked, rng):
    groups = catalog()[1]
    kind, region = level(code), code[:2]
    local = groups.get((kind, region), ())
    # Almost all usually available groups are large; avoid scanning them.
    for _ in range(32):
        if local:
            candidate = rng.choice(local)
            if candidate not in blocked:
                return candidate
    remaining = [x for x in local if x not in blocked]
    if not remaining:
        remaining = [x for (group, area), codes in groups.items() if group == kind and area != region
                     for x in codes if x not in blocked]
    if not remaining:
        raise ValueError("Исчерпаны допустимые коды КЛАДР этого уровня")
    return rng.choice(remaining)


def generate(label, key, original, blocked, rng):
    def number(length):
        return "".join(rng.choice("0123456789") for _ in range(length))

    if label == "KLADR":
        return _catalog_choice(key, blocked, rng)
    for _ in range(512):
        if label == "INN":
            candidate = make_inn(key[:4] + number(5 if len(key) == 10 else 6))
            if not valid_inn(candidate):
                continue
        elif label == "CARD":
            middle = number(len(key) - 11)
            # The original last digit must stay. Solve for a middle digit.
            prefix, suffix = key[:6] + middle, key[-4:]
            candidate = next((prefix + d + suffix for d in "0123456789" if valid_card(prefix + d + suffix)), None)
        elif label == "SNILS":
            body = number(9)
            candidate = body + f"{snils_check(body):02d}"
            if not valid_snils(candidate):
                continue
        elif label == "OGRNIP":
            body = key[:5] + number(9)
            candidate = body + str(int(body) % 13 % 10)
        elif label == "PASSPORT":
            candidate = key[:4] + number(6)
        elif label == "BIRTH_CERT":
            candidate = key[:-6] + number(6)
        elif label == "EMAIL":
            domain = key.rsplit("@", 1)[1]
            # Respect the original address's total length budget on long domains.
            budget = min(15, 253 - len(domain.encode("idna")))
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
            candidate = "".join(rng.choice(alphabet) for _ in range(max(1, budget))) + "@" + domain
            try:
                validate_email(candidate, check_deliverability=False)
            except EmailNotValidError:
                continue
        elif label == "PHONE":
            candidate = key[:4] + number(len(key) - 4)
            if not _phone_ok(display(label, candidate, original)):
                continue
        elif label == "IP":
            addr = ip_address(key)
            network = ip_network((addr, 24 if addr.version == 4 else 64), strict=False)
            candidate = str(type(addr)(int(network.network_address) + rng.randrange(1, network.num_addresses - 1)))
        else:
            raise ValueError(f"Неизвестный атрибут {label}")
        if candidate and candidate not in blocked:
            return candidate
    raise ValueError(f"Не удалось подобрать отличающееся допустимое значение {label}")

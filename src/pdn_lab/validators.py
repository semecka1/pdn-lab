"""Формат и контрольные разряды. Проверка не доказывает выдачу номера."""

import re

SEPARATOR = r"[ \t\r\n\u00a0\u202f\-–‑]"


def digits(value: str) -> str:
    return re.sub(r"[^0-9]", "", value)


def _weighted(number: str, weights: tuple[int, ...]) -> int:
    return sum(int(digit) * weight for digit, weight in zip(number, weights)) % 11 % 10


def valid_inn(value: str) -> bool:
    if not re.fullmatch(r"[0-9]{10}(?:[0-9]{2})?", value):
        return False
    number = digits(value)
    if len(number) not in (10, 12) or len(set(number)) == 1:
        return False
    if len(number) == 10:
        return int(number[-1]) == _weighted(number, (2, 4, 10, 3, 5, 9, 4, 6, 8))
    return (
        int(number[10]) == _weighted(number, (7, 2, 4, 10, 3, 5, 9, 4, 6, 8))
        and int(number[11]) == _weighted(number, (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8))
    )


def snils_check(number: str) -> int:
    if len(number) != 9 or not number.isascii() or not number.isdigit():
        raise ValueError("Основа СНИЛС должна содержать 9 цифр")
    total = sum(int(digit) * weight for digit, weight in zip(number, range(9, 0, -1)))
    return (total % 101) % 100


def valid_snils(value: str) -> bool:
    if not re.fullmatch(rf"[0-9]{{3}}{SEPARATOR}*[0-9]{{3}}{SEPARATOR}*[0-9]{{3}}{SEPARATOR}*[0-9]{{2}}", value):
        return False
    number = digits(value)
    # Для исторических основ <= 001001998 стандартная проверка не применялась.
    # Они не принимаются как подтверждённые контрольным числом.
    return (
        len(number) == 11
        and int(number[:9]) > 1001998
        and len(set(number)) > 1
        and int(number[-2:]) == snils_check(number[:9])
    )


def valid_card(value: str) -> bool:
    if not re.fullmatch(rf"[0-9]+(?:{SEPARATOR}+[0-9]+)*", value):
        return False
    number = digits(value)
    if not 13 <= len(number) <= 19 or len(set(number)) == 1:
        return False
    total = 0
    for index, digit in enumerate(reversed(number)):
        term = int(digit)
        if index % 2:
            term *= 2
            if term > 9:
                term -= 9
        total += term
    return total % 10 == 0


def valid_phone(value: str) -> bool:
    if not re.fullmatch(rf"(?:\+7|[78]){SEPARATOR}*(?:\([0-9]{{3}}\)|[0-9]{{3}}){SEPARATOR}*[0-9]{{3}}{SEPARATOR}*[0-9]{{2}}{SEPARATOR}*[0-9]{{2}}", value):
        return False
    number = digits(value)
    return len(number) == 11 and number[0] in "78" and number[1] in "345689" and len(set(number[1:])) > 1


def make_inn(base: str) -> str:
    if len(base) == 9:
        return base + str(_weighted(base, (2, 4, 10, 3, 5, 9, 4, 6, 8)))
    if len(base) != 10:
        raise ValueError("Основа ИНН должна содержать 9 или 10 цифр")
    number = base + str(_weighted(base, (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)))
    return number + str(_weighted(number, (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)))


def make_card(base: str) -> str:
    for digit in "0123456789":
        if valid_card(base + digit):
            return base + digit
    raise ValueError("Некорректная основа карты")

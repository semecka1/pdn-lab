"""Offline lookup against a fixed edition of the address classifier."""
from collections import defaultdict
from functools import lru_cache
import gzip
from importlib.resources import files


def level(code):
    if len(code) != 13:
        return "street" if len(code) == 17 else "building"
    for name, segment in (("settlement", code[8:11]), ("city", code[5:8]), ("district", code[2:5])):
        if segment != "000":
            return name
    return "region"


@lru_cache(maxsize=1)
def catalog():
    path = files("pdn_lab").joinpath("resources/kladr.txt.gz")
    with path.open("rb") as source, gzip.open(source, "rt", encoding="ascii") as stream:
        known = frozenset(line.strip() for line in stream)
    groups = defaultdict(list)
    for code in sorted(known):
        # The final two digits indicate currency for localities and streets.
        if len(code) == 19 or code.endswith("00"):
            groups[level(code), code[:2]].append(code)
    return known, groups

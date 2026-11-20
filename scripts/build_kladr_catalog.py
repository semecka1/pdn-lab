"""Extract only address codes from KLADR.DBF, STREET.DBF and DOMA.DBF."""
import argparse
import gzip
from pathlib import Path
import struct


def read_codes(directory, name):
    path = directory / (name + ".DBF")
    raw = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(".DBF.gz").read_bytes())
    count, header, width = struct.unpack_from("<IHH", raw, 4)
    offset = 1
    for pos in range(32, header - 1, 32):
        size = raw[pos + 16]
        if raw[pos:pos + 11].rstrip(b"\0") == b"CODE":
            break
        offset += size
    else:
        raise ValueError(f"Отсутствует поле CODE в {name}")
    for row in range(header, header + width * count, width):
        code = raw[row + offset:row + offset + size].strip()
        if raw[row:row + 1] == b" " and code.isdigit() and len(code) == size:
            yield code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    codes = {code for name in ("KLADR", "STREET", "DOMA") for code in read_codes(args.directory, name)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(b"\n".join(sorted(codes)) + b"\n", mtime=0))
    print(f"{len(codes)} codes")

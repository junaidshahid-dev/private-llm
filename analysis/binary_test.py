"""binary_test.py — the pure binary identifier parses ELF/PE headers and degrades gracefully.

    python analysis/binary_test.py
"""
from __future__ import annotations

import os
import struct
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from analysis.binary import identify                                        # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 70)
    print("BINARY IDENTIFY — ELF / PE header parsing (pure, no external tools)")
    print("=" * 70)

    # crafted ELF64, little-endian, EXEC, x86-64
    elf = bytearray(20)
    elf[0:4] = b"\x7fELF"
    elf[4], elf[5], elf[6] = 2, 1, 1                 # class=64, data=LE, version=1
    elf[16:18] = struct.pack("<H", 2)                # e_type = EXEC
    elf[18:20] = struct.pack("<H", 0x3e)             # e_machine = x86-64
    i = identify(bytes(elf))
    check("ELF format detected", i["format"] == "ELF", str(i))
    check("ELF class 64-bit", i["class"] == "64-bit")
    check("ELF type EXEC", "EXEC" in i["type"])
    check("ELF machine x86-64", i["machine"] == "x86-64")

    # crafted PE, x86-64, 3 sections, DLL+executable
    pe = bytearray(0x58)
    pe[0:2] = b"MZ"
    pe[0x3c:0x40] = struct.pack("<I", 0x40)
    pe[0x40:0x44] = b"PE\x00\x00"
    pe[0x44:0x46] = struct.pack("<H", 0x8664)
    pe[0x46:0x48] = struct.pack("<H", 3)
    pe[0x56:0x58] = struct.pack("<H", 0x2002)        # DLL | EXECUTABLE_IMAGE
    p = identify(bytes(pe))
    check("PE format detected", p["format"] == "PE", str(p))
    check("PE machine x86-64", p["machine"] == "x86-64")
    check("PE section count", p["sections"] == 3)
    check("PE DLL flag parsed", p["is_dll"] is True)

    # real file cross-check: the running interpreter (PE on Windows, ELF on Linux)
    with open(sys.executable, "rb") as f:
        real = identify(f.read(8192))
    check("the running interpreter is identified as a real binary format",
          real["format"] in ("PE", "ELF", "Mach-O"), str(real))

    check("unknown bytes -> unknown + magic", identify(b"just some text")["format"] == "unknown")
    check("gzip magic detected", identify(b"\x1f\x8b\x08\x00")["format"] == "gzip")
    check("empty input does not crash", identify(b"")["format"] == "unknown")

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL BINARY-IDENTIFY TESTS PASS — real header parsing, graceful on unknown/empty input.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

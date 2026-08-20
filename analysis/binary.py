"""binary.py — read-only binary identification + header parsing, pure Python (no external tools).

For RE / malware triage: identify a file's real format by magic bytes and parse the key header fields
(ELF class/type/machine, PE machine/characteristics/sections, Mach-O). This always works — no radare2
/gdb/objdump needed — so it is the reliable first step; the heavier tools (objdump/readelf/nm) are
separate, gated, and best-effort (they degrade gracefully when the binary is absent). Everything here
is OBSERVED structure, never a behavioural claim.
"""
from __future__ import annotations

import struct

_ELF_TYPES = {1: "REL (relocatable)", 2: "EXEC (executable)", 3: "DYN (shared object / PIE)",
              4: "CORE"}
_ELF_MACHINES = {0x03: "x86", 0x3e: "x86-64", 0x28: "ARM", 0xb7: "AArch64", 0xf3: "RISC-V",
                 0x08: "MIPS", 0x14: "PowerPC", 0x15: "PowerPC64"}
_PE_MACHINES = {0x14c: "x86", 0x8664: "x86-64", 0x1c0: "ARM", 0xaa64: "ARM64", 0x200: "IA64"}


def _elf(data: bytes) -> dict:
    if len(data) < 20:
        return {"error": "truncated ELF header"}
    ei_class, ei_data = data[4], data[5]
    endian = "<" if ei_data == 1 else ">"
    e_type = struct.unpack(endian + "H", data[16:18])[0]
    e_machine = struct.unpack(endian + "H", data[18:20])[0]
    return {"class": "64-bit" if ei_class == 2 else "32-bit",
            "endian": "little" if ei_data == 1 else "big",
            "type": _ELF_TYPES.get(e_type, str(e_type)),
            "machine": _ELF_MACHINES.get(e_machine, hex(e_machine)),
            "stripped": None}


def _pe(data: bytes) -> dict:
    if len(data) < 0x40:
        return {"error": "truncated PE"}
    pe_off = struct.unpack("<I", data[0x3c:0x40])[0]
    if len(data) < pe_off + 24 or data[pe_off:pe_off + 4] != b"PE\x00\x00":
        return {"error": "no PE header"}
    machine = struct.unpack("<H", data[pe_off + 4:pe_off + 6])[0]
    nsec = struct.unpack("<H", data[pe_off + 6:pe_off + 8])[0]
    chars = struct.unpack("<H", data[pe_off + 22:pe_off + 24])[0]
    return {"machine": _PE_MACHINES.get(machine, hex(machine)),
            "is_dll": bool(chars & 0x2000), "is_executable": bool(chars & 0x0002),
            "sections": nsec}


def identify(data: bytes) -> dict:
    """Identify a file's format by magic bytes and parse its header (best-effort, read-only)."""
    data = data or b""
    if data[:4] == b"\x7fELF":
        return {"format": "ELF", **_elf(data)}
    if data[:2] == b"MZ":
        return {"format": "PE", **_pe(data)}
    if data[:4] in (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe"):
        return {"format": "Mach-O", "bits": "64-bit" if data[3] in (0xcf, 0xfe) else "32-bit"}
    if data[:4] == b"\xca\xfe\xba\xbe":
        return {"format": "Mach-O universal (fat) / Java class"}
    if data[:2] == b"PK":
        return {"format": "ZIP/JAR/APK/OOXML"}
    if data[:4] == b"%PDF":
        return {"format": "PDF"}
    if data[:2] == b"\x1f\x8b":
        return {"format": "gzip"}
    return {"format": "unknown", "magic": data[:8].hex()}


def render(info: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in info.items() if v is not None)

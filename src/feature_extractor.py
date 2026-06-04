"""
feature_extractor.py
====================
Core PE feature extraction engine for MalScan-ML.

Extracts 72 features from Windows PE (Portable Executable) files:
- Header features (DOS, NT, Optional headers)
- Section features (entropy, size ratios, names)
- Import features (DLL names, API calls, suspicious imports)
- String features (URLs, IPs, registry keys, paths)
- File metadata

Usage:
    # Extract features from a single file
    extractor = PEFeatureExtractor()
    features = extractor.extract("malware.exe")

    # Batch extract a directory
    python feature_extractor.py --dir /path/to/samples/ --output features.csv
"""

import os
import re
import math
import hashlib
import struct
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pefile
import click
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Suspicious API calls commonly used by malware
# ─────────────────────────────────────────────
SUSPICIOUS_IMPORTS = {
    # Process injection
    "VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread",
    "NtUnmapViewOfSection", "SetThreadContext",
    # Credential theft
    "CryptAcquireContext", "CryptDeriveKey", "CryptEncrypt",
    "LsaRetrievePrivateData", "SamQueryInformationUser",
    # Anti-analysis / evasion
    "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
    "GetTickCount", "QueryPerformanceCounter", "NtQueryInformationProcess",
    # Network
    "WSAStartup", "connect", "InternetOpenUrl", "URLDownloadToFile",
    "HttpSendRequest", "WinHttpOpen",
    # File/Registry persistence
    "RegSetValueEx", "RegCreateKeyEx", "CreateService",
    "SetFileAttributes", "MoveFileEx",
    # Code injection helpers
    "LoadLibrary", "GetProcAddress", "VirtualProtect",
    # Shell execution
    "ShellExecute", "WinExec", "CreateProcess",
}

# Known packers / protectors section names
PACKER_SECTION_NAMES = {
    "UPX0", "UPX1", "UPX2", ".aspack", ".adata",
    ".nsp0", ".nsp1", ".nsp2", "execryptor",
    ".petite", "pec2", ".ccg", ".svkp",
}


def calculate_entropy(data: bytes) -> float:
    """Shannon entropy of a byte sequence. High entropy → possible encryption/packing."""
    if not data:
        return 0.0
    counter = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = counter[counter > 0] / len(data)
    return -np.sum(probabilities * np.log2(probabilities))


def extract_strings(data: bytes, min_len: int = 4) -> list[str]:
    """Extract printable ASCII strings from raw bytes."""
    pattern = rb"[^\x00-\x1f\x7f-\xff]{" + str(min_len).encode() + rb",}"
    return [s.decode("ascii", errors="ignore") for s in re.findall(pattern, data)]


class PEFeatureExtractor:
    """
    Extracts a flat feature vector from a PE file.
    Returns a dict of 72 named features suitable for ML.
    """

    FEATURE_NAMES = None  # populated after first extraction

    def extract(self, filepath: str) -> Optional[dict]:
        """
        Main entry point. Returns feature dict or None on parse failure.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        try:
            raw = filepath.read_bytes()
            pe = pefile.PE(data=raw, fast_load=False)
            pe.parse_data_directories()
        except pefile.PEFormatError as e:
            logger.warning(f"PE parse error [{filepath.name}]: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error [{filepath.name}]: {e}")
            return None

        features = {}
        features.update(self._header_features(pe, raw))
        features.update(self._section_features(pe))
        features.update(self._import_features(pe))
        features.update(self._string_features(raw))
        features.update(self._metadata_features(filepath, raw, pe))

        pe.close()
        return features

    # ───────────────────────────────────────────────
    # 1. HEADER FEATURES (15 features)
    # ───────────────────────────────────────────────
    def _header_features(self, pe: pefile.PE, raw: bytes) -> dict:
        f = {}

        # DOS header
        f["e_magic"] = pe.DOS_HEADER.e_magic           # Should be 0x5A4D (MZ)
        f["e_lfanew"] = pe.DOS_HEADER.e_lfanew         # Offset to PE header

        # File header
        fh = pe.FILE_HEADER
        f["machine_type"] = fh.Machine                  # Target arch (0x14c = x86)
        f["num_sections"] = fh.NumberOfSections
        f["compile_timestamp"] = fh.TimeDateStamp       # Raw timestamp
        f["num_symbols"] = fh.NumberOfSymbols
        f["characteristics"] = fh.Characteristics       # Bitmask

        # Optional header
        oh = pe.OPTIONAL_HEADER
        f["magic"] = oh.Magic                            # 0x10b = PE32, 0x20b = PE32+
        f["major_linker_version"] = oh.MajorLinkerVersion
        f["minor_linker_version"] = oh.MinorLinkerVersion
        f["size_of_code"] = oh.SizeOfCode
        f["size_of_initialized_data"] = oh.SizeOfInitializedData
        f["size_of_uninitialized_data"] = oh.SizeOfUninitializedData
        f["address_of_entry_point"] = oh.AddressOfEntryPoint
        f["image_base"] = oh.ImageBase
        f["section_alignment"] = oh.SectionAlignment
        f["file_alignment"] = oh.FileAlignment
        f["size_of_image"] = oh.SizeOfImage
        f["size_of_headers"] = oh.SizeOfHeaders
        f["checksum"] = oh.CheckSum
        f["subsystem"] = oh.Subsystem                   # GUI=2, Console=3
        f["dll_characteristics"] = oh.DllCharacteristics
        f["num_rva_and_sizes"] = oh.NumberOfRvaAndSizes

        # Entry point entropy (200 bytes around EP)
        try:
            ep_offset = pe.get_offset_from_rva(oh.AddressOfEntryPoint)
            ep_bytes = raw[ep_offset: ep_offset + 200]
            f["entry_point_entropy"] = calculate_entropy(ep_bytes)
        except Exception:
            f["entry_point_entropy"] = 0.0

        return f

    # ───────────────────────────────────────────────
    # 2. SECTION FEATURES (20 features)
    # ───────────────────────────────────────────────
    def _section_features(self, pe: pefile.PE) -> dict:
        f = {}
        sections = pe.sections

        entropies = []
        virt_sizes = []
        raw_sizes = []
        suspicious_names = 0
        exec_sections = 0

        for section in sections:
            data = section.get_data()
            ent = calculate_entropy(data)
            entropies.append(ent)

            virt = section.Misc_VirtualSize
            raw = section.SizeOfRawData
            virt_sizes.append(virt)
            raw_sizes.append(raw)

            # Packer heuristic: section name matches known packer names
            name = section.Name.decode("utf-8", errors="ignore").rstrip("\x00")
            if name.upper() in {n.upper() for n in PACKER_SECTION_NAMES}:
                suspicious_names += 1

            # Executable section
            if section.Characteristics & 0x20000000:
                exec_sections += 1

        f["num_sections"] = len(sections)
        f["mean_entropy"] = np.mean(entropies) if entropies else 0.0
        f["max_entropy"] = np.max(entropies) if entropies else 0.0
        f["min_entropy"] = np.min(entropies) if entropies else 0.0
        f["std_entropy"] = np.std(entropies) if entropies else 0.0
        f["high_entropy_sections"] = sum(1 for e in entropies if e > 7.0)  # >7 → likely packed
        f["mean_virtual_size"] = np.mean(virt_sizes) if virt_sizes else 0.0
        f["mean_raw_size"] = np.mean(raw_sizes) if raw_sizes else 0.0

        # Virtual vs raw ratio — large ratio = padding / hollowing indicator
        ratios = []
        for v, r in zip(virt_sizes, raw_sizes):
            if r > 0:
                ratios.append(v / r)
        f["mean_virt_raw_ratio"] = np.mean(ratios) if ratios else 0.0
        f["max_virt_raw_ratio"] = np.max(ratios) if ratios else 0.0
        f["suspicious_section_names"] = suspicious_names
        f["executable_sections"] = exec_sections

        # Is the entry point in a non-standard section?
        ep_rva = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        f["ep_in_unusual_section"] = int(self._ep_in_unusual_section(pe, ep_rva, sections))

        # Total size of all sections vs image size
        total_raw = sum(raw_sizes)
        image_size = pe.OPTIONAL_HEADER.SizeOfImage
        f["section_size_ratio"] = total_raw / image_size if image_size > 0 else 0.0

        # Writable + executable sections (W^X violation)
        wx_sections = 0
        for section in sections:
            if (section.Characteristics & 0x20000000) and (section.Characteristics & 0x80000000):
                wx_sections += 1
        f["wx_sections"] = wx_sections

        return f

    def _ep_in_unusual_section(self, pe, ep_rva, sections) -> bool:
        """Check if EP is in a section other than .text"""
        for section in sections:
            name = section.Name.decode("utf-8", errors="ignore").rstrip("\x00").lower()
            start = section.VirtualAddress
            end = start + max(section.Misc_VirtualSize, section.SizeOfRawData)
            if start <= ep_rva < end:
                return name not in (".text", "code", ".code")
        return False

    # ───────────────────────────────────────────────
    # 3. IMPORT FEATURES (20 features)
    # ───────────────────────────────────────────────
    def _import_features(self, pe: pefile.PE) -> dict:
        f = {}

        dlls = []
        all_imports = []

        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode("utf-8", errors="ignore").lower()
                dlls.append(dll_name)
                for imp in entry.imports:
                    if imp.name:
                        all_imports.append(imp.name.decode("utf-8", errors="ignore"))

        f["num_imported_dlls"] = len(dlls)
        f["num_imports"] = len(all_imports)

        # Count suspicious API calls
        suspicious_count = sum(1 for imp in all_imports if imp in SUSPICIOUS_IMPORTS)
        f["num_suspicious_imports"] = suspicious_count
        f["suspicious_import_ratio"] = suspicious_count / max(len(all_imports), 1)

        # Specific DLL presence flags (important signals)
        f["imports_kernel32"] = int("kernel32.dll" in dlls)
        f["imports_ntdll"] = int("ntdll.dll" in dlls)
        f["imports_ws2_32"] = int("ws2_32.dll" in dlls)       # Network
        f["imports_wininet"] = int("wininet.dll" in dlls)      # HTTP
        f["imports_advapi32"] = int("advapi32.dll" in dlls)    # Registry / crypto
        f["imports_crypt32"] = int("crypt32.dll" in dlls)      # Crypto
        f["imports_shell32"] = int("shell32.dll" in dlls)

        # Specific dangerous functions
        f["has_virtualalloc"] = int("VirtualAlloc" in all_imports or "VirtualAllocEx" in all_imports)
        f["has_createremotethread"] = int("CreateRemoteThread" in all_imports)
        f["has_writeprocessmemory"] = int("WriteProcessMemory" in all_imports)
        f["has_loadlibrary"] = int("LoadLibrary" in all_imports or "LoadLibraryA" in all_imports)
        f["has_getprocaddress"] = int("GetProcAddress" in all_imports)
        f["has_isdebuggerpresent"] = int("IsDebuggerPresent" in all_imports)
        f["has_regsetvalueex"] = int("RegSetValueEx" in all_imports or "RegSetValueExA" in all_imports)
        f["has_urldownloadtofile"] = int("URLDownloadToFile" in all_imports or "URLDownloadToFileA" in all_imports)
        f["has_shellexecute"] = int("ShellExecute" in all_imports or "ShellExecuteA" in all_imports)

        # Export table
        f["has_exports"] = int(hasattr(pe, "DIRECTORY_ENTRY_EXPORT"))
        if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            f["num_exports"] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
        else:
            f["num_exports"] = 0

        return f

    # ───────────────────────────────────────────────
    # 4. STRING FEATURES (10 features)
    # ───────────────────────────────────────────────
    def _string_features(self, raw: bytes) -> dict:
        f = {}
        strings = extract_strings(raw)
        all_strings = " ".join(strings)

        # URL patterns
        url_pattern = re.compile(r"https?://[^\s]{4,}", re.IGNORECASE)
        f["num_urls"] = len(url_pattern.findall(all_strings))

        # IP address patterns
        ip_pattern = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
        f["num_ip_addresses"] = len(ip_pattern.findall(all_strings))

        # Registry key patterns
        reg_pattern = re.compile(r"(HKEY_|HKLM|HKCU|SOFTWARE\\)", re.IGNORECASE)
        f["num_registry_refs"] = len(reg_pattern.findall(all_strings))

        # File path patterns
        path_pattern = re.compile(r"[a-zA-Z]:\\[^\s]{3,}", re.IGNORECASE)
        f["num_file_paths"] = len(path_pattern.findall(all_strings))

        # Total printable strings count and average length
        f["num_strings"] = len(strings)
        f["mean_string_length"] = np.mean([len(s) for s in strings]) if strings else 0.0

        # Entropy of all strings concatenated
        f["strings_entropy"] = calculate_entropy(all_strings.encode("utf-8", errors="ignore"))

        # Common malware string indicators
        mal_strings = ["cmd.exe", "powershell", "base64", "eval(", "exec(", ".bat", "schtasks"]
        f["num_malware_strings"] = sum(1 for m in mal_strings if m.lower() in all_strings.lower())

        # Encoded/obfuscated string heuristic: high ratio of very long strings
        long_strings = [s for s in strings if len(s) > 100]
        f["long_string_ratio"] = len(long_strings) / max(len(strings), 1)

        # Base64 pattern detection
        b64_pattern = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
        f["num_b64_strings"] = len(b64_pattern.findall(all_strings))

        return f

    # ───────────────────────────────────────────────
    # 5. METADATA FEATURES (7 features)
    # ───────────────────────────────────────────────
    def _metadata_features(self, filepath: Path, raw: bytes, pe: pefile.PE) -> dict:
        f = {}

        f["file_size"] = len(raw)
        f["file_entropy"] = calculate_entropy(raw)

        # Resource directory presence
        f["has_resources"] = int(hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"))

        # Debug directory (stripped in malware to hide info)
        f["has_debug_info"] = int(
            hasattr(pe, "DIRECTORY_ENTRY_DEBUG") and len(pe.DIRECTORY_ENTRY_DEBUG) > 0
        )

        # Relocation table (often absent in malware)
        f["has_reloc"] = int(hasattr(pe, "DIRECTORY_ENTRY_BASERELOC"))

        # TLS callbacks (used for anti-debug / code before EP)
        f["has_tls"] = int(hasattr(pe, "DIRECTORY_ENTRY_TLS"))

        # Packing heuristic: whole-file entropy > 7 = likely packed
        f["is_packed_heuristic"] = int(f["file_entropy"] > 7.0)

        return f


# ─────────────────────────────────────────────
# CLI for batch feature extraction
# ─────────────────────────────────────────────
@click.command()
@click.option("--dir", "-d", "directory", required=True, help="Directory with PE files")
@click.option("--output", "-o", default="features.csv", help="Output CSV path")
@click.option("--label", "-l", default=None, type=int, help="Label (1=malware, 0=benign)")
@click.option("--recursive", "-r", is_flag=True, default=False, help="Recurse into subdirectories")
def main(directory, output, label, recursive):
    """
    Batch-extract PE features from a directory of files.

    \b
    Example:
        python feature_extractor.py -d ./malware_samples/ -l 1 -o malware_features.csv
        python feature_extractor.py -d ./benign_samples/ -l 0 -o benign_features.csv
    """
    extractor = PEFeatureExtractor()
    directory = Path(directory)

    if recursive:
        files = list(directory.rglob("*"))
    else:
        files = list(directory.iterdir())

    files = [f for f in files if f.is_file()]
    logger.info(f"Found {len(files)} files in {directory}")

    records = []
    failed = 0

    for filepath in tqdm(files, desc="Extracting features"):
        features = extractor.extract(str(filepath))
        if features is not None:
            features["filename"] = filepath.name
            if label is not None:
                features["label"] = label
            records.append(features)
        else:
            failed += 1

    if not records:
        logger.error("No features extracted. Check input directory.")
        return

    df = pd.DataFrame(records)
    df.to_csv(output, index=False)
    logger.info(f"Saved {len(records)} records to {output} ({failed} failed)")
    logger.info(f"Feature dimensions: {df.shape[1]} columns")


if __name__ == "__main__":
    main()

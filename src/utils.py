"""
utils.py
========
Helper utilities for MalScan-ML.

- load_features(): Load and validate a feature CSV
- get_feature_columns(): Auto-detect feature columns
- create_sample_dataset(): Generate synthetic dataset for demonstration
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are NOT features
NON_FEATURE_COLS = {"filename", "label", "sha256", "md5", "sha1", "filepath"}


def load_features(csv_path: str) -> pd.DataFrame:
    """
    Load feature CSV, drop non-numeric columns, handle NaNs.

    Returns:
        Cleaned DataFrame with 'label' column present.
    """
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("Dataset must have a 'label' column (0=benign, 1=malware).")

    feature_cols = get_feature_columns(df)
    logger.info(f"Loaded {len(df)} samples, {len(feature_cols)} features")

    # Fill NaN with median per column
    for col in feature_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes label and metadata)."""
    return [
        col for col in df.columns
        if col not in NON_FEATURE_COLS
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]


def create_sample_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset mimicking PE feature distributions.

    This is for demonstration only. Use EMBER or VirusShare for real training.

    Malware patterns simulated:
    - Higher entropy (packed/encrypted)
    - More suspicious imports
    - Fewer debug symbols
    - Unusual section characteristics
    """
    rng = np.random.default_rng(random_state)
    n_malware = n_samples // 2
    n_benign = n_samples - n_malware

    records = []

    # ── Benign samples ──────────────────────────────
    for _ in range(n_benign):
        r = {
            # Header
            "e_magic": 23117,
            "e_lfanew": rng.integers(64, 256),
            "machine_type": 332,                    # x86
            "num_sections": rng.integers(3, 8),
            "compile_timestamp": rng.integers(1_000_000_000, 1_700_000_000),
            "num_symbols": rng.integers(0, 1000),
            "characteristics": rng.integers(256, 8192),
            "magic": 267,
            "major_linker_version": rng.integers(6, 14),
            "minor_linker_version": rng.integers(0, 30),
            "size_of_code": rng.integers(4096, 2_000_000),
            "size_of_initialized_data": rng.integers(1000, 500_000),
            "size_of_uninitialized_data": rng.integers(0, 4096),
            "address_of_entry_point": rng.integers(4096, 65536),
            "image_base": 4194304,
            "section_alignment": 4096,
            "file_alignment": 512,
            "size_of_image": rng.integers(65536, 10_000_000),
            "size_of_headers": 1024,
            "checksum": rng.integers(0, 100_000),
            "subsystem": 2,
            "dll_characteristics": rng.integers(0, 32768),
            "num_rva_and_sizes": 16,
            "entry_point_entropy": rng.uniform(3.0, 6.0),  # Low-moderate entropy
            # Sections
            "mean_entropy": rng.uniform(3.5, 6.0),
            "max_entropy": rng.uniform(5.0, 7.0),
            "min_entropy": rng.uniform(0.5, 3.0),
            "std_entropy": rng.uniform(0.5, 1.5),
            "high_entropy_sections": rng.integers(0, 1),
            "mean_virtual_size": rng.integers(4096, 500_000),
            "mean_raw_size": rng.integers(4096, 500_000),
            "mean_virt_raw_ratio": rng.uniform(0.8, 1.5),
            "max_virt_raw_ratio": rng.uniform(1.0, 2.0),
            "suspicious_section_names": 0,
            "executable_sections": rng.integers(1, 3),
            "ep_in_unusual_section": 0,
            "section_size_ratio": rng.uniform(0.6, 0.95),
            "wx_sections": 0,
            # Imports
            "num_imported_dlls": rng.integers(2, 10),
            "num_imports": rng.integers(10, 100),
            "num_suspicious_imports": rng.integers(0, 3),
            "suspicious_import_ratio": rng.uniform(0.0, 0.05),
            "imports_kernel32": 1,
            "imports_ntdll": rng.integers(0, 2),
            "imports_ws2_32": 0,
            "imports_wininet": 0,
            "imports_advapi32": rng.integers(0, 2),
            "imports_crypt32": 0,
            "imports_shell32": rng.integers(0, 2),
            "has_virtualalloc": rng.integers(0, 2),
            "has_createremotethread": 0,
            "has_writeprocessmemory": 0,
            "has_loadlibrary": 1,
            "has_getprocaddress": rng.integers(0, 2),
            "has_isdebuggerpresent": 0,
            "has_regsetvalueex": rng.integers(0, 2),
            "has_urldownloadtofile": 0,
            "has_shellexecute": rng.integers(0, 2),
            "has_exports": rng.integers(0, 2),
            "num_exports": rng.integers(0, 20),
            # Strings
            "num_urls": rng.integers(0, 5),
            "num_ip_addresses": rng.integers(0, 2),
            "num_registry_refs": rng.integers(0, 10),
            "num_file_paths": rng.integers(1, 20),
            "num_strings": rng.integers(50, 500),
            "mean_string_length": rng.uniform(8.0, 20.0),
            "strings_entropy": rng.uniform(3.0, 5.5),
            "num_malware_strings": rng.integers(0, 2),
            "long_string_ratio": rng.uniform(0.0, 0.05),
            "num_b64_strings": rng.integers(0, 3),
            # Metadata
            "file_size": rng.integers(50_000, 5_000_000),
            "file_entropy": rng.uniform(3.5, 6.5),
            "has_resources": rng.integers(0, 2),
            "has_debug_info": rng.integers(0, 2),    # Benign often HAS debug
            "has_reloc": rng.integers(0, 2),
            "has_tls": rng.integers(0, 2),
            "is_packed_heuristic": 0,
            "label": 0,
        }
        records.append(r)

    # ── Malware samples ─────────────────────────────
    for _ in range(n_malware):
        r = {
            # Header
            "e_magic": 23117,
            "e_lfanew": rng.integers(64, 256),
            "machine_type": 332,
            "num_sections": rng.integers(2, 5),       # Often fewer
            "compile_timestamp": rng.integers(0, 1_000_000_000),  # Often 0 or fake
            "num_symbols": 0,
            "characteristics": rng.integers(256, 8192),
            "magic": 267,
            "major_linker_version": rng.integers(0, 14),
            "minor_linker_version": rng.integers(0, 30),
            "size_of_code": rng.integers(4096, 500_000),
            "size_of_initialized_data": rng.integers(1000, 100_000),
            "size_of_uninitialized_data": rng.integers(0, 4096),
            "address_of_entry_point": rng.integers(4096, 65536),
            "image_base": 4194304,
            "section_alignment": 4096,
            "file_alignment": 512,
            "size_of_image": rng.integers(65536, 5_000_000),
            "size_of_headers": rng.integers(512, 2048),
            "checksum": 0,                              # Often 0 in malware
            "subsystem": rng.choice([2, 3]),
            "dll_characteristics": rng.integers(0, 32768),
            "num_rva_and_sizes": 16,
            "entry_point_entropy": rng.uniform(6.0, 8.0),  # HIGH entropy at EP
            # Sections — high entropy = packed
            "mean_entropy": rng.uniform(6.0, 8.0),
            "max_entropy": rng.uniform(7.0, 8.0),
            "min_entropy": rng.uniform(3.0, 7.0),
            "std_entropy": rng.uniform(0.3, 2.0),
            "high_entropy_sections": rng.integers(1, 4),
            "mean_virtual_size": rng.integers(4096, 1_000_000),
            "mean_raw_size": rng.integers(4096, 200_000),
            "mean_virt_raw_ratio": rng.uniform(1.5, 10.0),   # Large virtual vs raw
            "max_virt_raw_ratio": rng.uniform(2.0, 20.0),
            "suspicious_section_names": rng.integers(0, 3),
            "executable_sections": rng.integers(1, 4),
            "ep_in_unusual_section": rng.integers(0, 2),      # Often unusual
            "section_size_ratio": rng.uniform(0.3, 0.9),
            "wx_sections": rng.integers(0, 3),
            # Imports — more suspicious APIs
            "num_imported_dlls": rng.integers(1, 6),
            "num_imports": rng.integers(5, 50),
            "num_suspicious_imports": rng.integers(3, 15),
            "suspicious_import_ratio": rng.uniform(0.1, 0.8),
            "imports_kernel32": 1,
            "imports_ntdll": rng.integers(0, 2),
            "imports_ws2_32": rng.integers(0, 2),      # Network access
            "imports_wininet": rng.integers(0, 2),
            "imports_advapi32": rng.integers(0, 2),
            "imports_crypt32": rng.integers(0, 2),
            "imports_shell32": rng.integers(0, 2),
            "has_virtualalloc": rng.integers(0, 2),
            "has_createremotethread": rng.integers(0, 2),  # Injection
            "has_writeprocessmemory": rng.integers(0, 2),  # Injection
            "has_loadlibrary": rng.integers(0, 2),
            "has_getprocaddress": rng.integers(0, 2),
            "has_isdebuggerpresent": rng.integers(0, 2),   # Anti-debug
            "has_regsetvalueex": rng.integers(0, 2),
            "has_urldownloadtofile": rng.integers(0, 2),   # Download
            "has_shellexecute": rng.integers(0, 2),
            "has_exports": 0,
            "num_exports": 0,
            # Strings
            "num_urls": rng.integers(0, 15),
            "num_ip_addresses": rng.integers(0, 8),
            "num_registry_refs": rng.integers(0, 20),
            "num_file_paths": rng.integers(0, 10),
            "num_strings": rng.integers(10, 200),
            "mean_string_length": rng.uniform(15.0, 60.0),  # Often longer (encoded)
            "strings_entropy": rng.uniform(4.5, 7.0),
            "num_malware_strings": rng.integers(1, 8),
            "long_string_ratio": rng.uniform(0.05, 0.4),   # High ratio
            "num_b64_strings": rng.integers(1, 20),
            # Metadata
            "file_size": rng.integers(4096, 2_000_000),
            "file_entropy": rng.uniform(6.0, 8.0),        # HIGH = packed
            "has_resources": rng.integers(0, 2),
            "has_debug_info": 0,                            # Stripped in malware
            "has_reloc": rng.integers(0, 2),
            "has_tls": rng.integers(0, 2),
            "is_packed_heuristic": rng.integers(0, 2),
            "label": 1,
        }
        records.append(r)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

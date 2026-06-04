"""
tests/test_feature_extractor.py
================================
Unit tests for the MalScan-ML feature extraction pipeline.
Run with: pytest tests/ -v
"""

import sys
import os
import struct
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_extractor import calculate_entropy, extract_strings, PEFeatureExtractor
from utils import create_sample_dataset, get_feature_columns, load_features


# ─────────────────────────────────────────────────────────────
# Tests: calculate_entropy
# ─────────────────────────────────────────────────────────────

class TestCalculateEntropy:

    def test_empty_bytes_returns_zero(self):
        assert calculate_entropy(b"") == 0.0

    def test_uniform_bytes_max_entropy(self):
        """256 distinct bytes equally distributed = max entropy (8.0)."""
        data = bytes(range(256)) * 100
        entropy = calculate_entropy(data)
        assert abs(entropy - 8.0) < 0.01

    def test_single_byte_min_entropy(self):
        """All same bytes = entropy 0."""
        data = b"\x00" * 1000
        entropy = calculate_entropy(data)
        assert entropy == 0.0

    def test_high_entropy_indicates_packing(self):
        """Random data should have high entropy (simulates packed binary)."""
        rng = np.random.default_rng(42)
        random_bytes = bytes(rng.integers(0, 256, size=10000, dtype=np.uint8))
        entropy = calculate_entropy(random_bytes)
        assert entropy > 7.5  # Random data ≈ 8.0

    def test_text_has_moderate_entropy(self):
        """Natural language text has moderate entropy."""
        text = b"The quick brown fox jumps over the lazy dog" * 50
        entropy = calculate_entropy(text)
        assert 3.0 < entropy < 6.0

    def test_returns_float(self):
        assert isinstance(calculate_entropy(b"hello world"), float)


# ─────────────────────────────────────────────────────────────
# Tests: extract_strings
# ─────────────────────────────────────────────────────────────

class TestExtractStrings:

    def test_extracts_known_string(self):
        data = b"\x00\x00" + b"HelloWorld" + b"\x00\x00"
        strings = extract_strings(data, min_len=4)
        assert "HelloWorld" in strings

    def test_minimum_length_filtering(self):
        data = b"AB" + b"\x00" + b"LONGSTRING"
        strings = extract_strings(data, min_len=5)
        assert "AB" not in strings
        assert "LONGSTRING" in strings

    def test_empty_data(self):
        strings = extract_strings(b"", min_len=4)
        assert strings == []

    def test_binary_data_no_strings(self):
        data = bytes(range(256))
        strings = extract_strings(data, min_len=10)
        # May or may not find strings — just ensure it doesn't crash
        assert isinstance(strings, list)


# ─────────────────────────────────────────────────────────────
# Tests: create_sample_dataset
# ─────────────────────────────────────────────────────────────

class TestCreateSampleDataset:

    def test_correct_shape(self):
        df = create_sample_dataset(n_samples=100)
        assert len(df) == 100

    def test_has_label_column(self):
        df = create_sample_dataset(n_samples=100)
        assert 'label' in df.columns

    def test_balanced_classes(self):
        df = create_sample_dataset(n_samples=200)
        assert df['label'].sum() == 100  # 50% malware

    def test_no_nulls_in_features(self):
        df = create_sample_dataset(n_samples=200)
        feature_cols = get_feature_columns(df)
        null_count = df[feature_cols].isnull().sum().sum()
        assert null_count == 0

    def test_labels_binary(self):
        df = create_sample_dataset(n_samples=100)
        assert set(df['label'].unique()).issubset({0, 1})

    def test_entropy_range(self):
        df = create_sample_dataset(n_samples=500)
        # All entropy values should be in [0, 8]
        assert (df['file_entropy'] >= 0).all()
        assert (df['file_entropy'] <= 8.0).all()

    def test_malware_higher_entropy(self):
        """Malware samples should have higher mean entropy than benign."""
        df = create_sample_dataset(n_samples=1000)
        malware_entropy = df[df['label'] == 1]['file_entropy'].mean()
        benign_entropy = df[df['label'] == 0]['file_entropy'].mean()
        assert malware_entropy > benign_entropy

    def test_reproducible_with_seed(self):
        df1 = create_sample_dataset(n_samples=50, random_state=123)
        df2 = create_sample_dataset(n_samples=50, random_state=123)
        pd.testing.assert_frame_equal(df1, df2)


# ─────────────────────────────────────────────────────────────
# Tests: get_feature_columns
# ─────────────────────────────────────────────────────────────

class TestGetFeatureColumns:

    def test_excludes_label(self):
        df = create_sample_dataset(n_samples=10)
        cols = get_feature_columns(df)
        assert 'label' not in cols

    def test_excludes_non_numeric(self):
        df = create_sample_dataset(n_samples=10)
        df['filename'] = 'test.exe'
        cols = get_feature_columns(df)
        assert 'filename' not in cols

    def test_returns_list(self):
        df = create_sample_dataset(n_samples=10)
        cols = get_feature_columns(df)
        assert isinstance(cols, list)

    def test_minimum_feature_count(self):
        """Should have at least 50 features."""
        df = create_sample_dataset(n_samples=10)
        cols = get_feature_columns(df)
        assert len(cols) >= 50


# ─────────────────────────────────────────────────────────────
# Tests: PEFeatureExtractor (non-PE files)
# ─────────────────────────────────────────────────────────────

class TestPEFeatureExtractor:

    def test_nonexistent_file_returns_none(self):
        extractor = PEFeatureExtractor()
        result = extractor.extract("/nonexistent/path/file.exe")
        assert result is None

    def test_non_pe_file_returns_none(self, tmp_path):
        """A plain text file is not a valid PE."""
        txt_file = tmp_path / "notape.exe"
        txt_file.write_bytes(b"This is not a PE file at all.")
        extractor = PEFeatureExtractor()
        result = extractor.extract(str(txt_file))
        assert result is None

    def test_feature_names_consistent(self):
        """All expected feature names should be present in any extraction."""
        expected_features = [
            'file_entropy', 'num_sections', 'mean_entropy',
            'num_imports', 'has_debug_info', 'num_strings',
        ]
        # We can only test with a real PE — skip if not available
        # This tests the structure exists in our sample data
        df = create_sample_dataset(n_samples=5)
        feature_cols = get_feature_columns(df)
        for feat in expected_features:
            assert feat in feature_cols, f"Missing expected feature: {feat}"


# ─────────────────────────────────────────────────────────────
# Integration test: full pipeline on synthetic data
# ─────────────────────────────────────────────────────────────

class TestMLPipeline:

    def test_random_forest_trains_and_predicts(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        df = create_sample_dataset(n_samples=500)
        feature_cols = get_feature_columns(df)
        X = df[feature_cols].values
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        # Even with 10 trees and 500 samples, should get decent AUC
        assert auc > 0.75, f"AUC too low: {auc:.3f}"

    def test_xgboost_trains_and_predicts(self):
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        df = create_sample_dataset(n_samples=500)
        feature_cols = get_feature_columns(df)
        X = df[feature_cols].values
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model = XGBClassifier(n_estimators=50, random_state=42, verbosity=0, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        assert auc > 0.75, f"AUC too low: {auc:.3f}"

    def test_malware_higher_predicted_proba(self):
        """On average, malware samples should have higher predicted probability."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        df = create_sample_dataset(n_samples=1000)
        feature_cols = get_feature_columns(df)
        X = df[feature_cols].values
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=20, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        mean_proba_malware = y_proba[y_test == 1].mean()
        mean_proba_benign = y_proba[y_test == 0].mean()

        assert mean_proba_malware > mean_proba_benign

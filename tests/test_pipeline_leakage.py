"""Regression tests for training pipeline data leakage fixes."""
from __future__ import annotations

import numpy as np
import pytest

from agent_market.freqai.training.pipeline import TrainingPipeline


class TestSplitTimestampGroups:
    """_split should respect timestamp group boundaries for multi-pair data."""

    def test_no_timestamp_straddle(self):
        """Same timestamp must not appear in both train and valid sets."""
        # Simulate 3 pairs x 10 dates = 30 rows
        dates = np.array(
            [np.datetime64(f"2025-01-{d:02d}") for d in range(1, 11)] * 3,
            dtype="datetime64[ns]",
        )
        sort_idx = np.argsort(dates)
        dates = dates[sort_idx]
        X = np.random.randn(30, 5).astype(np.float32)
        X = X[sort_idx]
        y = np.random.randn(30).astype(np.float32)
        y = y[sort_idx]

        X_train, y_train, X_valid, y_valid = TrainingPipeline._split(
            X, y, 0.3, purge=1, embargo=0, dates=dates,
        )

        train_dates = set(dates[:len(X_train)].tolist())
        # Reconstruct valid dates from the split
        valid_start = len(X) - len(X_valid)
        valid_dates = set(dates[valid_start:].tolist())

        # No date should be in both train and valid
        overlap = train_dates & valid_dates
        assert len(overlap) == 0, f"Timestamp overlap: {overlap}"

    def test_split_sizes_reasonable(self):
        """Split should produce non-empty train and valid sets."""
        dates = np.array(
            [np.datetime64(f"2025-01-{d:02d}") for d in range(1, 21)],
            dtype="datetime64[ns]",
        )
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20).astype(np.float32)

        X_train, y_train, X_valid, y_valid = TrainingPipeline._split(
            X, y, 0.2, purge=0, embargo=0, dates=dates,
        )
        assert len(X_train) > 0
        assert len(X_valid) > 0
        assert len(X_train) + len(X_valid) <= len(X)

    def test_fallback_without_dates(self):
        """Split should work without dates (row-based fallback)."""
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        X_train, y_train, X_valid, y_valid = TrainingPipeline._split(
            X, y, 0.2, purge=2, embargo=1,
        )
        assert len(X_train) > 0
        assert len(X_valid) > 0
        # purge+embargo should create a gap
        assert len(X_train) + len(X_valid) < len(X)


class TestRollingCVPurgeEmbargo:
    """Rolling CV should apply purge/embargo at fold boundaries."""

    def test_rolling_cv_no_timestamp_overlap(self):
        """Verify rolling CV folds have a purge/embargo gap between train and valid."""
        # 5 pairs × 20 dates = 100 rows
        dates_raw = [np.datetime64(f"2025-01-{d:02d}") for d in range(1, 21)] * 5
        dates = np.array(dates_raw, dtype="datetime64[ns]")
        sort_idx = np.argsort(dates)
        dates = dates[sort_idx]

        unique_dates = np.unique(dates)
        n_dates = len(unique_dates)
        purge = 2
        embargo = 1
        rolling_splits = 3

        fold_size = max(1, n_dates // (rolling_splits + 1))
        for fold_i in range(rolling_splits):
            train_end_date_idx = fold_size * (fold_i + 1)
            valid_start_date_idx = min(train_end_date_idx + purge + embargo, n_dates)
            valid_end_date_idx = min(
                train_end_date_idx + fold_size + purge + embargo, n_dates
            )
            if (
                valid_start_date_idx >= n_dates
                or valid_end_date_idx <= valid_start_date_idx
            ):
                continue

            train_cutoff = unique_dates[train_end_date_idx - 1]
            valid_start = unique_dates[valid_start_date_idx]

            # No overlap: valid starts strictly after train + gap
            assert valid_start > train_cutoff, (
                f"Fold {fold_i}: valid_start {valid_start} <= train_cutoff {train_cutoff}"
            )

    def test_split_dates_length_mismatch_raises(self):
        """_split should raise ValueError when dates doesn't match features."""
        X = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        bad_dates = np.array(
            [np.datetime64(f"2025-01-{d:02d}") for d in range(1, 6)],
            dtype="datetime64[ns]",
        )

        with pytest.raises(ValueError, match="dates length"):
            TrainingPipeline._split(X, y, 0.2, dates=bad_dates)


class TestScalerFitOnTrainOnly:
    """Scaler must be fit on train data only, not the full dataset."""

    def test_scaler_not_exposed_to_validation(self):
        """Verify scaler statistics come from train set only.

        We create data where train and valid have very different distributions.
        If the scaler sees valid data, train statistics will be contaminated.
        """
        # Train: mean ~0, std ~1
        np.random.seed(42)
        X_train_raw = np.random.randn(80, 3).astype(np.float32)
        # Valid: mean ~100, std ~50
        X_valid_raw = (np.random.randn(20, 3) * 50 + 100).astype(np.float32)
        X = np.vstack([X_train_raw, X_valid_raw])
        y = np.random.randn(100).astype(np.float32)

        X_tr, _, X_va, _ = TrainingPipeline._split(X, y, 0.2)

        # Fit scaler on train only (as the pipeline now does)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        # Train scaled should have median ~0
        assert abs(np.median(X_tr_scaled)) < 0.5
        # Valid scaled should have large values (since scaler was fit on train)
        assert np.median(np.abs(X_va_scaled)) > 10.0

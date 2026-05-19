"""Tests for HQCTracker: hard-abort and checkpoint resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.hqc_tracker import BudgetExceededError, HQCTracker


class TestHQCTracker:
    def test_hard_abort_before_5th_call(self, tmp_path: Path):
        """Budget=50, calls at 13 HQC each — 4 calls succeed, 5th raises."""
        tracker = HQCTracker(
            phase="phase1",
            budget=50.0,
            checkpoint_path=tmp_path / "tracker.json",
        )
        costs = [13.0] * 5
        for i, cost in enumerate(costs):
            ok = tracker.check(cost)
            if not ok:
                with pytest.raises(BudgetExceededError):
                    tracker.hard_abort()
                # Should have raised before record
                assert tracker.spent == pytest.approx(float(i * 13))
                break
            tracker.record(cost, batch_id=i)
        else:
            pytest.fail("BudgetExceededError was never raised")

    def test_checkpoint_resume(self, tmp_path: Path):
        """After crash, reload tracker and verify cumulative HQC restored."""
        tracker = HQCTracker(
            phase="phase1",
            budget=100.0,
            checkpoint_path=tmp_path / "tracker.json",
        )
        tracker.record(13.0, batch_id=0)
        tracker.record(13.0, batch_id=1)
        assert tracker.spent == 26.0
        assert len(tracker.batches) == 2

        # Simulate crash — new instance loads from disk
        tracker2 = HQCTracker.load(
            tmp_path / "tracker.json",
            phase="phase1",
            budget=100.0,
            checkpoint_path=tmp_path / "tracker.json",
        )
        assert tracker2.spent == 26.0
        assert len(tracker2.batches) == 2
        assert tracker2.batches[0].batch_id == 0
        assert tracker2.batches[1].batch_id == 1

    def test_remaining(self, tmp_path: Path):
        tracker = HQCTracker(
            phase="phase1",
            budget=100.0,
            checkpoint_path=tmp_path / "tracker.json",
        )
        tracker.record(30.0, batch_id=0)
        assert tracker.remaining() == 70.0

    def test_reset(self, tmp_path: Path):
        tracker = HQCTracker(
            phase="phase1",
            budget=100.0,
            checkpoint_path=tmp_path / "tracker.json",
        )
        tracker.record(50.0, batch_id=0)
        tracker.reset(phase="phase2", budget=200.0)
        assert tracker.phase == "phase2"
        assert tracker.budget == 200.0
        assert tracker.spent == 0.0
        assert len(tracker.batches) == 0

    def test_to_dict(self, tmp_path: Path):
        tracker = HQCTracker(
            phase="phase1",
            budget=100.0,
            checkpoint_path=tmp_path / "tracker.json",
            git_sha="abc123",
            config_hash="def456",
        )
        tracker.record(13.0, batch_id=0)
        d = tracker.to_dict()
        assert d["phase"] == "phase1"
        assert d["budget_allocated"] == 100.0
        assert d["budget_spent"] == 13.0
        assert d["git_sha"] == "abc123"
        assert d["config_hash"] == "def456"
        assert len(d["batches"]) == 1
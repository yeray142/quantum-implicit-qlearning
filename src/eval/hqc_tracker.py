"""HQC budget tracker with hard-abort and checkpointing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class HQCBatchRecord:
    batch_id: int
    hqc: float
    timestamp: str


@dataclass
class HQCTrackerState:
    phase: str
    budget_allocated: float
    budget_spent: float
    batches: list[HQCBatchRecord]
    checkpoint_path: str
    git_sha: str
    config_hash: str


class HQCTracker:
    """Tracks HQC spending with hard-abort and checkpointing.

    Usage:
        tracker = HQCTracker.load("eval_pools/hqc_tracker.json", phase="phase1", budget=9000.0)
        if not tracker.check(next_cost=13.0):
            tracker.hard_abort()
        tracker.record(cost=13.0, batch_id=0)
    """

    def __init__(
        self,
        phase: str,
        budget: float,
        checkpoint_path: str | Path,
        git_sha: str = "",
        config_hash: str = "",
    ):
        self.phase = phase
        self.budget = budget
        self.spent = 0.0
        self.batches: list[HQCBatchRecord] = []
        self.checkpoint_path = Path(checkpoint_path)
        self.git_sha = git_sha
        self.config_hash = config_hash

    def check(self, next_cost: float) -> bool:
        """Return True if we can afford next_cost without exceeding budget."""
        return (self.spent + next_cost) <= self.budget

    def record(self, cost: float, batch_id: int) -> None:
        """Record spending and persist state to disk."""
        self.spent += cost
        self.batches.append(HQCBatchRecord(
            batch_id=batch_id,
            hqc=cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        self._persist()

    def hard_abort(self) -> None:
        raise BudgetExceededError(
            f"[{self.phase}] Budget exceeded: {self.budget} HQC allocated, "
            f"{self.spent} HQC spent across {len(self.batches)} batches."
        )

    def remaining(self) -> float:
        return self.budget - self.spent

    def _persist(self) -> None:
        state = HQCTrackerState(
            phase=self.phase,
            budget_allocated=self.budget,
            budget_spent=self.spent,
            batches=self.batches,
            checkpoint_path=str(self.checkpoint_path),
            git_sha=self.git_sha,
            config_hash=self.config_hash,
        )
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(asdict(state), f, indent=2)

    def to_dict(self) -> dict:
        return asdict(HQCTrackerState(
            phase=self.phase,
            budget_allocated=self.budget,
            budget_spent=self.spent,
            batches=self.batches,
            checkpoint_path=str(self.checkpoint_path),
            git_sha=self.git_sha,
            config_hash=self.config_hash,
        ))

    @staticmethod
    def load(path: str | Path, **kwargs) -> HQCTracker:
        """Load existing tracker or create new one."""
        path = Path(path)
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            t = HQCTracker(
                phase=state["phase"],
                budget=state["budget_allocated"],
                checkpoint_path=state["checkpoint_path"],
                git_sha=state.get("git_sha", ""),
                config_hash=state.get("config_hash", ""),
            )
            t.spent = state["budget_spent"]
            t.batches = [HQCBatchRecord(**b) for b in state["batches"]]
            return t
        return HQCTracker(checkpoint_path=path, **kwargs)

    def reset(self, phase: str, budget: float) -> None:
        """Reset for a new phase."""
        self.phase = phase
        self.budget = budget
        self.spent = 0.0
        self.batches = []
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelfConfigEvaluation:
    planned_test_error: float
    full_search_test_error: float
    n_planned_candidates: int
    n_full_candidates: int

    @property
    def regret(self) -> float:
        return float(self.planned_test_error - self.full_search_test_error)

    @property
    def relative_regret(self) -> float:
        if self.full_search_test_error == 0:
            return float("nan")
        return float(self.regret / self.full_search_test_error)

    @property
    def search_reduction(self) -> float:
        if self.n_full_candidates <= 0:
            return float("nan")
        return float(1.0 - self.n_planned_candidates / self.n_full_candidates)

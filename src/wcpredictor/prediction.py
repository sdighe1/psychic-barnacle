"""Unified prediction container shared by every model.

A :class:`Prediction` always carries win/draw/loss probabilities and expected
goals; when a model can produce a full scoreline distribution (Dixon-Coles, the
ensemble) it also carries a ``score_matrix`` where ``score_matrix[i, j]`` is the
probability of the exact result *home i – away j*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Confidence banding on the most-likely outcome probability.
_CONF_HIGH = 0.60
_CONF_MED = 0.45


@dataclass
class Prediction:
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float
    exp_away_goals: float
    score_matrix: Optional[np.ndarray] = None

    # -- outcome ------------------------------------------------------- #
    @property
    def probs(self) -> dict[str, float]:
        return {"home": self.p_home, "draw": self.p_draw, "away": self.p_away}

    @property
    def favorite(self) -> str:
        return max(self.probs, key=self.probs.get)

    @property
    def confidence(self) -> float:
        return float(max(self.p_home, self.p_draw, self.p_away))

    @property
    def confidence_label(self) -> str:
        c = self.confidence
        if c >= _CONF_HIGH:
            return "High"
        if c >= _CONF_MED:
            return "Medium"
        return "Low"

    # -- scoreline ----------------------------------------------------- #
    def most_likely_score(self) -> tuple[int, int]:
        if self.score_matrix is not None:
            i, j = np.unravel_index(int(np.argmax(self.score_matrix)), self.score_matrix.shape)
            return int(i), int(j)
        return int(round(self.exp_home_goals)), int(round(self.exp_away_goals))

    def score_probability(self, i: int, j: int) -> Optional[float]:
        if self.score_matrix is None:
            return None
        if 0 <= i < self.score_matrix.shape[0] and 0 <= j < self.score_matrix.shape[1]:
            return float(self.score_matrix[i, j])
        return 0.0

    def top_scorelines(self, k: int = 3) -> list[tuple[int, int, float]]:
        if self.score_matrix is None:
            i, j = self.most_likely_score()
            return [(i, j, float("nan"))]
        flat = np.argsort(self.score_matrix, axis=None)[::-1][:k]
        out = []
        for idx in flat:
            i, j = np.unravel_index(int(idx), self.score_matrix.shape)
            out.append((int(i), int(j), float(self.score_matrix[i, j])))
        return out

    # -- serialisation ------------------------------------------------- #
    def to_dict(self) -> dict:
        i, j = self.most_likely_score()
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "p_home": round(self.p_home, 4),
            "p_draw": round(self.p_draw, 4),
            "p_away": round(self.p_away, 4),
            "exp_home_goals": round(self.exp_home_goals, 3),
            "exp_away_goals": round(self.exp_away_goals, 3),
            "projected_score": [i, j],
            "projected_score_prob": (round(self.score_probability(i, j), 4)
                                     if self.score_matrix is not None else None),
            "confidence": round(self.confidence, 4),
            "confidence_label": self.confidence_label,
            "favorite": (self.home_team if self.favorite == "home"
                         else self.away_team if self.favorite == "away" else "Draw"),
            "top_scorelines": [
                {"score": [i, j], "prob": (round(p, 4) if p == p else None)}
                for i, j, p in self.top_scorelines(3)
            ],
        }

    def summary(self) -> str:
        i, j = self.most_likely_score()
        return (f"{self.home_team} {i}-{j} {self.away_team}  "
                f"[{self.home_team} {self.p_home:.0%} / Draw {self.p_draw:.0%} / "
                f"{self.away_team} {self.p_away:.0%}]  "
                f"confidence: {self.confidence_label} ({self.confidence:.0%})")


def outcome_from_scores(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"

"""Result models for the S11 payload intelligence stage.

Every conclusion travels as a Finding with confidence, evidence and
limitations. Verdict wording follows a fixed scoring policy (see
confidence.py) so the report never uses misleading certainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Finding:
    category: str                 # e.g. "text", "compression", "protocol"
    verdict: str                  # short human-readable claim
    confidence: float             # 0..1 per the scoring policy
    evidence: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(float(self.confidence), 3)
        d["strength"] = confidence_band(self.confidence)
        return d


def confidence_band(c: float) -> str:
    """Fixed scoring policy shared by every S11 module."""
    if c >= 0.90:
        return "VALIDATED"
    if c >= 0.75:
        return "LIKELY"
    if c >= 0.50:
        return "POSSIBLE"
    return "WEAK"

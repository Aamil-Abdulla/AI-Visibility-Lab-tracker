"""Data record written to the CSV. One row = one saved AI answer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class AnswerRecord:
    run_id: str
    timestamp: str          # UTC, ISO format
    day: str                # local date, used to separate run days
    provider: str
    model: str
    question_id: str
    question: str
    run_number: int
    answer_text: str        # full raw answer, so every number can be traced back
    brands_found: str       # ";" separated canonical brand names
    domains_found: str      # ";" separated, "as stated by the model" (may be invented)
    finish_reason: str = "unknown"  # "stop" = finished normally, "length" = cut off at max_tokens

    def to_row(self) -> dict:
        return asdict(self)


CSV_FIELDS = [f.name for f in fields(AnswerRecord)]

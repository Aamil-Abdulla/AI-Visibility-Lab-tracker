"""CSV storage. Each answer is appended and closed immediately, so a crash loses nothing."""
from __future__ import annotations

import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

from avl.models import CSV_FIELDS, AnswerRecord


class StorageError(Exception):
    """The CSV on disk does not match the expected columns."""


class CsvStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _existing_header(self) -> list[str] | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with open(self.path, newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None)

    def check_format(self) -> None:
        header = self._existing_header()
        if header is not None and header != CSV_FIELDS:
            raise StorageError(
                f"{self.path} has an old column layout. Run once: python -m avl migrate "
                "(it makes a backup first)."
            )

    def append(self, record: AnswerRecord) -> None:
        self.check_format()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = self._existing_header() is None
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record.to_row())

    def completed(self, run_id: str) -> set[tuple[str, int]]:
        """(question_id, run_number) pairs already saved for this run_id."""
        done: set[tuple[str, int]] = set()
        if not self.path.exists():
            return done
        with open(self.path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("run_id") == run_id:
                    done.add((row["question_id"], int(row["run_number"])))
        return done

    def migrate(self) -> int:
        """Add any missing columns (old rows get 'unknown'). Makes a backup first. Returns rows migrated."""
        header = self._existing_header()
        if header is None or header == CSV_FIELDS:
            return 0
        backup = self.path.with_name(f"{self.path.name}.backup-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(self.path, backup)
        with open(self.path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) or ("unknown" if k == "finish_reason" else "") for k in CSV_FIELDS})
        os.replace(tmp, self.path)
        return len(rows)

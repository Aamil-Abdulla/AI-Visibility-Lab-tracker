"""Find brand names and domains in answer text."""
from __future__ import annotations

import re

from avl.config import Brand

# Simple domain finder. It can miss unusual endings and can match odd text,
# so domains are always labelled "as stated by the model".
DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|ae|net|org|co|io|shop|store|uk|de|eu)\b",
    re.IGNORECASE,
)


def find_domains(text: str) -> list[str]:
    found = {d.lower().removeprefix("www.") for d in DOMAIN_PATTERN.findall(text)}
    return sorted(found)


class BrandMatcher:
    """Case-insensitive whole-word matching, with optional aliases per brand."""

    def __init__(self, brands: list[Brand]):
        self._patterns: list[tuple[str, re.Pattern]] = []
        for brand in brands:
            names = [brand.name, *brand.aliases]
            alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
            pattern = re.compile(
                r"(?<![A-Za-z0-9])(?:" + alternatives + r")(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            self._patterns.append((brand.name, pattern))

    def find(self, text: str) -> list[str]:
        return [name for name, pattern in self._patterns if pattern.search(text)]

    def find_ordered(self, text: str) -> list[str]:
        """Brands in the order they FIRST appear in the text (used for the stability metric)."""
        hits = []
        for name, pattern in self._patterns:
            m = pattern.search(text)
            if m:
                hits.append((m.start(), name))
        return [name for _, name in sorted(hits)]

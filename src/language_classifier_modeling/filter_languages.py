"""Filter the raw Tatoeba sentences dump down to the 20 chosen languages.

The raw file (``data/raw_data/sentences.csv``) is tab-separated, no header,
columns ``id``, ``lang``, ``text``, and covers ~430 languages / ~13.6M rows
(~727MB) — far too large to use or commit directly.

This script streams the raw file once and keeps a random reservoir sample
of up to ``MAX_PER_LANGUAGE`` sentences for each chosen language, so the
output is small enough to commit while still giving every later sampling
step (the sample-size learning-curve screen, which goes up to 6,000/language)
plenty of headroom to draw different random subsamples from.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# Romance cluster
ROMANCE = ["spa", "por", "ita", "fra", "cat", "glg"]
# East Slavic cluster
EAST_SLAVIC = ["rus", "ukr", "bel"]
# Distinct contrast set
DISTINCT = ["eng", "deu", "jpn", "cmn", "ara", "fin", "tur", "srp", "ell", "hin", "kab"]

LANGUAGES = ROMANCE + EAST_SLAVIC + DISTINCT

MAX_PER_LANGUAGE = 10_000
SEED = 0

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw_data" / "sentences.csv"
OUT_PATH = REPO_ROOT / "data" / "filtered" / "sentences_filtered.csv"


def reservoir_sample_by_language(
    raw_path: Path, languages: list[str], cap: int, seed: int
) -> dict[str, list[tuple[str, str, str]]]:
    """Single pass over raw_path; reservoir-sample up to `cap` rows per language."""
    rng = random.Random(seed)
    wanted = set(languages)
    reservoirs: dict[str, list[tuple[str, str, str]]] = {lang: [] for lang in languages}
    seen_counts: dict[str, int] = {lang: 0 for lang in languages}

    with raw_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) != 3:
                continue
            sent_id, lang, text = row
            if lang not in wanted:
                continue
            seen_counts[lang] += 1
            reservoir = reservoirs[lang]
            if len(reservoir) < cap:
                reservoir.append((sent_id, lang, text))
            else:
                j = rng.randint(0, seen_counts[lang] - 1)
                if j < cap:
                    reservoir[j] = (sent_id, lang, text)

    return reservoirs


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}. Download the Tatoeba sentences "
            "dump first (see README.md)."
        )

    reservoirs = reservoir_sample_by_language(RAW_PATH, LANGUAGES, MAX_PER_LANGUAGE, SEED)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        for lang in LANGUAGES:
            for row in reservoirs[lang]:
                writer.writerow(row)

    print(f"Wrote {OUT_PATH}")
    print(f"{'language':10s} {'available':>10s} {'kept':>10s}")
    for lang in LANGUAGES:
        kept = len(reservoirs[lang])
        available = "<=cap" if kept == MAX_PER_LANGUAGE else str(kept)
        print(f"{lang:10s} {available:>10s} {kept:>10d}")


if __name__ == "__main__":
    main()

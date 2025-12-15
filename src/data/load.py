from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


def triplet_news_txt(path: str = "data/AAPL.txt") -> pd.DataFrame:
    """
    Parse a simple text file where each article is encoded as 3 consecutive lines:
        1) title/headline
        2) date string (e.g. 'Dec. 09')
        3) source code (e.g. 'ZD', 'MT')

    Blank lines are ignored. If the total number of nonempty lines is not
    a multiple of 3, the last incomplete block is dropped.

    Returns
    -------
    DataFrame with columns:
        ['title', 'date_raw', 'source_code', 'body']
    where 'body' is (for now) just equal to the title so it fits the
    sentiment pipeline, which expects some 'text' field.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    raw = p.read_text(encoding="utf-8", errors="ignore")
    lines: List[str] = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    if len(lines) < 3:
        raise ValueError(f"Expected at least 3 non-empty lines, got {len(lines)}.")

    if len(lines) % 3 != 0:
        print(
            f"[load_triplet_news_txt] Warning: {len(lines)} nonempty lines "
            f"not divisible by 3. Truncating the last {len(lines) % 3} line(s)."
        )
        lines = lines[: len(lines) // 3 * 3]

    records: List[Dict[str, str]] = []
    for i in range(0, len(lines), 3):
        title = lines[i]
        date_raw = lines[i + 1]
        source_code = lines[i + 2]

        records.append(
            {
                "title": title,
                "date_raw": date_raw,
                "source_code": source_code,
                # For now, we have no full article body, only the headline.
                # We map body = title so sentiment function has some text field.
                "body": title,
            }
        )

    df = pd.DataFrame.from_records(records)
    return df
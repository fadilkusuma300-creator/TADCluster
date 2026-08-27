"""Build TADCluster input tables from public data sources."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import random
import re
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence
import xml.etree.ElementTree as ET

import pandas as pd
from bs4 import BeautifulSoup

SEED = 42
TARGET_TAGS = ("java", "python", "javascript", "machine-learning")
D3_YEAR_COUNTS = {2017: 3333, 2018: 3333, 2019: 3333, 2020: 3333, 2021: 3332, 2022: 3332}


@dataclass
class Reservoir:
    limit: int
    rng: random.Random
    seen: int = 0
    rows: List[dict] | None = None

    def __post_init__(self) -> None:
        self.rows = []

    def add(self, row: dict) -> None:
        self.seen += 1
        assert self.rows is not None
        if len(self.rows) < self.limit:
            self.rows.append(row)
            return
        j = self.rng.randrange(self.seen)
        if j < self.limit:
            self.rows[j] = row


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _strip_html_and_block_code(title: str, body: str) -> str:
    """Remove markup and standalone code blocks while retaining inline technical tokens."""
    title = html.unescape(title or "")
    body = html.unescape(body or "")
    soup = BeautifulSoup(body, "html.parser")
    for node in soup.find_all("pre"):
        node.decompose()
    for node in soup.find_all("code"):
        node.unwrap()
    body_text = soup.get_text(" ", strip=True)
    combined = f"{title} {body_text}".lower()
    return re.sub(r"\s+", " ", combined).strip()


def _parse_tags(raw: str) -> List[str]:
    if not raw:
        return []
    return re.findall(r"<([^<>]+)>", raw.lower())


def _iter_xml_rows(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "row":
            yield dict(elem.attrib)
        elem.clear()


def _question_row(attrs: Mapping[str, str], source: str) -> dict | None:
    if attrs.get("PostTypeId") != "1":
        return None
    created = _parse_iso_datetime(attrs.get("CreationDate", ""))
    if created is None:
        return None
    text = _strip_html_and_block_code(attrs.get("Title", ""), attrs.get("Body", ""))
    if not text:
        return None
    return {
        "id": str(attrs.get("Id", "")),
        "text": text,
        "timestamp": created.isoformat(),
        "source": source,
        "score": int(attrs.get("Score", "0") or 0),
        "tags": "|".join(_parse_tags(attrs.get("Tags", ""))),
    }


def _primary_target_tag(tags: Sequence[str]) -> str | None:
    present = set(tags)
    for tag in TARGET_TAGS:
        if tag in present:
            return tag
    return None


def _balanced_counts(total: int, strata: Sequence[str]) -> Dict[str, int]:
    base, remainder = divmod(total, len(strata))
    return {name: base + (1 if i < remainder else 0) for i, name in enumerate(strata)}


def build_d1(stackoverflow_dir: Path, output: Path, seed: int = SEED) -> None:
    posts = stackoverflow_dir / "Posts.xml"
    comments = stackoverflow_dir / "Comments.xml"
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    stop = datetime(2023, 1, 1, tzinfo=timezone.utc)

    candidates: Dict[str, dict] = {}
    strata: Dict[str, List[str]] = defaultdict(list)
    for attrs in _iter_xml_rows(posts):
        row = _question_row(attrs, "StackOverflow")
        if row is None or row["score"] < 5:
            continue
        dt = _parse_iso_datetime(row["timestamp"])
        if dt is None or not (start <= dt < stop):
            continue
        tags = row["tags"].split("|") if row["tags"] else []
        stratum = _primary_target_tag(tags)
        if stratum is None:
            continue
        row["stratum"] = stratum
        candidates[row["id"]] = row
        strata[stratum].append(row["id"])

    if not candidates:
        raise RuntimeError("No D1 candidates were found. Check the Stack Overflow dump directory.")

    comment_counts: Counter[str] = Counter()
    candidate_ids = set(candidates)
    for attrs in _iter_xml_rows(comments):
        post_id = attrs.get("PostId")
        if post_id in candidate_ids:
            comment_counts[post_id] += 1

    eligible_by_tag: Dict[str, List[dict]] = defaultdict(list)
    for post_id, row in candidates.items():
        if comment_counts[post_id] >= 3:
            row = dict(row)
            row["comment_count"] = int(comment_counts[post_id])
            eligible_by_tag[row["stratum"]].append(row)

    targets = _balanced_counts(1698, TARGET_TAGS)
    rng = random.Random(seed)
    selected: List[dict] = []
    for tag in TARGET_TAGS:
        pool = eligible_by_tag[tag]
        if len(pool) < targets[tag]:
            raise RuntimeError(
                f"D1 stratum '{tag}' has {len(pool)} eligible rows, fewer than required {targets[tag]}."
            )
        pool = sorted(pool, key=lambda x: int(x["id"]))
        selected.extend(rng.sample(pool, targets[tag]))

    _write_study_csv(selected, output)


def _reservoir_questions(
    posts_xml: Path,
    source: str,
    start: datetime,
    stop: datetime,
    limit: int,
    seed: int,
) -> List[dict]:
    reservoir = Reservoir(limit=limit, rng=random.Random(seed))
    for attrs in _iter_xml_rows(posts_xml):
        row = _question_row(attrs, source)
        if row is None:
            continue
        dt = _parse_iso_datetime(row["timestamp"])
        if dt is not None and start <= dt < stop:
            reservoir.add(row)
    assert reservoir.rows is not None
    if len(reservoir.rows) != limit:
        raise RuntimeError(f"{source}: only {len(reservoir.rows)} rows were available; {limit} are required")
    return reservoir.rows


def build_d2(
    superuser_dir: Path,
    serverfault_dir: Path,
    askubuntu_dir: Path,
    output: Path,
    seed: int = SEED,
    start: datetime | None = None,
    stop: datetime | None = None,
) -> None:
    if start is None or stop is None:
        raise ValueError("D2 requires explicit start and stop timestamps")
    if start >= stop:
        raise ValueError("D2 start must precede stop")
    specs = [
        (superuser_dir, "SuperUser", 300, seed + 11),
        (serverfault_dir, "ServerFault", 300, seed + 23),
        (askubuntu_dir, "AskUbuntu", 300, seed + 37),
    ]
    selected: List[dict] = []
    for folder, source, count, local_seed in specs:
        selected.extend(_reservoir_questions(folder / "Posts.xml", source, start, stop, count, local_seed))
    _write_study_csv(selected, output)


def build_d3(stackoverflow_dir: Path, output: Path, seed: int = SEED) -> None:
    reservoirs = {
        year: Reservoir(limit=count, rng=random.Random(seed + year))
        for year, count in D3_YEAR_COUNTS.items()
    }
    for attrs in _iter_xml_rows(stackoverflow_dir / "Posts.xml"):
        row = _question_row(attrs, "StackOverflow")
        if row is None:
            continue
        dt = _parse_iso_datetime(row["timestamp"])
        if dt is None or dt.year not in reservoirs:
            continue
        reservoirs[dt.year].add(row)

    selected: List[dict] = []
    for year in sorted(reservoirs):
        reservoir = reservoirs[year]
        assert reservoir.rows is not None
        if len(reservoir.rows) != reservoir.limit:
            raise RuntimeError(f"D3 year {year}: expected {reservoir.limit} rows, found {len(reservoir.rows)}")
        selected.extend(reservoir.rows)
    _write_study_csv(selected, output)


def _write_study_csv(rows: Sequence[Mapping], output: Path) -> None:
    if not rows:
        raise ValueError("no rows to write")
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    required = ["id", "text", "timestamp", "source"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.drop_duplicates(subset=["id", "source"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values(["timestamp", "source", "id"]).reset_index(drop=True)
    df.to_csv(output, index=False)
    print(f"wrote {len(df)} rows -> {output}")
    print(f"time span: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400:.1f} days")


def _load_json_array(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        obj = json.load(fh)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("documents", "articles", "data", "items"):
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _get_first(item: Mapping, names: Sequence[str]):
    for name in names:
        if name in item and item[name] not in (None, ""):
            return item[name]
    return None


def _normalize_news_language(value) -> str:
    return str(value or "").strip().lower()


def _pick_news_label_field(items: Sequence[Mapping], expected_events: int | None) -> str:
    candidates = ("event_id", "eventUri", "event_uri", "cluster", "story_id", "event")
    scored = []
    for field in candidates:
        values = [str(x[field]) for x in items if x.get(field) not in (None, "")]
        if not values:
            continue
        unique = len(set(values))
        error = abs(unique - expected_events) if expected_events else 0
        scored.append((error, -len(values), field, unique))
    if not scored:
        raise ValueError("Could not locate an event-label field in the News2013 JSON")
    scored.sort()
    _, _, field, unique = scored[0]
    if expected_events is not None and unique != expected_events:
        raise ValueError(
            f"Best label field '{field}' has {unique} events, expected {expected_events}. "
            "Verify that the English split from the published dataset is being used."
        )
    return field


def build_news2013_split(input_json: Path, output: Path, expected_docs: int, expected_events: int) -> None:
    raw = _load_json_array(input_json)
    english = []
    for item in raw:
        lang = _normalize_news_language(_get_first(item, ("lang", "language")))
        if lang not in {"eng", "en", "english"}:
            continue
        english.append(item)
    if len(english) != expected_docs:
        raise ValueError(
            f"English split contains {len(english)} documents; expected {expected_docs}. "
            "Use dataset.dev.json / dataset.test.json from the Priberam release."
        )
    label_field = _pick_news_label_field(english, expected_events)

    rows = []
    for item in english:
        body = _get_first(item, ("text", "body", "content", "article")) or ""
        title = _get_first(item, ("title", "headline")) or ""
        date = _get_first(item, ("date", "timestamp", "published", "published_at"))
        time_value = _get_first(item, ("time",))
        if time_value is not None and date is not None:
            date_text = str(date).strip()
            # Priberam releases may store date and time in separate fields. Avoid
            # appending time when the date field already contains a clock value.
            if not re.search(r"[T\s]\d{1,2}:\d{2}", date_text):
                date = f"{date_text} {time_value}"
        dt = pd.to_datetime(date, utc=True, errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"Invalid News2013 timestamp for document {item.get('id')}")
        text = _strip_html_and_block_code(str(title), str(body))
        source_value = _get_first(item, ("source", "publisher"))
        if isinstance(source_value, Mapping):
            source_value = _get_first(source_value, ("title", "name", "id"))
        rows.append(
            {
                "id": str(_get_first(item, ("id", "article_id", "doc_id"))),
                "text": text,
                "timestamp": dt.isoformat(),
                "source": str(source_value or "News2013"),
                "event_label": str(item[label_field]),
            }
        )
    _write_news_csv(rows, output, expected_events)


def _write_news_csv(rows: Sequence[Mapping], output: Path, expected_events: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df["id"].duplicated().any():
        raise ValueError("Duplicate News2013 document ids detected")
    if df["event_label"].nunique() != expected_events:
        raise ValueError("Unexpected News2013 event count after conversion")
    df = df.sort_values(["timestamp", "id"]).reset_index(drop=True)
    df.to_csv(output, index=False)
    print(f"wrote {len(df)} rows / {df['event_label'].nunique()} events -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("d1", help="Build D1 from a Stack Overflow XML dump")
    p1.add_argument("--stackoverflow", type=Path, required=True, help="Folder containing Posts.xml and Comments.xml")
    p1.add_argument("--output", type=Path, default=Path("data/D1.csv"))

    p2 = sub.add_parser("d2", help="Build D2 from three Stack Exchange XML dumps")
    p2.add_argument("--superuser", type=Path, required=True)
    p2.add_argument("--serverfault", type=Path, required=True)
    p2.add_argument("--askubuntu", type=Path, required=True)
    p2.add_argument("--start", required=True, help="Inclusive UTC start timestamp (ISO-8601)")
    p2.add_argument("--stop", required=True, help="Exclusive UTC stop timestamp (ISO-8601)")
    p2.add_argument("--output", type=Path, default=Path("data/D2.csv"))

    p3 = sub.add_parser("d3", help="Build D3 from a Stack Overflow XML dump")
    p3.add_argument("--stackoverflow", type=Path, required=True)
    p3.add_argument("--output", type=Path, default=Path("data/D3.csv"))

    pn = sub.add_parser("news2013", help="Convert the published News2013 English splits")
    pn.add_argument("--dev", type=Path, required=True, help="Priberam dataset.dev.json")
    pn.add_argument("--test", type=Path, required=True, help="Priberam dataset.test.json")
    pn.add_argument("--output-dir", type=Path, default=Path("data"))

    args = parser.parse_args()
    if args.command == "d1":
        build_d1(args.stackoverflow, args.output)
    elif args.command == "d2":
        start = _parse_iso_datetime(args.start)
        stop = _parse_iso_datetime(args.stop)
        if start is None or stop is None:
            raise ValueError("D2 --start and --stop must be valid ISO-8601 datetimes")
        build_d2(args.superuser, args.serverfault, args.askubuntu, args.output, start=start, stop=stop)
    elif args.command == "d3":
        build_d3(args.stackoverflow, args.output)
    elif args.command == "news2013":
        # The English release contains 12,233 development documents and 8,726 test
        # documents, with 593 and 222 event labels, respectively.
        build_news2013_split(args.dev, args.output_dir / "news2013_train.csv", 12_233, 593)
        build_news2013_split(args.test, args.output_dir / "news2013_test.csv", 8_726, 222)


if __name__ == "__main__":
    main()

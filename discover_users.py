"""F1: discover active starring users from GH Archive.

Stream the trailing N days of hourly GH Archive files (~21MB gzip each), keep
WatchEvents, count stars per actor, drop bots, and write the top users to
data/seed_users.csv. Consumed once by F2; not a feature group.

Deployed as a Hopsworks job (python-feature-pipeline).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent / "data"
ARCHIVE = "https://data.gharchive.org/{:%Y-%m-%d-%-H}.json.gz"
BOT_RE = re.compile(r"(\[bot\]$|[-_]bot$|^bot[-_]|^(dependabot|renovate|github-actions))", re.I)
PREFILTER = b'"type":"WatchEvent"'


def hours(days: int, end: datetime) -> list[datetime]:
    start = end - timedelta(days=days)
    out, t = [], start
    while t < end:
        out.append(t)
        t += timedelta(hours=1)
    return out


def count_stars(days: int, end: datetime) -> Counter:
    users: Counter = Counter()
    hrs = hours(days, end)
    for i, h in enumerate(hrs):
        url = ARCHIVE.format(h)
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"skip {url}: {e}", flush=True)
            continue
        n = 0
        with gzip.GzipFile(fileobj=resp.raw) as fh:
            for line in fh:
                if PREFILTER not in line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") != "WatchEvent":
                    continue
                login = ev["actor"]["login"]
                if BOT_RE.search(login):
                    continue
                users[login] += 1
                n += 1
        if i % 24 == 0:
            print(f"{h:%Y-%m-%d-%H}: {n} stars this hour, {len(users)} users total", flush=True)
    return users


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--end", default=None, help="ISO hour, default now-3h (archive lag)")
    ap.add_argument("--max-users", type=int, default=20000)
    ap.add_argument("--min-stars", type=int, default=2)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)
    DATA.mkdir(parents=True, exist_ok=True)

    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    )
    print(f"window: {args.days}d ending {end:%Y-%m-%d-%H} UTC", flush=True)

    users = count_stars(args.days, end)
    ranked = [(u, c) for u, c in users.most_common() if c >= args.min_stars][: args.max_users]
    if len(ranked) < args.max_users:  # backfill with single-star users, most recent window is thin
        singles = [(u, c) for u, c in users.most_common() if c < args.min_stars]
        ranked += singles[: args.max_users - len(ranked)]

    out = DATA / "seed_users.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["login", "window_stars"])
        w.writerows(ranked)
    print(f"wrote {len(ranked)} users to {out} (of {len(users)} seen)", flush=True)


if __name__ == "__main__":
    main()

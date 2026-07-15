"""F2: pull each seed user's full star history and own repos from the GitHub API.

Reads data/seed_users.csv (F1), snowballs with GITHUB_TOKEN (project secret,
5000 req/hr), checkpoints one JSONL line per user so the overnight run is
resumable, then writes three feature groups:

  star_events  (user_login, repo_id, starred_at)      the interaction matrix
  repos        (repo_id, metadata..., captured_at)     every repo seen
  own_repos    (user_login, repo_id, captured_at)      what each user builds

Rate limiting is bounded active polling on the X-RateLimit-Reset header.
Deployed as a Hopsworks job (python-feature-pipeline).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import hopsworks
from unstarred_features import REPO_DOC, STAR_DOC, now_utc, repo_row

DATA = Path(__file__).resolve().parent / "data"
API = "https://api.github.com"
STAR_ACCEPT = "application/vnd.github.star+json"
PER_PAGE = 100


def token() -> str:
    hopsworks.login()
    return hopsworks.get_secrets_api().get_secret("GITHUB_TOKEN").value


def make_session(tok: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {tok}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "unstarred-012",
        }
    )
    return s


def wait_for_reset(resp: requests.Response) -> None:
    remaining = int(resp.headers.get("X-RateLimit-Remaining", "1"))
    if remaining > 50:
        return
    reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
    while time.time() < reset:
        left = int(reset - time.time()) + 1
        print(f"rate limit: sleeping {min(left, 60)}s ({left}s to reset)", flush=True)
        time.sleep(min(left, 60))


def get(s: requests.Session, url: str, accept: str | None = None, params: dict | None = None):
    for attempt in range(3):
        try:
            resp = s.get(url, params=params, timeout=60, headers={"Accept": accept} if accept else None)
        except requests.RequestException as e:
            print(f"retry {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code in (403, 429):
            wait_for_reset(resp)
            continue
        resp.raise_for_status()
        wait_for_reset(resp)
        return resp.json()
    return None


def pull_user(s: requests.Session, login: str, max_pages: int, captured: datetime) -> dict | None:
    stars = []
    for page in range(1, max_pages + 1):
        batch = get(
            s,
            f"{API}/users/{login}/starred",
            accept=STAR_ACCEPT,
            params={"per_page": PER_PAGE, "page": page},
        )
        if batch is None:
            return None if page == 1 else {"login": login, "stars": stars, "own": []}
        for item in batch:
            if "repo" not in item:  # accept header ignored, no timestamps: skip user
                return None
            stars.append({"starred_at": item["starred_at"], "repo": repo_row(item["repo"], captured)})
        if len(batch) < PER_PAGE:
            break
    own_batch = get(s, f"{API}/users/{login}/repos", params={"per_page": PER_PAGE, "sort": "pushed"}) or []
    own = [repo_row(r, captured) for r in own_batch if not r.get("private")]
    return {"login": login, "stars": stars, "own": own}


def pull(limit: int | None, max_pages: int) -> Path:
    seeds = [row["login"] for row in csv.DictReader((DATA / "seed_users.csv").open())]
    if limit:
        seeds = seeds[:limit]
    ckpt = DATA / "pull_checkpoint.jsonl"
    done = set()
    if ckpt.exists():
        with ckpt.open() as fh:
            done = {json.loads(line)["login"] for line in fh if line.strip()}
    todo = [u for u in seeds if u not in done]
    print(f"seeds {len(seeds)}, done {len(done)}, todo {len(todo)}", flush=True)

    s = make_session(token())
    captured = now_utc()
    with ckpt.open("a") as out:
        for i, login in enumerate(todo):
            rec = pull_user(s, login, max_pages, captured)
            out.write(json.dumps(rec or {"login": login, "stars": [], "own": []}, default=str) + "\n")
            out.flush()
            if i % 200 == 0:
                print(f"{i}/{len(todo)} users pulled", flush=True)
    return ckpt


def frames(ckpt: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    star_rows, repo_rows, own_rows = [], {}, []
    with ckpt.open() as fh:
        for line in fh:
            rec = json.loads(line)
            for s in rec.get("stars", []):
                r = s["repo"]
                star_rows.append(
                    {"user_login": rec["login"], "repo_id": r["repo_id"], "starred_at": s["starred_at"]}
                )
                repo_rows.setdefault(r["repo_id"], r)
            for r in rec.get("own", []):
                repo_rows.setdefault(r["repo_id"], r)
                own_rows.append(
                    {"user_login": rec["login"], "repo_id": r["repo_id"], "captured_at": r["captured_at"]}
                )
    stars = pd.DataFrame(star_rows)
    stars["starred_at"] = pd.to_datetime(stars["starred_at"], utc=True, format="mixed")
    repos = pd.DataFrame(repo_rows.values())
    for col in ("created_at", "pushed_at", "captured_at"):
        repos[col] = pd.to_datetime(repos[col], utc=True, format="mixed")
    own = pd.DataFrame(own_rows)
    own["captured_at"] = pd.to_datetime(own["captured_at"], utc=True, format="mixed")
    return stars, repos, own


def write_fgs(stars: pd.DataFrame, repos: pd.DataFrame, own: pd.DataFrame) -> None:
    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="star_events",
        version=1,
        description="Timestamped user-repo star interactions pulled from the GitHub API",
        primary_key=["user_login", "repo_id"],
        event_time="starred_at",
        online_enabled=False,
    )
    fg.insert(stars)
    for name, desc in STAR_DOC.items():
        fg.update_feature_description(name, desc)

    fg = fs.get_or_create_feature_group(
        name="repos",
        version=1,
        description="Repo metadata at capture time for every repo seen in the pull",
        primary_key=["repo_id"],
        event_time="captured_at",
        online_enabled=False,
    )
    fg.insert(repos)
    for name, desc in REPO_DOC.items():
        fg.update_feature_description(name, desc)

    fg = fs.get_or_create_feature_group(
        name="own_repos",
        version=1,
        description="Which public repos each seed user owns (user-side taste evidence)",
        primary_key=["user_login", "repo_id"],
        event_time="captured_at",
        online_enabled=False,
    )
    fg.insert(own)
    print(f"wrote FGs: star_events {len(stars)}, repos {len(repos)}, own_repos {len(own)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--skip-pull", action="store_true", help="write FGs from the existing checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="pull + build frames, no FG write")
    args = ap.parse_args()

    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)

    ckpt = DATA / "pull_checkpoint.jsonl" if args.skip_pull else pull(args.limit, args.max_pages)
    stars, repos, own = frames(ckpt)
    print(f"stars {len(stars)} ({stars['user_login'].nunique()} users, {stars['repo_id'].nunique()} repos), "
          f"repos {len(repos)}, own {len(own)}", flush=True)
    if args.dry_run:
        print(stars.head().to_string())
        return
    write_fgs(stars, repos, own)


if __name__ == "__main__":
    main()

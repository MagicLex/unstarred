"""Shared, pure feature extraction for unstarred.

One module imported by every pipeline (F2 pull, T1 train, I2 serve, the app), so
training and serving cannot skew. No I/O here: GitHub API objects in, flat rows
out.

Temporal honesty is in the signatures: everything user-side takes `asof` and only
looks at history strictly before it.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone

MAX_TOPICS = 8
HISTORY_LEN = 50  # most recent stars fed to the user tower

REPO_DOC = {
    "repo_id": "GitHub numeric repo id (primary key)",
    "full_name": "owner/name",
    "owner_login": "repo owner login",
    "language": "primary language reported by GitHub, 'none' when absent",
    "topics": "space-joined first 8 repo topics, '' when absent",
    "stars": "stargazer count at capture",
    "forks": "fork count at capture",
    "open_issues": "open issue count at capture",
    "size_kb": "repo size in KB at capture",
    "created_at": "repo creation timestamp",
    "pushed_at": "last push timestamp at capture",
    "is_fork": "repo is a fork",
    "archived": "repo is archived",
    "description": "repo description, '' when absent",
    "captured_at": "capture timestamp (event time)",
}

STAR_DOC = {
    "user_login": "starring user (primary key with repo_id)",
    "repo_id": "starred repo id",
    "starred_at": "star timestamp from the GitHub API (event time)",
}


def _iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def repo_row(repo: dict, captured_at: datetime) -> dict:
    """GitHub API repo object (from /starred .repo, /users/x/repos, /repositories)
    to one flat repos-FG row."""
    return {
        "repo_id": int(repo["id"]),
        "full_name": repo["full_name"],
        "owner_login": repo["owner"]["login"],
        "language": (repo.get("language") or "none").lower(),
        "topics": " ".join((repo.get("topics") or [])[:MAX_TOPICS]),
        "stars": int(repo.get("stargazers_count") or 0),
        "forks": int(repo.get("forks_count") or 0),
        "open_issues": int(repo.get("open_issues_count") or 0),
        "size_kb": int(repo.get("size") or 0),
        "created_at": _iso(repo.get("created_at")),
        "pushed_at": _iso(repo.get("pushed_at")),
        "is_fork": bool(repo.get("fork")),
        "archived": bool(repo.get("archived")),
        "description": (repo.get("description") or "")[:500],
        "captured_at": captured_at,
    }


def history(stars: list[tuple[datetime, int]], asof: datetime, k: int = HISTORY_LEN) -> list[int]:
    """The user's most recent k starred repo_ids strictly before asof, newest first."""
    past = [(ts, rid) for ts, rid in stars if ts < asof]
    past.sort(key=lambda x: x[0], reverse=True)
    return [rid for _, rid in past[:k]]


def user_scalars(
    history_rows: list[dict],
    own_rows: list[dict],
    asof: datetime,
) -> dict:
    """Scalar user-tower features from the repos the user starred before asof
    (history_rows) and the repos the user owns. Pure aggregates, fixed schema."""
    langs = Counter(r["language"] for r in history_rows if r["language"] != "none")
    total = sum(langs.values()) or 1
    top_lang, top_n = (langs.most_common(1) or [("none", 0)])[0]
    entropy = -sum((c / total) * math.log(c / total) for c in langs.values()) if langs else 0.0
    log_stars = [math.log1p(r["stars"]) for r in history_rows]
    ages = [
        (asof - r["created_at"]).days
        for r in history_rows
        if r.get("created_at") is not None
    ]
    own_langs = {r["language"] for r in own_rows if r["language"] != "none"}
    return {
        "n_stars": len(history_rows),
        "top_lang": top_lang,
        "top_lang_share": top_n / total,
        "lang_entropy": entropy,
        "mean_log_stars": sum(log_stars) / len(log_stars) if log_stars else 0.0,
        "median_repo_age_days": float(sorted(ages)[len(ages) // 2]) if ages else 0.0,
        "own_repo_count": len(own_rows),
        "own_lang_overlap": (
            sum(1 for r in history_rows if r["language"] in own_langs) / len(history_rows)
            if history_rows and own_langs
            else 0.0
        ),
    }


USER_DOC = {
    "n_stars": "stars in the user's visible history before asof",
    "top_lang": "most-starred language",
    "top_lang_share": "share of history in the top language",
    "lang_entropy": "entropy of the language distribution (taste breadth)",
    "mean_log_stars": "mean log1p stargazers of starred repos (obscurity taste: low = hipster)",
    "median_repo_age_days": "median age of starred repos at asof",
    "own_repo_count": "repos the user owns",
    "own_lang_overlap": "share of starred history matching a language the user writes",
}


# the numeric user-tower vector, one order, used verbatim by trainer and predictor
SCALARS = (
    "n_stars", "top_lang_share", "lang_entropy", "mean_log_stars",
    "median_repo_age_days", "own_repo_count", "own_lang_overlap",
)
HISTORY_CAP = 200  # scalars computed over at most this many recent stars, both sides


def scalars_vector(history_rows: list[dict], own_rows: list[dict], asof: datetime) -> list[float]:
    """The user tower's numeric input. history_rows/own_rows need language,
    stars, created_at. Counts go through log1p so a 5k-star train user and a
    200-star live pull live on the same scale."""
    s = user_scalars(history_rows[:HISTORY_CAP], own_rows, asof)
    return [
        math.log1p(s["n_stars"]),
        s["top_lang_share"],
        s["lang_entropy"],
        s["mean_log_stars"],
        math.log1p(max(s["median_repo_age_days"], 0.0)),
        math.log1p(s["own_repo_count"]),
        s["own_lang_overlap"],
    ]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)

"""F3: LLM fingerprints for the candidate repo corpus.

Take the top N repos by distinct in-corpus stargazers, snapshot their READMEs
(one capture date, so no text from after the temporal split leaks in), have
Claude emit a structured fingerprint per repo via the Batch API (structured
outputs, so no parse failures), embed pitch+topics with MiniLM, and write the
`repo_fingerprints` FG with an online KNN embedding index.

This is the cold-start path: a 12-star repo is retrievable by content before
CF ever sees it, and the chat's semantic surface.

Stages are resumable: readmes -> data/readmes.jsonl, fingerprints ->
data/fingerprints.jsonl. Deployed as a Hopsworks job (unstarred-pipeline env).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

import hopsworks

DATA = Path(__file__).resolve().parent / "data"
RAW = "https://raw.githubusercontent.com/{}/HEAD/{}"
README_NAMES = ("README.md", "readme.md", "README.rst", "README", "Readme.md")
README_CHARS = 6000
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_CHUNK = 5000  # requests per Anthropic batch (stays under the 256MB cap)

CATEGORIES = [
    "library", "framework", "cli-tool", "application", "dev-tool", "infra",
    "ml-ai", "data", "learning-resource", "awesome-list", "config-dotfiles",
    "game", "demo-toy", "research", "other",
]
MATURITIES = ["experimental", "active", "mature", "stale", "archived"]

FINGERPRINT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "maturity": {"type": "string", "enum": MATURITIES},
        "audience": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-5 short tags for who stars this, e.g. 'systems programmers', 'frontend developers', 'ml researchers', 'self-hosters'",
        },
        "pitch": {"type": "string", "description": "one plain sentence, what this repo is, max 140 chars"},
    },
    "required": ["category", "maturity", "audience", "pitch"],
    "additionalProperties": False,
}

PROMPT = """You are fingerprinting a GitHub repo for a recommender system.
Repo: {full_name}
Language: {language}
Topics: {topics}
Description: {description}
README (truncated):
{readme}

Emit the structured fingerprint. Be concrete about the audience: who actually
stars a repo like this. If the README is missing, judge from the metadata."""

FP_DOC = {
    "repo_id": "GitHub numeric repo id (primary key)",
    "category": "LLM-judged repo category",
    "maturity": "LLM-judged maturity from the README voice and metadata",
    "audience": "space-joined LLM tags for who stars this",
    "pitch": "one-sentence plain-language pitch",
    "readme_found": "a README was fetched at snapshot time",
    "pitch_embedding": "MiniLM embedding of pitch + topics + audience (online KNN index)",
    "snapshot_at": "README snapshot timestamp (event time)",
}


def candidates(top_n: int) -> pd.DataFrame:
    fs = hopsworks.login().get_feature_store()
    stars = fs.get_feature_group("star_events", version=1).read()
    repos = fs.get_feature_group("repos", version=1).read()
    # offline FGs append across job runs (PK dedup is online-only)
    repos = repos.sort_values("captured_at").drop_duplicates("repo_id", keep="last")
    counts = stars.groupby("repo_id")["user_login"].nunique().rename("corpus_stars")
    top = counts.sort_values(ascending=False).head(top_n).index
    return repos[repos["repo_id"].isin(top)].reset_index(drop=True)


def fetch_readmes(df: pd.DataFrame) -> None:
    out = DATA / "readmes.jsonl"
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {json.loads(l)["repo_id"] for l in fh if l.strip()}
    todo = df[~df["repo_id"].isin(done)]
    print(f"readmes: {len(done)} done, {len(todo)} to fetch", flush=True)

    def fetch(row) -> dict:
        s = requests.Session()
        for name in README_NAMES:
            try:
                r = s.get(RAW.format(row.full_name, name), timeout=20)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.text.strip():
                return {"repo_id": int(row.repo_id), "text": r.text[:README_CHARS]}
        return {"repo_id": int(row.repo_id), "text": ""}

    with out.open("a") as fh, ThreadPoolExecutor(max_workers=16) as ex:
        n_miss = 0
        for i, rec in enumerate(ex.map(fetch, todo.itertuples())):
            fh.write(json.dumps(rec) + "\n")
            n_miss += not rec["text"]
            if i % 1000 == 0:
                fh.flush()
                print(f"readmes {i}/{len(todo)}, misses {n_miss}", flush=True)
    print(f"readme misses total this run: {n_miss}", flush=True)


def fingerprints(df: pd.DataFrame, model: str) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    readmes = {}
    with (DATA / "readmes.jsonl").open() as fh:
        readmes = {json.loads(l)["repo_id"]: json.loads(l)["text"] for l in fh if l.strip()}

    out = DATA / "fingerprints.jsonl"
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {json.loads(l)["repo_id"] for l in fh if l.strip()}
    todo = df[~df["repo_id"].isin(done)].reset_index(drop=True)
    print(f"fingerprints: {len(done)} done, {len(todo)} to run on {model}", flush=True)
    if todo.empty:
        return

    hopsworks.login()
    client = anthropic.Anthropic(
        api_key=hopsworks.get_secrets_api().get_secret("ANTHROPIC_API_KEY").value
    )

    for start in range(0, len(todo), BATCH_CHUNK):
        chunk = todo.iloc[start : start + BATCH_CHUNK]
        reqs = [
            Request(
                custom_id=str(int(r.repo_id)),
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=1024,
                    output_config={"format": {"type": "json_schema", "schema": FINGERPRINT_SCHEMA}},
                    messages=[{
                        "role": "user",
                        "content": PROMPT.format(
                            full_name=r.full_name,
                            language=r.language,
                            topics=r.topics or "none",
                            description=r.description or "none",
                            readme=readmes.get(int(r.repo_id), "") or "(missing)",
                        ),
                    }],
                ),
            )
            for r in chunk.itertuples()
        ]
        for attempt in range(8):  # API 5xx outlives the SDK's 3 retries
            try:
                batch = client.messages.batches.create(requests=reqs)
                break
            except anthropic.APIStatusError as e:
                if e.status_code < 500 or attempt == 7:
                    raise
                print(f"batch create {e.status_code}, retry {attempt + 1}/8 in {60 * (attempt + 1)}s", flush=True)
                time.sleep(60 * (attempt + 1))
        print(f"batch {batch.id}: {len(reqs)} requests", flush=True)
        while True:  # bounded active polling: batches end within 24h by contract
            try:
                b = client.messages.batches.retrieve(batch.id)
            except anthropic.APIStatusError as e:
                if e.status_code < 500:
                    raise
                print(f"batch poll {e.status_code}, retrying", flush=True)
                time.sleep(60)
                continue
            if b.processing_status == "ended":
                break
            print(f"  {b.processing_status}: {b.request_counts.processing} processing", flush=True)
            time.sleep(60)
        n_err = 0
        with out.open("a") as fh:
            for result in client.messages.batches.results(batch.id):
                rid = int(result.custom_id)
                if result.result.type == "succeeded":
                    msg = result.result.message
                    if msg.stop_reason == "refusal":
                        n_err += 1
                        continue
                    text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                    fp = json.loads(text)
                    fp["repo_id"] = rid
                    fh.write(json.dumps(fp) + "\n")
                else:
                    n_err += 1
        print(f"batch done, {n_err} errored/refused (will retry on next run)", flush=True)


def write_fg(df: pd.DataFrame) -> None:
    from sentence_transformers import SentenceTransformer
    from hsfs import embedding

    fps = {}
    with (DATA / "fingerprints.jsonl").open() as fh:
        fps = {json.loads(l)["repo_id"]: json.loads(l) for l in fh if l.strip()}
    readme_found = {}
    with (DATA / "readmes.jsonl").open() as fh:
        readme_found = {json.loads(l)["repo_id"]: bool(json.loads(l)["text"]) for l in fh if l.strip()}

    snapshot_at = pd.Timestamp.utcnow()
    rows = []
    for r in df.itertuples():
        rid = int(r.repo_id)
        fp = fps.get(rid)
        if not fp:
            continue
        rows.append({
            "repo_id": rid,
            "category": fp["category"],
            "maturity": fp["maturity"],
            "audience": " ".join(fp["audience"][:5]),
            "pitch": fp["pitch"][:200],
            "readme_found": readme_found.get(rid, False),
            "embed_text": f'{fp["pitch"]} | {r.topics} | {" ".join(fp["audience"][:5])}',
            "snapshot_at": snapshot_at,
        })
    out = pd.DataFrame(rows)
    print(f"embedding {len(out)} fingerprints with {EMBED_MODEL}", flush=True)
    st = SentenceTransformer(EMBED_MODEL)
    vecs = st.encode(out["embed_text"].tolist(), batch_size=256, show_progress_bar=False)
    out["pitch_embedding"] = [v.tolist() for v in vecs]
    out = out.drop(columns=["embed_text"])

    fs = hopsworks.login().get_feature_store()
    ei = embedding.EmbeddingIndex()
    ei.add_embedding(name="pitch_embedding", dimension=EMBED_DIM)
    fg = fs.get_or_create_feature_group(
        name="repo_fingerprints",
        version=1,
        description="LLM README fingerprints per candidate repo, with a MiniLM KNN index (cold-start + chat semantic surface)",
        primary_key=["repo_id"],
        event_time="snapshot_at",
        online_enabled=True,
        embedding_index=ei,
    )
    fg.insert(out)
    for name, desc in FP_DOC.items():
        fg.update_feature_description(name, desc)
    print(f"wrote {len(out)} fingerprint rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=40000)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--stage", choices=["readmes", "fingerprints", "fg", "all"], default="all")
    args = ap.parse_args()

    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)
    DATA.mkdir(parents=True, exist_ok=True)

    df = candidates(args.top_n)
    print(f"candidate corpus: {len(df)} repos", flush=True)
    if args.stage in ("readmes", "all"):
        fetch_readmes(df)
    if args.stage in ("fingerprints", "all"):
        fingerprints(df, args.model)
    if args.stage in ("fg", "all"):
        write_fg(df)


if __name__ == "__main__":
    main()

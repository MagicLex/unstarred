"""unstarred: the review app. Shelf, dossier, chat.

Server-rendered FastAPI (no SPA): the shelf is in the initial payload; JS
hydrates the dossier stream and the librarian tab over websockets (the
platform proxy buffers plain HTTP streaming).

The rank comes from the model, always: the shelf is a KNN over the trained
item space with the user vector from the deployment; the chat compiles
utterances into deterministic tool calls over the same indexes and never
invents a repo. The LLM explains and steers, it does not pick.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse

import hopsworks

DOSSIER_MODEL = "claude-sonnet-5"
ASK_MODEL = "claude-sonnet-5"
SHELF_K = 60
SHELF_FIRST = 20  # cards visible before "five more"
PROFILE_TTL = 900
ALIVE_TTL = 6 * 3600  # repos die/rename after capture; recheck links this often

state: dict = {}


def boot() -> None:
    import anthropic
    from sentence_transformers import SentenceTransformer

    project = hopsworks.login()
    fs = project.get_feature_store()
    state["emb_fg"] = fs.get_feature_group("repo_embeddings", version=1)
    state["fp_fg"] = fs.get_feature_group("repo_fingerprints", version=1)
    state["deployment"] = project.get_model_serving().get_deployment("unstarredquery")

    mr = project.get_model_registry()
    model = max(mr.get_models("unstarred"), key=lambda m: m.version)
    state["model_version"] = model.version
    # Hopsworks sanitizes metric keys on register (recall_at_50 -> recall_at__50);
    # collapse runs of underscores so the trainer's names resolve at read time.
    state["metrics"] = {re.sub(r"_+", "_", k): v for k, v in (model.training_metrics or {}).items()}

    # display frame + anchor vectors in memory: 300k x 64 floats is ~80MB and
    # buys instant anchor lookup + name search; the online KNN stays the
    # retrieval path for shelf and chat
    df = state["emb_fg"].select_all().read()
    # offline FGs append across job runs (PK dedup is online-only): keep the
    # newest embedding per repo so anchors match the online KNN's vectors
    df = df.sort_values("embedded_at").drop_duplicates("repo_id", keep="last").reset_index(drop=True)
    df["full_name_lc"] = df["full_name"].str.lower()
    state["vecs"] = np.array(df["item_vector"].tolist(), dtype="float32")
    state["repos"] = df.drop(columns=["item_vector"]).reset_index(drop=True)
    state["row_by_id"] = {int(r): i for i, r in enumerate(df["repo_id"])}
    state["id_by_name"] = {n: int(i) for n, i in zip(df["full_name_lc"], df["repo_id"])}

    state["alive"] = {}
    state["st"] = SentenceTransformer("all-MiniLM-L6-v2")
    state["claude"] = anthropic.Anthropic(
        api_key=hopsworks.get_secrets_api().get_secret("ANTHROPIC_API_KEY").value
    )
    state["profiles"] = {}
    print(f"boot done: {len(df)} candidates, model v{model.version}", flush=True)


def get_profile(login: str) -> dict:
    login = login.strip().lstrip("@").lower()
    hit = state["profiles"].get(login)
    if hit and time.time() - hit[0] < PROFILE_TTL:
        return hit[1]
    res = state["deployment"].predict(inputs=[{"login": login}])
    pred = res["predictions"][0]
    state["profiles"][login] = (time.time(), pred)
    return pred


def _rows(fg, hits) -> list[dict]:
    """find_neighbors returns (score, [values in schema order])."""
    names = [f.name for f in fg.features]
    out = []
    for score, vals in hits:
        rec = dict(zip(names, vals))
        rec["score"] = float(score)
        out.append(rec)
    return out


def alive_only(rows: list[dict]) -> list[dict]:
    """Drop repos that 404 on github.com (deleted/renamed since capture)."""
    now = time.time()
    cache = state["alive"]
    todo = [r["full_name"] for r in rows
            if r["full_name"] not in cache or now - cache[r["full_name"]][0] > ALIVE_TTL]

    def check(fn: str):
        try:
            resp = requests.head(f"https://github.com/{fn}", timeout=5, allow_redirects=True)
            return fn, resp.status_code < 400
        except requests.RequestException:
            return fn, True  # network doubt: keep the card rather than blank the shelf
    if todo:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for fn, ok in ex.map(check, todo):
                cache[fn] = (now, ok)
    return [r for r in rows if cache[r["full_name"]][1]]


def shelf_for(pred: dict, k: int = SHELF_K) -> list[dict]:
    starred = set(pred.get("starred_ids", []))
    hits = _rows(state["emb_fg"], state["emb_fg"].find_neighbors(pred["user_vector"], k=k + len(starred) + 40))
    rows = []
    for rec in hits:
        rid = int(rec["repo_id"])
        if rid in starred:
            continue
        rows.append({
            "repo_id": rid, "full_name": rec["full_name"], "language": rec["language"],
            "stars": int(rec["stars"]), "category": rec.get("category") or "",
            "pitch": rec.get("pitch") or "", "score": rec["score"],
        })
    return alive_only(rows)[:k]


def text_search(query: str, k: int = 40) -> list[dict]:
    vec = state["st"].encode([query])[0].tolist()
    return _rows(state["fp_fg"], state["fp_fg"].find_neighbors(vec, k=k))


# ---------------------------------------------------------------- chat tools

TOOLS = [
    {"name": "search_repos",
     "description": "Semantic search over the fingerprinted corpus (~top repos by "
                    "corpus stars). Finds repos by what they ARE ('tui database "
                    "client', 'rust game engine'). Optional filters applied after "
                    "the search.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "language": {"type": "string"},
         "category": {"type": "string"},
         "max_stars": {"type": "integer", "description": "hidden-gem hunting: only repos below this"},
         "min_stars": {"type": "integer"}},
         "required": ["query"]}},
    {"name": "similar_repos",
     "description": "KNN in the TRAINED taste space around an anchor repo "
                    "(owner/name). 'People who star X star Y', not text similarity.",
     "input_schema": {"type": "object", "properties": {
         "full_name": {"type": "string"}}, "required": ["full_name"]}},
    {"name": "shelf_for_user",
     "description": "The model's current shelf for the GitHub user on screen (or "
                    "any public login): top recommendations from their live stars.",
     "input_schema": {"type": "object", "properties": {
         "login": {"type": "string"}}, "required": ["login"]}},
    {"name": "repo_lookup",
     "description": "Exact metadata for one repo by owner/name substring: stars, "
                    "language, category, pitch, whether it is in the corpus.",
     "input_schema": {"type": "object", "properties": {
         "name_contains": {"type": "string"}}, "required": ["name_contains"]}},
    {"name": "corpus_stats",
     "description": "What this recommender is: corpus size, model version, held-out "
                    "metrics vs the popularity baseline.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
]


def _fmt(rows: list[dict], n: int = 12) -> str:
    out = []
    for r in rows[:n]:
        bits = [r.get("full_name", "?"), f'{r.get("stars", "?")}★', r.get("language") or ""]
        if r.get("category"):
            bits.append(r["category"])
        if r.get("pitch"):
            bits.append(r["pitch"][:100])
        out.append(" | ".join(str(b) for b in bits if b))
    return "\n".join(out) if out else "nothing found"


def exec_tool(name: str, args: dict, page_login: str | None) -> str:
    repos = state["repos"]
    if name == "search_repos":
        rows = text_search(args["query"], k=60)
        ids = [int(r["repo_id"]) for r in rows]
        sub = repos[repos["repo_id"].isin(ids)].copy()
        order = {rid: i for i, rid in enumerate(ids)}
        sub["o"] = sub["repo_id"].map(order)
        sub = sub.sort_values("o")
        if args.get("language"):
            sub = sub[sub["language"].str.lower() == args["language"].lower()]
        if args.get("category"):
            sub = sub[sub["category"] == args["category"]]
        if args.get("max_stars"):
            sub = sub[sub["stars"] <= args["max_stars"]]
        if args.get("min_stars"):
            sub = sub[sub["stars"] >= args["min_stars"]]
        return _fmt(sub.to_dict("records"))
    if name == "similar_repos":
        rid = state["id_by_name"].get(args["full_name"].strip().lower())
        if rid is None:
            return f'{args["full_name"]} is not in the corpus'
        vec = state["vecs"][state["row_by_id"][rid]].tolist()
        rows = [r for r in _rows(state["emb_fg"], state["emb_fg"].find_neighbors(vec, k=15))
                if int(r["repo_id"]) != rid]
        return _fmt(rows)
    if name == "shelf_for_user":
        login = args.get("login") or page_login
        if not login:
            return "no user on screen; ask for a login"
        pred = get_profile(login)
        if "error" in pred:
            return f'{login}: {pred["error"]}'
        return _fmt(shelf_for(pred, k=15))
    if name == "repo_lookup":
        q = args["name_contains"].strip().lower()
        sub = repos[repos["full_name_lc"].str.contains(q, regex=False)].nlargest(8, "stars")
        return _fmt(sub.to_dict("records")) if len(sub) else f"no corpus repo matches '{q}'"
    if name == "corpus_stats":
        m = state["metrics"]
        return (f'{len(repos)} candidate repos, model unstarred v{state["model_version"]}. '
                f'Held-out future-star recall@50: {m.get("recall_at_50", "?")} vs popularity '
                f'{m.get("pop_recall_at_50", "?")}; MRR@100 {m.get("mrr_at_100", "?")} vs '
                f'{m.get("pop_mrr_at_100", "?")}. Metric: predicting stars users actually '
                f'added after the temporal split.')
    return f"unknown tool {name}"


ASK_SYSTEM = """You are the librarian behind "unstarred", a GitHub repo recommender: \
a two-tower model trained on public star histories, plus these deterministic tools \
over its indexes. Every repo you name comes from a tool result: never invent a repo, \
a star count or a metric. If a tool returns nothing, say so and try a different query \
or filter.

You have two different searches and should pick deliberately: search_repos finds \
repos by what they ARE (text); similar_repos and shelf_for_user rank by taste \
(who-stars-what). For "find me a tool for X" start with search_repos; for "more \
like this" or "for me" use the taste tools; combining both in one answer is often \
the best move. Use max_stars when the user wants hidden gems: the whole point of \
this recommender is surfacing what trending never shows.

Recommendations are resemblance to starring behavior, never a quality verdict. Say \
what a repo is from its pitch, not from imagination.

House voice: direct, concrete, no em dashes, no "it's not X, it's Y", no closing \
flourish. Under 160 words unless asked for depth. Plain prose; name repos as \
owner/name."""

MAX_TURNS = 6

_REPO_TOKEN = re.compile(r"\b[\w.-]+/[\w.-]+\b")


def cards_for_answer(text: str, cap: int = 8) -> list[dict]:
    """The corpus rows behind every owner/name the librarian actually said."""
    repos, seen, out = state["repos"], set(), []
    for tok in _REPO_TOKEN.findall(text):
        lc = tok.lower()
        rid = state["id_by_name"].get(lc)
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        r = repos.iloc[state["row_by_id"][rid]]
        out.append({"full_name": r["full_name"], "stars": int(r["stars"]),
                    "language": r["language"], "pitch": r["pitch"] or ""})
        if len(out) >= cap:
            break
    return out


async def run_ask_stream(ws: WebSocket, question: str, history: list, page_login: str | None):
    client = state["claude"]
    msgs = []
    for q, a in history[-6:]:
        msgs.append({"role": "user", "content": str(q)[:500]})
        msgs.append({"role": "assistant", "content": str(a)[:1500]})
    lead = f"[on screen: shelf for {page_login}]\n" if page_login else ""
    msgs.append({"role": "user", "content": lead + question[:500]})

    loop = asyncio.get_event_loop()
    for _ in range(MAX_TURNS):
        def call():
            with client.messages.stream(model=ASK_MODEL, max_tokens=800,
                                        system=ASK_SYSTEM, tools=TOOLS, messages=msgs) as s:
                chunks = []
                for delta in s.text_stream:
                    chunks.append(delta)
                    asyncio.run_coroutine_threadsafe(
                        ws.send_json({"t": "tok", "d": delta}), loop).result()
                return s.get_final_message(), "".join(chunks)
        resp, text = await asyncio.to_thread(call)
        if resp.stop_reason != "tool_use":
            cards = cards_for_answer(text)
            if cards:
                await ws.send_json({"t": "cards", "d": cards})
            return
        msgs.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                await ws.send_json({"t": "tool", "d": b.name})
                try:
                    out = await asyncio.to_thread(exec_tool, b.name, dict(b.input or {}), page_login)
                except Exception as e:
                    out = f"tool error: {e}"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)[:5000]})
        msgs.append({"role": "user", "content": results})
    await ws.send_json({"t": "tok", "d": "\n(tool budget exhausted; ask narrower)"})


DOSSIER_SYSTEM = """You write the taste dossier for "unstarred". Input: a GitHub \
user's recent public stars and own repos, pulled live. Output: what their GitHub \
actually says about them, in 90-140 words of plain prose. Be specific and a little \
sharp: name the pattern (the languages, the kind of tool they hoard, what they build \
vs what they admire, the gap between the two). Ground every claim in the listed \
repos; name at most four. No flattery, no em dashes, no "it's not X, it's Y", no \
markdown. End on the fact, not a flourish."""


async def stream_dossier(ws: WebSocket, pred: dict):
    profile = pred["profile"]
    content = json.dumps({
        "login": pred["login"],
        "top_languages": profile["top_languages"],
        "n_stars_pulled": profile["n_stars_pulled"],
        "recent_stars": profile["recent_stars"],
        "own_repos": profile["own_repos"],
    })
    client = state["claude"]
    loop = asyncio.get_event_loop()

    def call():
        with client.messages.stream(model=DOSSIER_MODEL, max_tokens=400,
                                    system=DOSSIER_SYSTEM,
                                    messages=[{"role": "user", "content": content}]) as s:
            for delta in s.text_stream:
                asyncio.run_coroutine_threadsafe(ws.send_json({"t": "tok", "d": delta}), loop).result()
    await asyncio.to_thread(call)


# ------------------------------------------------------------------- pages

# A dark instrument, not a GitHub clone: warm ink ground, one star-gold accent,
# and mono for every number so the readout reads like a machine showing its work.
CSS = """
:root{--bg:#0a0a0d;--bg2:#0f0f14;--panel:#14141b;--panel2:#1b1b24;
--ink:#ece7dd;--dim:#8f8f9d;--faint:#5b5b67;
--line:#26262f;--line2:#34343f;
--gold:#f5c451;--gold-soft:#f5c45122;--gold-line:#f5c45140;
--bad:#f0708a;--good:#7ee0b8;
--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:
radial-gradient(1100px 500px at 15% -10%,#16121f 0%,transparent 60%),var(--bg);
color:var(--ink);font:15px/1.6 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--gold);text-decoration:none}a:hover{text-decoration:underline}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.wrap{max-width:1080px;margin:0 auto;padding:18px 22px 90px}
h1{font-size:20px;margin:0;font-weight:650;letter-spacing:-.01em}
h1 a{color:var(--ink)}.ustar{color:var(--gold);font-weight:400}
.crumb{color:var(--faint)}
.sub{color:var(--dim);max-width:660px;font-size:14.5px}

/* home hero */
.hero{margin:34px 0 8px}
.hero h2{font-size:clamp(28px,4.5vw,44px);line-height:1.08;font-weight:680;
letter-spacing:-.02em;margin:0 0 14px;max-width:15ch}
.hero h2 em{font-style:normal;color:var(--gold)}
.lede{font-size:16px;color:var(--dim);max-width:640px;margin:0 0 26px}
form.login{display:flex;gap:10px;margin:0 0 8px;max-width:480px}
input[type=text]{flex:1;background:var(--bg2);border:1px solid var(--line2);
border-radius:9px;padding:11px 14px;color:var(--ink);font-size:15px}
input[type=text]:focus{outline:none;border-color:var(--gold);box-shadow:0 0 0 3px var(--gold-soft)}
button{background:var(--panel2);color:var(--ink);border:1px solid var(--line2);border-radius:9px;
padding:11px 18px;font-weight:600;font-size:14px;cursor:pointer;font-family:var(--sans)}
button:hover{border-color:var(--faint)}
button.cta{background:var(--gold);color:#1a1405;border-color:transparent}
button.cta:hover{background:#ffd469}

/* the flex: model vs trending */
.flex{background:linear-gradient(180deg,var(--panel) 0%,#111119 100%);
border:1px solid var(--line);border-radius:14px;padding:24px 26px;margin:36px 0}
.flex .big{font-family:var(--mono);font-size:clamp(40px,7vw,68px);font-weight:700;
color:var(--gold);line-height:1;letter-spacing:-.03em}
.flex .cap{color:var(--dim);max-width:600px;margin:6px 0 22px;font-size:14.5px}
.cmp{display:flex;flex-direction:column;gap:12px;max-width:640px}
.cmprow{display:grid;grid-template-columns:118px 1fr 70px;align-items:center;gap:14px}
.cmprow .lbl{font-size:12.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
.track{height:12px;background:#00000055;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.track i{display:block;height:100%;border-radius:6px}
.track i.model{background:linear-gradient(90deg,#f5c451,#ffdd7a)}
.track i.pop{background:var(--faint)}
.cmprow .val{font-family:var(--mono);font-size:14px;text-align:right;font-variant-numeric:tabular-nums}
.cmprow .val.model{color:var(--gold);font-weight:600}.cmprow .val.pop{color:var(--dim)}
.flex .foot{color:var(--faint);font-size:12.5px;margin-top:18px;font-family:var(--mono)}

/* machinery strip */
.mach{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:30px 0}
@media(max-width:820px){.mach{grid-template-columns:1fr}}
.step{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;position:relative}
.step .k{font-family:var(--mono);font-size:11px;color:var(--gold);letter-spacing:.08em}
.step h4{margin:8px 0 6px;font-size:14.5px;font-weight:640}
.step p{margin:0;color:var(--dim);font-size:13px;line-height:1.5}
.step .badge{display:inline-block;margin-top:10px;font-family:var(--mono);font-size:10.5px;
letter-spacing:.06em;color:var(--gold);background:var(--gold-soft);border:1px solid var(--gold-line);
border-radius:6px;padding:2px 8px}

/* shelf */
.shelfhead{display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap;margin:8px 0 4px}
.readout{font-family:var(--mono);font-size:12px;color:var(--faint);display:flex;gap:16px;flex-wrap:wrap}
.readout b{color:var(--gold);font-weight:600}
.cols{display:grid;grid-template-columns:1fr 300px;gap:26px;margin-top:22px}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:0 0 14px;
overflow:hidden;transition:border-color .12s,transform .12s}
.card:hover{border-color:var(--line2);transform:translateY(-2px)}
.card .og{width:100%;aspect-ratio:2/1;object-fit:cover;display:block;background:#06060a;border-bottom:1px solid var(--line)}
.card .body{padding:12px 15px 0}
.card .nm{font-weight:640;font-size:15px;letter-spacing:-.01em}
.card .meta{color:var(--dim);font-size:12px;margin-top:5px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-family:var(--mono)}
.ldot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;
vertical-align:-1px;border:1px solid #ffffff22}
.tag{color:var(--faint);border:1px solid var(--line2);border-radius:5px;padding:0 6px;font-size:11px}
.card .pitch{font-size:13px;margin-top:9px;color:#c7c2b8;line-height:1.5}
.scorow{display:flex;align-items:center;gap:9px;margin-top:12px}
.scorow .lab{font-family:var(--mono);font-size:10px;color:var(--faint);letter-spacing:.05em}
.bar{flex:1;height:5px;background:#00000055;border:1px solid var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#f5c451,#ffdd7a)}
.scorow .sc{color:var(--gold);font-size:12px;font-weight:600;min-width:36px;text-align:right;
font-family:var(--mono);font-variant-numeric:tabular-nums}
#more{display:block;margin:22px auto 0}

/* side rail */
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:16px}
.panel h3{margin:0 0 10px;font-family:var(--mono);font-size:11px;letter-spacing:.09em;
text-transform:uppercase;color:var(--dim);display:flex;align-items:center;gap:8px}
.live{width:7px;height:7px;border-radius:50%;background:var(--gold);box-shadow:0 0 0 0 var(--gold-soft);
animation:pulse 1.6s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 #f5c45166}70%{box-shadow:0 0 0 7px #f5c45100}100%{box-shadow:0 0 0 0 #f5c45100}}
#dossier{white-space:pre-wrap;font-size:13.5px;min-height:90px;color:#d8d3c8;line-height:1.6}
#dossier:empty::before{content:"reading your stars";color:var(--faint);font-family:var(--mono);font-size:12px}
#dossier:empty::after{content:"…";color:var(--faint)}
.langs{margin-bottom:12px}
.langs span{display:inline-block;background:var(--gold-soft);border:1px solid var(--gold-line);border-radius:20px;
padding:1px 10px;margin:2px 4px 2px 0;font-size:12px;color:var(--gold);font-family:var(--mono)}
.rstat{font-family:var(--mono);font-size:12px;color:var(--dim);line-height:1.9}
.rstat b{color:var(--gold)}

/* tabs */
.tabs{display:flex;gap:6px;margin-bottom:22px;border-bottom:1px solid var(--line)}
.tabs button{background:none;color:var(--dim);border:0;border-bottom:2px solid transparent;
border-radius:0;padding:9px 4px;margin-right:14px;font-weight:500;font-size:14px;cursor:pointer}
.tabs button:hover{color:var(--ink)}
.tabs button.active{color:var(--ink);border-bottom-color:var(--gold)}

/* librarian: full-height column, input pinned at the bottom like a chat app */
#librarian{max-width:740px;display:flex;flex-direction:column;
height:calc(100vh - 210px);min-height:360px}
.libintro{color:var(--dim);font-size:13.5px;margin:0 0 14px}
#log{flex:1;overflow-y:auto;padding:4px 2px;font-size:14px;
display:flex;flex-direction:column;gap:12px}
#log .q{color:var(--ink);background:var(--bg2);border:1px solid var(--line);border-radius:14px;
padding:8px 14px;align-self:flex-end;max-width:82%}
#log .a{white-space:pre-wrap;color:#d8d3c8;line-height:1.7;align-self:stretch}
#log .t{color:var(--faint);font-family:var(--mono);font-size:11.5px;
align-self:flex-start;background:var(--bg2);border:1px solid var(--line);border-radius:6px;padding:2px 9px}
.rchip{display:inline-flex;align-items:center;gap:6px;background:var(--bg2);
border:1px solid var(--line);border-radius:8px;padding:1px 9px;margin:1px 0;
font-size:12.5px;font-weight:600;vertical-align:-2px;white-space:nowrap}
.rchip:hover{border-color:var(--gold);text-decoration:none}
.rchip .rc-stars{color:var(--dim);font-weight:400;font-size:11.5px}
#askform{display:flex;gap:8px;margin-top:12px}
#askform input{flex:1;border-radius:22px;padding-left:16px}
#askform button{border-radius:22px}
.err{color:var(--bad)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 18px}
.chips button{border-radius:20px;padding:5px 14px;font-weight:500;font-size:12.5px;color:var(--dim);
font-family:var(--mono)}
.chips button.active{color:var(--gold);border-color:var(--gold-line);background:var(--gold-soft)}
footer{margin-top:52px;color:var(--faint);font-size:12px;border-top:1px solid var(--line);padding-top:18px;line-height:1.7}
"""

# GitHub's own linguist colors for the dot; dim grey for the long tail
LANG_COLORS = {
    "Python": "#3572A5", "JavaScript": "#f1e05a", "TypeScript": "#3178c6",
    "Rust": "#dea584", "Go": "#00ADD8", "C++": "#f34b7d", "C": "#555555",
    "Java": "#b07219", "Shell": "#89e051", "Ruby": "#701516", "HTML": "#e34c26",
    "CSS": "#563d7c", "Jupyter Notebook": "#DA5B0B", "Swift": "#F05138",
    "Kotlin": "#A97BFF", "PHP": "#4F5D95", "C#": "#178600", "Zig": "#ec915c",
    "Lua": "#000080", "Dart": "#00B4AB", "Elixir": "#6e4a7e", "Haskell": "#5e5086",
}


def lang_dot(language: str) -> str:
    if not language or language == "none":
        return ""
    color = LANG_COLORS.get(language, "#8b949e")
    return (f'<span><span class="ldot" style="background:{color}"></span>'
            f'{html.escape(language)}</span>')

ASK_JS = """
const log=document.getElementById('log');
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('active',x===b));
  document.getElementById('main').hidden=b.dataset.t!=='main';
  document.getElementById('librarian').hidden=b.dataset.t!=='librarian';
  if(b.dataset.t==='librarian')document.getElementById('q').focus();
});
let hist=[],ws=null;
document.getElementById('askform').onsubmit=(e)=>{
  e.preventDefault();
  const inp=document.getElementById('q'),q=inp.value.trim();if(!q)return;inp.value='';
  const qd=document.createElement('div');qd.className='q';qd.textContent=q;log.appendChild(qd);
  const ad=document.createElement('div');ad.className='a';log.appendChild(ad);
  const base=location.pathname.replace(/\\/u\\/[^/]+$/,'').replace(/\\/$/,'');
  ws=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+base+'/ws/ask');
  ws.onopen=()=>ws.send(JSON.stringify({q:q,history:hist,login:PAGE_LOGIN}));
  ws.onmessage=(m)=>{const d=JSON.parse(m.data);
    if(d.t==='tok'){ad.textContent+=d.d;log.scrollTop=log.scrollHeight}
    else if(d.t==='tool'){const t=document.createElement('div');t.className='t';
      t.textContent='consulting '+d.d;log.insertBefore(t,ad)}
    else if(d.t==='cards'){
      const by={};d.d.forEach(r=>by[r.full_name.toLowerCase()]=r);
      const names=d.d.map(r=>r.full_name.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&'));
      const rx=new RegExp('('+names.join('|')+')','gi');
      const txt=ad.textContent.replace(/\\*\\*/g,'');
      ad.textContent='';
      txt.split(rx).forEach(part=>{
        const r=by[(part||'').toLowerCase()];
        if(r){const a=document.createElement('a');a.className='rchip';
          a.href='https://github.com/'+r.full_name;a.target='_blank';a.rel='noopener';
          a.title=r.pitch||'';
          const dt=document.createElement('span');dt.className='ldot';
          dt.style.background=LANG_COLORS[r.language]||'#8b949e';a.appendChild(dt);
          a.appendChild(document.createTextNode(r.full_name));
          const st=document.createElement('span');st.className='rc-stars';
          st.textContent='\\u2606 '+r.stars.toLocaleString();a.appendChild(st);
          ad.appendChild(a);}
        else if(part)ad.appendChild(document.createTextNode(part));});
      log.scrollTop=log.scrollHeight}
    else if(d.t==='end'){hist.push([q,ad.textContent]);ws.close()}};
};
"""


def page(title: str, body: str, page_login: str = "") -> HTMLResponse:
    main_tab = "shelf" if page_login else "home"
    return HTMLResponse(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<div class="wrap">
<nav class="tabs"><button class="active" data-t="main">{main_tab}</button>
<button data-t="librarian">ask the librarian</button></nav>
<section id="main">{body}</section>
<section id="librarian" hidden>
<p class="libintro">Ask in plain english. The librarian compiles it into deterministic KNN
queries over the model's indexes and streams the answer. Every repo it names comes from a
tool call, never invented. Try "a tui database client under 1k stars" or "more like tqdm".</p>
<div id="log"></div>
<form id="askform"><input type="text" id="q" placeholder="the perfect repo for…" autocomplete="off">
<button>ask</button></form></section>
<footer>unstarred #012 · a two-tower recommender with an LLM on top · rank is resemblance
to starring behavior, never a quality verdict · <a href="https://github.com/MagicLex/unstarred">source</a></footer>
</div>
<script>const PAGE_LOGIN={json.dumps(page_login or None)};const LANG_COLORS={json.dumps(LANG_COLORS)};{ASK_JS}</script>
</body></html>""")


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    n = len(state["repos"])
    m = state["metrics"]
    r50 = float(m.get("recall_at_50", 0))
    p50 = float(m.get("pop_recall_at_50", 0))
    mult = r50 / p50 if p50 else 0
    n_users = int(float(m.get("n_test_users", 0)))
    n_stars = int(float(m.get("n_test_stars", 0)))
    pop_w = max(3, round(100 * p50 / r50)) if r50 else 3  # trending bar, relative to the model bar
    body = f"""
<div class="hero">
<h2>Which repos would you have <em>starred already</em>, if you had seen them?</h2>
<p class="lede">A two-tower model trained on {n:,} repos' worth of public star histories reads
your stars and your own repos, then ranks the ones you have not seen. Type a GitHub
username and watch it read you.</p>
<form class="login" action="go" method="get">
<input type="text" name="login" placeholder="your GitHub username" required autofocus>
<button class="cta">read my stars</button></form>
</div>

<div class="flex">
<div class="big">{mult:.0f}&times;</div>
<p class="cap">Held out by time, the model puts a repo you would later star in its top 50
about <b>{mult:.0f} times more often</b> than showing everyone the same trending list does.
Same corpus, same held-out future, one blind popularity baseline.</p>
<div class="cmp">
<div class="cmprow"><span class="lbl">the model</span>
<div class="track"><i class="model" style="width:100%"></i></div>
<span class="val model">{r50 * 100:.1f}%</span></div>
<div class="cmprow"><span class="lbl">trending</span>
<div class="track"><i class="pop" style="width:{pop_w}%"></i></div>
<span class="val pop">{p50 * 100:.2f}%</span></div>
</div>
<div class="foot">recall@50 on {n_users:,} held-out users · {n_stars:,} future stars · model v{state["model_version"]}</div>
</div>

<div class="mach">
<div class="step"><div class="k">RETRIEVAL</div><h4>Two towers, one taste space</h4>
<p>Your stars become a vector. Every repo becomes a vector. The shelf is a nearest-neighbour
lookup, not a hand-tuned rule.</p></div>
<div class="step"><div class="k">LANGUAGE</div><h4>The LLM writes, never picks</h4>
<p>It fingerprints READMEs for cold-start, writes your taste dossier, and turns your words
into queries. It never chooses a repo for you.</p>
<span class="badge">LLM ∉ the ranking</span></div>
<div class="step"><div class="k">ASK</div><h4>A librarian you can talk to</h4>
<p>Plain English compiles into deterministic KNN calls over the same indexes. Every repo it
names comes from a tool result.</p></div>
</div>"""
    return page("unstarred", body)


@app.get("/go")
def go(login: str = ""):
    return RedirectResponse(f"u/{login.strip().lstrip('@')}", status_code=302)


@app.get("/u/{login}", response_class=HTMLResponse)
def shelf_page(login: str):
    try:
        pred = get_profile(login)
    except Exception as e:
        return page("unstarred", f'<h1><a href=".">unstarred</a></h1><p class="err">deployment error: {html.escape(str(e))}</p>')
    if "error" in pred:
        return page("unstarred", f'<h1><a href="../">unstarred</a></h1><p class="err">{html.escape(pred["error"])}</p>')
    rows = shelf_for(pred)
    top = max((r["score"] for r in rows), default=1.0) or 1.0
    cards = "".join(
        f"""<div class="card" data-s="{r["stars"]}"{' hidden' if i >= SHELF_FIRST else ''}>
<a href="https://github.com/{html.escape(r["full_name"])}" target="_blank" rel="noopener">
<img class="og" loading="lazy" alt="" src="https://opengraph.githubassets.com/1/{html.escape(r["full_name"])}"></a>
<div class="body">
<div class="nm"><a href="https://github.com/{html.escape(r["full_name"])}"
target="_blank" rel="noopener">{html.escape(r["full_name"])}</a></div>
<div class="meta"><span>☆ {r["stars"]:,}</span>{lang_dot(r["language"])}{f'<span class="tag">{html.escape(r["category"])}</span>' if r["category"] else ""}</div>
<div class="pitch">{html.escape(r["pitch"])}</div>
<div class="scorow"><span class="lab">FIT</span><div class="bar"><i style="width:{max(6, int(100 * r["score"] / top))}%"></i></div>
<span class="sc">{int(100 * r["score"] / top)}%</span></div></div></div>"""
        for i, r in enumerate(rows))
    prof = pred["profile"]
    langs = "".join(f"<span>{html.escape(x)}</span>" for x in prof["top_languages"])
    m = state["metrics"]
    r50 = float(m.get("recall_at_50", 0)) * 100
    p50 = float(m.get("pop_recall_at_50", 0)) * 100
    body = f"""
<div class="shelfhead">
<h1><a href="../"><span class="ustar">☆</span> unstarred</a> <span class="crumb">/ {html.escape(pred["login"])}</span></h1>
<div class="readout"><span>KNN in trained taste space</span>
<span>from <b>{prof["n_stars_pulled"]}</b> live stars</span>
<span>model <b>v{state["model_version"]}</b></span></div>
</div>
<p class="sub">Already-starred repos filtered out, dead links checked live. FIT is resemblance
to your starring behaviour, relative to your top hit. Not a quality verdict.</p>
<div class="cols"><div>
<div class="chips" id="chips"><button class="active" data-lo="0" data-hi="">all</button>
<button data-lo="10000" data-hi="">10k+ ☆</button>
<button data-lo="1000" data-hi="10000">1k to 10k</button>
<button data-lo="0" data-hi="1000">under 1k · the gems</button></div>
<div class="grid">{cards}</div>
<button id="more">give me five more</button></div>
<div>
<div class="panel"><h3><span class="live"></span> the dossier</h3><div class="langs">{langs}</div>
<div id="dossier"></div></div>
<div class="panel"><h3>the readout</h3><div class="rstat">
recall@50 &nbsp;<b>{r50:.1f}%</b><br>trending &nbsp;<b>{p50:.2f}%</b><br>
{prof["n_stars_pulled"]} stars read<br>{len(prof["own_repos"])} own repos<br>model v{state["model_version"]}
</div></div></div></div>
<script>
const d=document.getElementById('dossier');
const w=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+
  location.pathname.replace(/\\/u\\/[^/]+$/,'')+'/ws/dossier/{html.escape(pred["login"])}');
w.onmessage=(m)=>{{const x=JSON.parse(m.data);if(x.t==='tok')d.textContent+=x.d}};
const cards=[...document.querySelectorAll('.grid .card')];
let vis={SHELF_FIRST},lo=0,hi=Infinity;
const fits=c=>{{const s=+c.dataset.s;return s>=lo&&s<hi}};
const paint=()=>{{let n=0;cards.forEach(c=>{{c.hidden=!(fits(c)&&n<vis);n+=fits(c)}});
  document.getElementById('more').hidden=vis>=n}};
document.getElementById('more').onclick=()=>{{vis+=5;paint()}};
document.querySelectorAll('#chips button').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('#chips button').forEach(x=>x.classList.toggle('active',x===b));
  lo=+b.dataset.lo;hi=b.dataset.hi===''?Infinity:+b.dataset.hi;vis={SHELF_FIRST};paint()}});
paint();
</script>"""
    return page(f"unstarred / {pred['login']}", body, page_login=pred["login"])


@app.websocket("/ws/dossier/{login}")
async def ws_dossier(ws: WebSocket, login: str):
    await ws.accept()
    try:
        pred = await asyncio.to_thread(get_profile, login)
        if "error" in pred:
            await ws.send_json({"t": "tok", "d": pred["error"]})
        else:
            await stream_dossier(ws, pred)
        await ws.send_json({"t": "end"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"t": "tok", "d": f"dossier error: {e}"})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/ws/ask")
async def ws_ask(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_json()
        await run_ask_stream(ws, str(raw.get("q", ""))[:500],
                             [tuple(h) for h in raw.get("history", [])][-6:],
                             raw.get("login") or None)
        await ws.send_json({"t": "end"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"t": "tok", "d": f"error: {e}"})
            await ws.send_json({"t": "end"})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/health")
def health():
    return {"ok": True, "candidates": len(state["repos"]), "model_version": state["model_version"]}


_PROXY_MOUNT = re.compile(r"^/hopsworks-api/pythonapp/[^/]+/[^/]+")


class StripForwardedPrefix:
    """Strip the Hopsworks proxy mount from the path (this cluster sets no
    APP_BASE_URL_PATH env and no X-Forwarded-Prefix header) so routes match, and
    record it as root_path for absolute links. Without it the app 404s forever."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            prefix = dict(scope.get("headers") or {}).get(b"x-forwarded-prefix", b"").decode().rstrip("/")
            if not prefix:
                m = _PROXY_MOUNT.match(scope["path"])
                prefix = m.group(0) if m else ""
            if prefix and scope["path"].startswith(prefix):
                scope = dict(scope)
                scope["path"] = scope["path"][len(prefix):] or "/"
                scope["root_path"] = prefix
        await self.inner(scope, receive, send)


application = StripForwardedPrefix(app)


if __name__ == "__main__":
    boot()
    uvicorn.run(application, host="0.0.0.0", port=8000)

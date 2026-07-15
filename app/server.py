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
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse

import hopsworks

DOSSIER_MODEL = "claude-sonnet-5"
ASK_MODEL = "claude-sonnet-5"
SHELF_K = 60
PROFILE_TTL = 900

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
    state["metrics"] = model.training_metrics or {}

    # display frame + anchor vectors in memory: 300k x 64 floats is ~80MB and
    # buys instant anchor lookup + name search; the online KNN stays the
    # retrieval path for shelf and chat
    df = state["emb_fg"].select_all().read()
    df["full_name_lc"] = df["full_name"].str.lower()
    state["vecs"] = np.array(df["item_vector"].tolist(), dtype="float32")
    state["repos"] = df.drop(columns=["item_vector"]).reset_index(drop=True)
    state["row_by_id"] = {int(r): i for i, r in enumerate(df["repo_id"])}
    state["id_by_name"] = {n: int(i) for n, i in zip(df["full_name_lc"], df["repo_id"])}

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


def shelf_for(pred: dict, k: int = SHELF_K) -> list[dict]:
    starred = set(pred.get("starred_ids", []))
    hits = _rows(state["emb_fg"], state["emb_fg"].find_neighbors(pred["user_vector"], k=k + len(starred) + 20))
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
        if len(rows) >= k:
            break
    return rows


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
        resp, _text = await asyncio.to_thread(call)
        if resp.stop_reason != "tool_use":
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

CSS = """
:root{--bg:#0b0e11;--panel:#12161b;--ink:#e5e7eb;--dim:#9ca3af;--acc:#34d399;--line:#1f2937}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 ui-sans-serif,system-ui,sans-serif}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1060px;margin:0 auto;padding:24px 20px 80px}
h1{font-size:26px;margin:.2em 0}h1 a{color:var(--ink)}
.sub{color:var(--dim);max-width:640px}
form.login{display:flex;gap:8px;margin:22px 0}
input[type=text]{flex:1;max-width:340px;background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:10px 12px;color:var(--ink);font-size:15px}
button{background:var(--acc);color:#04110b;border:0;border-radius:8px;padding:10px 18px;
font-weight:600;cursor:pointer}
.cols{display:grid;grid-template-columns:1fr 320px;gap:24px}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .nm{font-weight:600}.card .meta{color:var(--dim);font-size:12.5px;margin-top:2px}
.card .pitch{font-size:13.5px;margin-top:6px;color:#c8cdd4}
.bar{height:3px;background:var(--line);border-radius:2px;margin-top:8px}
.bar i{display:block;height:3px;background:var(--acc);border-radius:2px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.panel h3{margin:0 0 8px;font-size:14px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim)}
#dossier{white-space:pre-wrap;font-size:14px;min-height:80px}
.langs span{display:inline-block;background:#0f1720;border:1px solid var(--line);border-radius:20px;
padding:2px 10px;margin:2px 4px 2px 0;font-size:12.5px;color:var(--dim)}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:20px}
.tabs button{background:none;color:var(--dim);border:0;border-bottom:2px solid transparent;
border-radius:0;padding:10px 14px;font-weight:600;cursor:pointer}
.tabs button.active{color:var(--ink);border-bottom-color:var(--acc)}
#librarian{max-width:720px}
#log{min-height:200px;max-height:60vh;overflow-y:auto;padding:12px 16px;font-size:14px;
background:var(--panel);border:1px solid var(--line);border-radius:10px}
#log .q{color:var(--acc);margin-top:10px}#log .a{white-space:pre-wrap}#log .t{color:var(--dim);font-size:12px}
#askform{display:flex;gap:6px;margin-top:10px}
#askform input{flex:1}
.err{color:#f87171}
footer{margin-top:40px;color:var(--dim);font-size:12.5px}
"""

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
      t.textContent='[consulting '+d.d+'…]';log.insertBefore(t,ad)}
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
<section id="librarian" hidden><div id="log"></div>
<form id="askform"><input type="text" id="q" placeholder="the perfect repo for…" autocomplete="off">
<button>ask</button></form></section>
<footer>unstarred #012 · a two-tower recommender with an LLM on top · rank is resemblance
to starring behavior, never a quality verdict · <a href="https://github.com/MagicLex/unstarred">source</a></footer>
</div>
<script>const PAGE_LOGIN={json.dumps(page_login or None)};{ASK_JS}</script>
</body></html>""")


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    n = len(state["repos"])
    m = state["metrics"]
    body = f"""
<h1>unstarred</h1>
<p class="sub">Which repos would you have starred already, if you had seen them?
A two-tower model trained on {n:,} repos' worth of public star histories reads your
stars and your own repos, and ranks what you haven't seen. The librarian in the corner
talks to the same model.</p>
<form class="login" action="go" method="get">
<input type="text" name="login" placeholder="your GitHub username" required>
<button>read my stars</button></form>
<p class="sub">Held out by time, the model puts a future star in its top 50
{float(m.get("recall_at_50", 0)) * 100:.1f}% of the time; showing everyone the same
trending list manages {float(m.get("pop_recall_at_50", 0)) * 100:.1f}%. Model
v{state["model_version"]}.</p>"""
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
        f"""<div class="card"><div class="nm"><a href="https://github.com/{html.escape(r["full_name"])}"
target="_blank" rel="noopener">{html.escape(r["full_name"])}</a></div>
<div class="meta">{r["stars"]:,}★ · {html.escape(r["language"])}{" · " + html.escape(r["category"]) if r["category"] else ""}</div>
<div class="pitch">{html.escape(r["pitch"])}</div>
<div class="bar"><i style="width:{max(6, int(100 * r["score"] / top))}%"></i></div></div>"""
        for r in rows)
    prof = pred["profile"]
    langs = "".join(f"<span>{html.escape(x)}</span>" for x in prof["top_languages"])
    body = f"""
<h1><a href="../">unstarred</a> / {html.escape(pred["login"])}</h1>
<div class="cols"><div>
<p class="sub">{prof["n_stars_pulled"]} recent stars read, {len(prof["own_repos"])} own repos.
Already-starred repos are filtered out. Bars are model score relative to your top hit.</p>
<div class="grid">{cards}</div></div>
<div><div class="panel"><h3>the dossier</h3><div class="langs">{langs}</div>
<div id="dossier"></div></div></div></div>
<script>
const d=document.getElementById('dossier');
const w=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+
  location.pathname.replace(/\\/u\\/[^/]+$/,'')+'/ws/dossier/{html.escape(pred["login"])}');
w.onmessage=(m)=>{{const x=JSON.parse(m.data);if(x.t==='tok')d.textContent+=x.d}};
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

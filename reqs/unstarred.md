# unstarred: spec (#012 awesome-ml-systems)

**The question:** which repos would you have starred already, if you had seen them?

A GitHub repo recommender: a real two-tower retrieval model trained on public star
histories, with an LLM on top in three load-bearing roles. Type your username and
get your shelf; open the chat and talk your way to the perfect repo, tool, or
reference. The LLM never picks items: every recommendation is a K-NN hit in the
trained embedding space, filtered and steered by what you said.

## AI-system card

| | |
|---|---|
| prediction problem | ranking / retrieval: predict which repos a user stars next |
| KPI | user finds a repo worth acting on (star/clone) that trending would not have shown |
| proxy metric | recall@k and MRR on temporally held-out future stars |
| blind baseline | global popularity in the feature window (what "trending" gives everyone) |
| ablations | popularity vs pure two-tower vs two-tower + LLM item fingerprints |
| data sources | GH Archive (bulk, free), GitHub REST API (free token), raw READMEs |
| ML-system type | real-time (KServe query tower + online embedding index) |
| consumption | server-rendered app: personal shelf + dossier + chat pane (JS hydrates) |
| monitoring | inference-log FG (inputs, candidates, scores); score-distribution drift |
| secrets | `GITHUB_TOKEN`, `ANTHROPIC_API_KEY` in project secrets |

Signal framing: a recommendation is "this resembles what you star", never a claim
about quality of the repo or its authors.

## Data (measured 2026-07-15)

- **GH Archive**: `https://data.gharchive.org/YYYY-MM-DD-H.json.gz`, ~21 MB/hour,
  no auth. WatchEvent (star) rate observed 113 to 372/hour. Used only to *discover*
  currently-active starring users, not as the training matrix.
- **GitHub REST API**, 5000 req/hr with token:
  `/users/{u}/starred` with `Accept: application/vnd.github.star+json` returns the
  full star history **with `starred_at` timestamps** (verified). `/users/{u}/repos`
  gives the user's own repos (language, description, topics).
- **READMEs** via `raw.githubusercontent.com` for the candidate corpus.

Scale decided: ~20k active users, est. 2 to 4M timestamped interactions, 150 to
300k distinct repos; LLM fingerprints capped to the top ~40k candidate repos by
star count in corpus. Captures frozen to `data/`, kept out of git, attached to the
release (series standard).

## Pipelines

### F1. discover-users (job). Blocked by: nothing
GH Archive, trailing ~7 days, stream-filter WatchEvents. Emit ~20k distinct active
starrers (dedup, drop bots by login heuristics). Output: seed list file in
`data/` (intermediate, not an FG: it is consumed once by F2).
Skill: hops-job.

### F2. pull-interactions (job, overnight). Blocked by: F1
GitHub API snowball with `GITHUB_TOKEN`: per user, full star history + own repos.
Writes two FGs:
- `star_events`: pk `(user_login, repo_id)`, event_time `starred_at`. The
  interaction matrix, temporally honest.
- `repos`: pk `repo_id`. Metadata: full_name, language, topics, stars, forks,
  created/pushed dates, description. Event time = capture time.
Rate-limit aware (bounded active polling on the reset header), resumable,
checkpoints to `data/`. Skill: hops-fg.

### F3. readme-fingerprints (job). Blocked by: F2
For the top ~40k candidate repos: fetch README (snapshot to `data/`, one capture
date so no future text leaks past the split), then Claude (batch API) emits a
structured fingerprint: audience tags (who stars this), maturity, category,
one-line pitch. Writes `repo_fingerprints` FG: pk `repo_id`, structured fields
plus a text-embedding column with an **embedding index** (online KNN). This is the
cold-start path: a 12-star repo is retrievable by content before CF ever sees it.
Skill: hops-fg (embeddings), hops-transformations.

### T1. train-towers (job, 8 GB). Blocked by: F2, F3
Feature view `unstarred_fv`: `star_events` joined with `repos` +
`repo_fingerprints`. Label = the interaction (implicit positive), negatives
sampled in-batch. MDTs (categorical encodings, normalizations) attached to the FV
as UDFs, identical at train and serve. **Temporal split**: train on stars before
T, validate/test on stars after; popularity baseline computed on the same split.
Two-tower Keras: user tower (aggregates of the user's history: language/topic
distribution, own-repo signals), item tower (repo metadata + fingerprint fields).
Controls: shuffle-label run, popularity blind baseline, fingerprint on/off
ablation. Register in Model Registry: query tower and candidate tower as one
versioned model with metrics JSON, recall@k curves, embedding-space plots. Every
version kept; retired versions keep the note why. Skill: hops-fv, hops-train.

### I1. embed-candidates (job). Blocked by: T1
Run the candidate tower over the corpus, write `repo_embeddings` FG with embedding
index (online KNN over the trained space). Re-runs on retrain. Skill:
hops-batch-inference.

### I2. query-tower deployment (KServe). Blocked by: T1
Deploy the query tower. Request-time inputs (the live user's stars + own repos
pulled from the GitHub API at request) become on-demand features computed in the
predictor; precomputed repo features come from the online store. Logs every
request + returned candidates to an inference-log FG. Skill: hops-online-inference.

### I3. app `unstarred`. Blocked by: I1, I2
Server-rendered (FastAPI, series standard), JS hydration on top, no SPA.
- **Shelf**: username, then live pull of stars + own repos, query tower, KNN over
  `repo_embeddings`, ranked shelf with already-starred filtered out.
- **Dossier**: Claude reads the pulled profile and writes a taste dossier rendered
  to the user ("what your GitHub says about you"), also the grounding for chat.
- **Chat**: the steering wheel. Claude compiles each utterance into a typed query:
  anchor repos, embedding-space direction, hard filters (language, topics,
  activity, size). Executes against the same index; answers only from returned
  hits with scores. Tool-use loop, deterministic tools, same never-invent rules as
  empty-chair's ask.py.
Skill: hops-app.

## Order

```
F1 -> F2 -> F3 -> T1 -> I1 -> I3
                   \--> I2 --/
```

## Honesty checklist

- Metric on future stars only; feature window strictly before label window.
- README snapshots dated before the split; fingerprints generated from them only.
- Shuffle-label control must collapse to baseline.
- Popularity baseline is the real competitor (it is what GitHub trending gives).
- If the LLM fingerprints do not lift recall, the README says so and they stay
  only for cold-start + chat, reported separately.

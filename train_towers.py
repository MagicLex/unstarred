"""T1: two-tower retrieval trained on the star matrix, evaluated temporally.

Feature view `unstarred_fv` (star_events joined with repos + fingerprints,
log1p UDFs attached) is the read side. The trainer:

  1. temporal split: train on stars before the 80th-percentile timestamp T,
     validate on (70q, 80q], test on stars after T (never shuffled)
  2. Keras two-tower: item tower = id embedding + content (language, category,
     maturity, audience, numerics); user tower = shared-id-embedding average
     over the last HIST starred repos + taste scalars. In-batch softmax.
  3. honest eval on future stars: recall@k / MRR vs the popularity blind
     baseline, with --shuffle-labels and --drop fingerprints controls
  4. registers query + candidate towers as one versioned model with metrics
     JSON + plots. Every version stays in the registry.

Deployed as a Hopsworks job (tensorflow-training-pipeline, 16GB).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import hopsworks
from hopsworks import udf

from unstarred_features import SCALARS, scalars_vector

HIST = 30
DIM = 64
ID_EMB = 32
MAX_TRAIN = 2_000_000  # cap on training pairs; logged when it bites

ITEM_STR = ["language", "category", "maturity", "audience"]
ITEM_NUM = ["log1p_stars", "log1p_forks", "log1p_size_kb", "age_days", "is_fork_f", "archived_f", "readme_found_f"]
USER_NUM = list(SCALARS)


@udf(return_type=float, drop=[])
def log1p_stars(stars):
    import numpy as np
    return np.log1p(stars)


@udf(return_type=float, drop=[])
def log1p_forks(forks):
    import numpy as np
    return np.log1p(forks)


@udf(return_type=float, drop=[])
def log1p_size_kb(size_kb):
    import numpy as np
    return np.log1p(size_kb)


def get_or_create_fv(fs):
    stars = fs.get_feature_group("star_events", version=1)
    repos = fs.get_feature_group("repos", version=1)
    fps = fs.get_feature_group("repo_fingerprints", version=1)
    query = (
        stars.select(["user_login", "repo_id", "starred_at"])
        .join(repos.select(["language", "topics", "stars", "forks", "size_kb", "created_at", "is_fork", "archived"]), on=["repo_id"])
        .join(fps.select(["category", "maturity", "audience", "readme_found"]), on=["repo_id"], join_type="left")
    )
    return fs.get_or_create_feature_view(
        name="unstarred_fv",
        version=1,
        query=query,
        description="Star interactions with item content features; log1p UDFs attached (train/serve identical)",
        transformation_functions=[log1p_stars("stars"), log1p_forks("forks"), log1p_size_kb("size_kb")],
    )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["starred_at"] = pd.to_datetime(df["starred_at"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["age_days"] = (df["starred_at"] - df["created_at"]).dt.days.clip(lower=0).fillna(0).astype("float32")
    for src, dst in (("is_fork", "is_fork_f"), ("archived", "archived_f"), ("readme_found", "readme_found_f")):
        df[dst] = df[src].fillna(False).astype("float32")
    for col in ("language", "category", "maturity", "audience"):
        df[col] = df[col].fillna("none").astype(str)
    # UDF output naming varies across hsfs versions; normalize, recompute if absent
    for base in ("stars", "forks", "size_kb"):
        target = f"log1p_{base}"
        df[base] = pd.to_numeric(df.get(base), errors="coerce").fillna(0)
        if target not in df.columns:
            cand = [c for c in df.columns if c.startswith(target)]
            df[target] = df[cand[0]] if cand else np.log1p(df[base])
        df[target] = df[target].fillna(0.0).astype("float32")
    df["repo_id"] = df["repo_id"].astype(str)
    return df.sort_values("starred_at").reset_index(drop=True)


def user_scalar_frame(train: pd.DataFrame, own: pd.DataFrame, asof) -> pd.DataFrame:
    """Taste scalars per user, via the SAME shared function the predictor calls
    on the live pull (unstarred_features.scalars_vector). History = the user's
    most recent train-window stars, capped identically to serve."""
    own_by_user = {}
    if not own.empty:
        for u, grp in own.groupby("user_login"):
            own_by_user[u] = grp[["language"]].assign(stars=0, created_at=pd.NaT).to_dict("records")
    rows, index = [], []
    cols = ["language", "stars", "created_at"]
    for u, grp in train.groupby("user_login", sort=False):
        hist = grp[cols].iloc[::-1].to_dict("records")  # newest first
        rows.append(scalars_vector(hist, own_by_user.get(u, []), asof))
        index.append(u)
    return pd.DataFrame(rows, columns=USER_NUM, index=index).astype("float32")


def build_model(item_vocab: dict, drop_fingerprints: bool):
    import keras
    from keras import layers, ops

    id_lookup = layers.StringLookup(vocabulary=item_vocab["repo_id"], name="id_lookup")
    id_emb = layers.Embedding(len(item_vocab["repo_id"]) + 1, ID_EMB, name="id_emb")

    # item tower
    it_id = keras.Input(shape=(), dtype="string", name="repo_id")
    it_num = keras.Input(shape=(len(ITEM_NUM),), dtype="float32", name="item_num")
    parts = [id_emb(id_lookup(it_id)), it_num]
    str_inputs = {}
    for col in ITEM_STR:
        if drop_fingerprints and col in ("category", "maturity", "audience"):
            continue
        inp = keras.Input(shape=(), dtype="string", name=col)
        str_inputs[col] = inp
        lk = layers.StringLookup(vocabulary=item_vocab[col], name=f"{col}_lookup")
        parts.append(layers.Embedding(len(item_vocab[col]) + 1, 8, name=f"{col}_emb")(lk(inp)))
    x = layers.Concatenate()(parts)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(DIM)(x)
    item_out = layers.UnitNormalization(axis=-1)(x)
    candidate_tower = keras.Model([it_id, it_num, *str_inputs.values()], item_out, name="candidate_tower")

    # user tower: shared id embedding averaged over history + scalars
    u_hist = keras.Input(shape=(HIST,), dtype="string", name="hist_ids")
    u_num = keras.Input(shape=(len(USER_NUM),), dtype="float32", name="user_num")
    h_idx = id_lookup(u_hist)
    h_emb = id_emb(h_idx)
    mask = ops.expand_dims(ops.cast(ops.not_equal(h_idx, 0), "float32"), -1)
    h_avg = ops.sum(h_emb * mask, axis=1) / ops.maximum(ops.sum(mask, axis=1), 1.0)
    y = layers.Concatenate()([h_avg, u_num])
    y = layers.Dense(128, activation="relu")(y)
    y = layers.Dense(DIM)(y)
    user_out = layers.UnitNormalization(axis=-1)(y)
    query_tower = keras.Model([u_hist, u_num], user_out, name="query_tower")
    return query_tower, candidate_tower


def make_examples(train: pd.DataFrame, scalars: pd.DataFrame, rng: np.random.Generator):
    """Per positive (u, r, t): the user's HIST most recent repo ids before t."""
    hist_ids = np.full((len(train), HIST), "", dtype=object)
    by_user = train.groupby("user_login", sort=False).indices
    keep = np.ones(len(train), dtype=bool)
    for user, idx in by_user.items():
        ids = train["repo_id"].to_numpy()[idx]  # idx is time-ordered (df sorted)
        for j, row in enumerate(idx):
            if j == 0:
                keep[row] = False  # no history yet, nothing to learn from
                continue
            h = ids[max(0, j - HIST) : j][::-1]
            hist_ids[row, : len(h)] = h
    sel = np.where(keep)[0]
    if len(sel) > MAX_TRAIN:
        print(f"capping training pairs {len(sel)} -> {MAX_TRAIN}", flush=True)
        sel = rng.choice(sel, MAX_TRAIN, replace=False)
    t = train.iloc[sel]
    u_num = scalars.reindex(t["user_login"]).fillna(0.0).to_numpy("float32")
    return {
        "hist_ids": hist_ids[sel].astype(str),
        "user_num": u_num,
        "repo_id": t["repo_id"].to_numpy(),
        "item_num": t[ITEM_NUM].to_numpy("float32"),
        **{c: t[c].to_numpy() for c in ITEM_STR},
    }


def evaluate(query_tower, candidate_tower, items: pd.DataFrame, test: pd.DataFrame,
             train: pd.DataFrame, scalars: pd.DataFrame, ks=(10, 50, 100)) -> dict:
    import tensorflow as tf

    item_inputs = {"repo_id": items["repo_id"].to_numpy(), "item_num": items[ITEM_NUM].to_numpy("float32")}
    for c in ITEM_STR:
        if c in {i.name for i in candidate_tower.inputs}:
            item_inputs[c] = items[c].to_numpy()
    item_vecs = candidate_tower.predict(item_inputs, batch_size=4096, verbose=0)
    id_to_row = {rid: i for i, rid in enumerate(items["repo_id"])}

    seen = train.groupby("user_login")["repo_id"].agg(set)
    last_hist = train.groupby("user_login")["repo_id"].agg(lambda s: list(s)[-HIST:][::-1])
    users = [u for u in test["user_login"].unique() if u in last_hist.index]
    hist = np.full((len(users), HIST), "", dtype=object)
    for i, u in enumerate(users):
        h = last_hist[u]
        hist[i, : len(h)] = h
    u_num = scalars.reindex(users).fillna(0.0).to_numpy("float32")
    user_vecs = query_tower.predict({"hist_ids": tf.constant(hist.astype(str)), "user_num": u_num}, batch_size=2048, verbose=0)
    u_row = {u: i for i, u in enumerate(users)}

    pop_rank = items["log1p_stars"].to_numpy()  # popularity blind baseline score

    hits = {k: [0, 0] for k in ks}  # model
    pop_hits = {k: [0, 0] for k in ks}
    rr, pop_rr = [], []
    test_by_user = test[test["user_login"].isin(u_row)].groupby("user_login")["repo_id"].agg(list)
    for u, wanted in test_by_user.items():
        scores = item_vecs @ user_vecs[u_row[u]]
        for rid in seen.get(u, ()):  # never recommend already-starred
            if rid in id_to_row:
                scores[id_to_row[rid]] = -np.inf
        order = np.argsort(-scores)
        rank_of = {items["repo_id"].iloc[r]: p for p, r in enumerate(order[:200])}
        pop_order = np.argsort(-pop_rank)
        pop_rank_of = {items["repo_id"].iloc[r]: p for p, r in enumerate(pop_order[:200])}
        for rid in wanted:
            if rid not in id_to_row:
                continue
            r_m = rank_of.get(rid)
            r_p = pop_rank_of.get(rid)
            for k in ks:
                hits[k][1] += 1
                pop_hits[k][1] += 1
                hits[k][0] += r_m is not None and r_m < k
                pop_hits[k][0] += r_p is not None and r_p < k
            rr.append(1.0 / (r_m + 1) if r_m is not None and r_m < 100 else 0.0)
            pop_rr.append(1.0 / (r_p + 1) if r_p is not None and r_p < 100 else 0.0)

    m = {f"recall_at_{k}": hits[k][0] / max(hits[k][1], 1) for k in ks}
    m |= {f"pop_recall_at_{k}": pop_hits[k][0] / max(pop_hits[k][1], 1) for k in ks}
    m["mrr_at_100"] = float(np.mean(rr)) if rr else 0.0
    m["pop_mrr_at_100"] = float(np.mean(pop_rr)) if pop_rr else 0.0
    m["n_test_users"] = len(test_by_user)
    m["n_test_stars"] = int(sum(len(v) for v in test_by_user.values))
    return m


def plots(metrics: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [10, 50, 100]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, [metrics[f"recall_at_{k}"] for k in ks], "o-", label="two-tower")
    ax.plot(ks, [metrics[f"pop_recall_at_{k}"] for k in ks], "s--", label="popularity (blind)")
    ax.set_xlabel("k"); ax.set_ylabel("recall@k"); ax.legend(); ax.set_title("future-star recall")
    fig.tight_layout(); fig.savefig(out / "recall.png", dpi=120); plt.close(fig)


def main() -> None:
    import tensorflow as tf

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--drop", choices=["fingerprints"], default=None)
    ap.add_argument("--shuffle-labels", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(11)
    tf.keras.utils.set_random_seed(11)

    project = hopsworks.login()
    fs = project.get_feature_store()
    fv = get_or_create_fv(fs)
    df, _ = fv.training_data(description="full star corpus; temporal split in trainer")
    df = prepare(df)
    # offline FGs append across job runs (PK dedup is online-only); the repos
    # join doubles star rows for re-seen repos, so dedupe the joined event
    df = df.drop_duplicates(["user_login", "repo_id"], keep="last")
    print(f"corpus: {len(df)} stars, {df['user_login'].nunique()} users, {df['repo_id'].nunique()} repos", flush=True)

    t_hi = df["starred_at"].quantile(0.8)
    train, test = df[df["starred_at"] <= t_hi], df[df["starred_at"] > t_hi]
    print(f"split at {t_hi}: train {len(train)}, test {len(test)}", flush=True)

    if args.shuffle_labels:
        train = train.copy()
        perm = rng.permutation(len(train))
        for c in ["repo_id", *ITEM_STR, *ITEM_NUM]:
            train[c] = train[c].to_numpy()[perm]
        print("CONTROL: labels shuffled", flush=True)

    try:
        own = fs.get_feature_group("own_repos", version=1).read()
        repo_langs = fs.get_feature_group("repos", version=1).read()[["repo_id", "language"]] \
            .drop_duplicates("repo_id", keep="last")
        own = own.merge(repo_langs, on="repo_id", how="left")
        own["language"] = own["language"].fillna("none")
    except Exception as e:
        print(f"own_repos unavailable: {e}", flush=True)
        own = pd.DataFrame(columns=["user_login", "language"])
    scalars = user_scalar_frame(train, own, t_hi)

    items = train.drop_duplicates("repo_id", keep="last")[["repo_id", *ITEM_STR, *ITEM_NUM]].reset_index(drop=True)
    vocab = {"repo_id": items["repo_id"].tolist()}
    for c in ITEM_STR:
        vocab[c] = sorted(train[c].unique().tolist())

    query_tower, candidate_tower = build_model(vocab, drop_fingerprints=args.drop == "fingerprints")
    ex = make_examples(train, scalars, rng)
    n = len(ex["repo_id"])
    print(f"training pairs: {n}", flush=True)

    item_keys = [i.name for i in candidate_tower.inputs]
    # logQ correction (Yi et al. 2019): in-batch negatives appear with P(item) ~
    # its frequency, which otherwise punishes popular repos twice; subtract
    # log P so the softmax scores taste, not rarity
    freq = pd.Series(ex["repo_id"]).value_counts(normalize=True)
    ex["log_q"] = np.log(pd.Series(ex["repo_id"]).map(freq).to_numpy("float32"))
    ds = tf.data.Dataset.from_tensor_slices((
        {"hist_ids": ex["hist_ids"], "user_num": ex["user_num"]},
        {k: ex[k] for k in item_keys},
        ex["log_q"],
    )).shuffle(200_000, seed=11).batch(args.batch_size, drop_remainder=True).prefetch(2)

    opt = tf.keras.optimizers.Adam(1e-3)
    temperature = 0.05

    @tf.function
    def step(u_in, i_in, log_q):
        with tf.GradientTape() as tape:
            u = query_tower(u_in, training=True)
            v = candidate_tower(i_in, training=True)
            logits = tf.matmul(u, v, transpose_b=True) / temperature - log_q[None, :]
            labels = tf.range(tf.shape(logits)[0])
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
        variables = query_tower.trainable_variables + candidate_tower.trainable_variables
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    for epoch in range(args.epochs):
        losses = []
        for u_in, i_in, log_q in ds:
            losses.append(float(step(u_in, i_in, log_q)))
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}", flush=True)

    metrics = evaluate(query_tower, candidate_tower, items, test, train, scalars)
    print(json.dumps(metrics, indent=2), flush=True)

    if args.no_register:
        return
    out = Path("model_out")
    shutil.rmtree(out, ignore_errors=True)
    (out / "assets_x").mkdir(parents=True)
    query_tower.export(str(out / "query_tower"))
    candidate_tower.export(str(out / "candidate_tower"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shutil.copy(Path(__file__).resolve().parent / "unstarred_features.py", out / "unstarred_features.py")
    (out / "config.json").write_text(json.dumps({
        "hist": HIST, "dim": DIM, "item_num": ITEM_NUM, "user_num": USER_NUM,
        "item_str": ITEM_STR, "drop": args.drop, "shuffle": args.shuffle_labels,
        "split_t": str(t_hi), "temperature": temperature,
    }, indent=2))
    plots(metrics, out)

    mr = project.get_model_registry()
    # python, not tensorflow: TF models reject custom predictor scripts at deploy
    model = mr.python.create_model(
        name="unstarred",
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        description=f"two-tower retrieval; drop={args.drop}; shuffle={args.shuffle_labels}",
    )
    model.save(str(out))
    print(f"registered unstarred v{model.version}", flush=True)


if __name__ == "__main__":
    main()

"""I1: run the candidate tower over the whole repo corpus.

Loads the latest `unstarred` model from the registry, builds the item frame
from `repos` + `repo_fingerprints` (same features as training, age measured at
scoring time), and writes `repo_embeddings`: an online FG with a KNN index
over the trained space. This is the retrieval surface the app and the chat
query against. Re-run after every retrain.

Deployed as a Hopsworks job (tensorflow-training-pipeline).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hopsworks

EMB_DOC = {
    "repo_id": "GitHub numeric repo id (primary key)",
    "full_name": "owner/name (denormalized for direct rendering)",
    "language": "primary language",
    "stars": "stargazer count at capture",
    "category": "LLM fingerprint category ('' when not fingerprinted)",
    "pitch": "LLM one-sentence pitch ('' when not fingerprinted)",
    "model_version": "unstarred model version that produced the vector",
    "embedded_at": "embedding timestamp (event time)",
    "item_vector": "candidate-tower embedding (trained space, online KNN index)",
}


def item_frame(fs) -> pd.DataFrame:
    repos = fs.get_feature_group("repos", version=1).read()
    # offline FGs append across job runs (PK dedup is online-only)
    repos = repos.sort_values("captured_at").drop_duplicates("repo_id", keep="last")
    try:
        fps = fs.get_feature_group("repo_fingerprints", version=1).read()
        fps = fps[["repo_id", "category", "maturity", "audience", "pitch", "readme_found"]] \
            .drop_duplicates("repo_id", keep="last")
    except Exception as e:
        print(f"no fingerprints FG: {e}", flush=True)
        fps = pd.DataFrame(columns=["repo_id", "category", "maturity", "audience", "pitch", "readme_found"])
    df = repos.merge(fps, on="repo_id", how="left")
    now = pd.Timestamp.utcnow()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["age_days"] = (now - df["created_at"]).dt.days.clip(lower=0).fillna(0).astype("float32")
    df["is_fork_f"] = df["is_fork"].fillna(False).astype("float32")
    df["archived_f"] = df["archived"].fillna(False).astype("float32")
    df["readme_found_f"] = df["readme_found"].fillna(False).astype("float32")
    for col in ("language", "category", "maturity", "audience"):
        df[col] = df[col].fillna("none").astype(str)
    # same math as the FV's log1p UDFs; item side is precomputed here, so
    # online inference never recomputes item transforms
    for src in ("stars", "forks", "size_kb"):
        df[f"log1p_{src}"] = np.log1p(df[src].fillna(0)).astype("float32")
    df["repo_id_str"] = df["repo_id"].astype(str)
    return df


def embed(df: pd.DataFrame, model_dir: Path, config: dict) -> np.ndarray:
    import tensorflow as tf

    tower = tf.saved_model.load(str(model_dir / "candidate_tower"))
    fn = tower.signatures["serving_default"]
    input_names = list(fn.structured_input_signature[1].keys())
    print(f"candidate tower inputs: {input_names}", flush=True)

    n = len(df)
    out = np.zeros((n, config["dim"]), dtype="float32")
    bs = 4096
    for start in range(0, n, bs):
        chunk = df.iloc[start : start + bs]
        feed = {}
        for name in input_names:
            if name == "repo_id":
                feed[name] = tf.constant(chunk["repo_id_str"].to_numpy())
            elif name == "item_num":
                feed[name] = tf.constant(chunk[config["item_num"]].to_numpy("float32"))
            else:
                feed[name] = tf.constant(chunk[name].to_numpy())
        vec = list(fn(**feed).values())[0].numpy()
        out[start : start + len(chunk)] = vec
        if start % (bs * 10) == 0:
            print(f"embedded {start}/{n}", flush=True)
    return out


def main() -> None:
    from hsfs import embedding

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-version", type=int, default=None, help="default: latest")
    args = ap.parse_args()

    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    if args.model_version:
        model = mr.get_model("unstarred", version=args.model_version)
    else:
        model = max(mr.get_models("unstarred"), key=lambda m: m.version)
    model_dir = Path(model.download())
    config = json.loads((model_dir / "config.json").read_text())
    print(f"model unstarred v{model.version}, dim {config['dim']}", flush=True)

    df = item_frame(fs)
    print(f"corpus: {len(df)} repos", flush=True)
    vecs = embed(df, model_dir, config)

    out = pd.DataFrame({
        "repo_id": df["repo_id"].astype("int64"),
        "full_name": df["full_name"],
        "language": df["language"],
        "stars": df["stars"].fillna(0).astype("int64"),
        "category": df["category"].replace("none", ""),
        "pitch": df["pitch"].fillna(""),
        "model_version": int(model.version),
        "embedded_at": pd.Timestamp.utcnow(),
        "item_vector": [v.tolist() for v in vecs],
    })

    ei = embedding.EmbeddingIndex()
    ei.add_embedding(name="item_vector", dimension=config["dim"])
    fg = fs.get_or_create_feature_group(
        name="repo_embeddings",
        version=1,
        description="Candidate-tower embeddings over the repo corpus (trained space, online KNN)",
        primary_key=["repo_id"],
        event_time="embedded_at",
        online_enabled=True,
        embedding_index=ei,
    )
    fg.insert(out)
    for name, desc in EMB_DOC.items():
        fg.update_feature_description(name, desc)
    print(f"wrote {len(out)} embeddings (model v{model.version})", flush=True)


if __name__ == "__main__":
    main()

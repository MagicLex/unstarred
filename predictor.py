"""I2: KServe predictor for the query tower.

Request: {"instances": [{"login": "<github user>"}]}. The login is the only
request parameter; everything else is on-demand: the predictor pulls the
user's public stars + own repos live from the GitHub API, builds the history
and taste scalars through the SAME shared module training used
(unstarred_features, bundled in the model artifact), and runs the query tower.

Returns the user vector plus the pulled profile (the app does the KNN against
repo_embeddings and grounds the dossier/chat on the profile). Inference
logging is platform-side (InferenceLogger on the deployment).
"""

import glob
import os
import sys
from datetime import datetime, timezone

import requests


def model_files_root() -> str:
    for root in (os.environ.get("MODEL_FILES_PATH"), os.environ.get("ARTIFACT_FILES_PATH"),
                 "/mnt/models", "/mnt/artifacts"):
        if root and glob.glob(f"{root}/**/config.json", recursive=True):
            return os.path.dirname(glob.glob(f"{root}/**/config.json", recursive=True)[0])
    raise FileNotFoundError("model files not found under the model/artifact mounts")


API = "https://api.github.com"
STAR_ACCEPT = "application/vnd.github.star+json"


class Predict:
    def __init__(self):
        import json
        import tensorflow as tf
        import hopsworks

        root = model_files_root()
        sys.path.insert(0, root)
        self.config = json.loads(open(f"{root}/config.json").read())
        tower = tf.saved_model.load(f"{root}/query_tower")
        self.fn = tower.signatures["serving_default"]
        self.tf = tf

        hopsworks.login()
        token = hopsworks.get_secrets_api().get_secret("GITHUB_TOKEN").value
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "unstarred-012",
        })
        print("query tower loaded, github session ready", flush=True)

    def _get(self, url, accept=None, params=None):
        resp = self.session.get(url, params=params, timeout=10,
                                headers={"Accept": accept} if accept else None)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def _pull(self, login):
        from unstarred_features import repo_row

        captured = datetime.now(timezone.utc)
        stars = []
        for page in (1, 2):
            batch = self._get(f"{API}/users/{login}/starred", accept=STAR_ACCEPT,
                              params={"per_page": 100, "page": page})
            if batch is None:
                return None if page == 1 else (stars, [], captured)
            stars.extend((s["starred_at"], repo_row(s["repo"], captured)) for s in batch if "repo" in s)
            if len(batch) < 100:
                break
        own_batch = self._get(f"{API}/users/{login}/repos",
                              params={"per_page": 100, "sort": "pushed"}) or []
        own = [repo_row(r, captured) for r in own_batch if not r.get("private")]
        return stars, own, captured

    def _vector(self, stars, own, asof):
        from unstarred_features import scalars_vector

        hist = [""] * self.config["hist"]
        for i, (_, r) in enumerate(stars[: self.config["hist"]]):  # API returns newest first
            hist[i] = str(r["repo_id"])
        u_num = scalars_vector([r for _, r in stars], own, asof)
        out = self.fn(
            hist_ids=self.tf.constant([hist]),
            user_num=self.tf.constant([u_num], dtype=self.tf.float32),
        )
        return list(out.values())[0].numpy()[0].tolist()

    def predict(self, inputs):
        if isinstance(inputs, dict):
            inputs = [inputs]
        out = []
        for inst in inputs:
            login = (inst.get("login") or "").strip().lstrip("@")
            if not login:
                out.append({"error": "no login given"})
                continue
            try:
                pulled = self._pull(login)
            except requests.RequestException as e:
                out.append({"login": login, "error": f"github unreachable: {e}"})
                continue
            if pulled is None:
                out.append({"login": login, "error": "user not found"})
                continue
            stars, own, captured = pulled
            if not stars and not own:
                out.append({"login": login, "error": "no public stars or repos to read taste from"})
                continue
            vec = self._vector(stars, own, captured)
            langs = {}
            for _, r in stars:
                if r["language"] != "none":
                    langs[r["language"]] = langs.get(r["language"], 0) + 1
            out.append({
                "login": login,
                "user_vector": vec,
                "starred_ids": [int(r["repo_id"]) for _, r in stars],
                "profile": {
                    "n_stars_pulled": len(stars),
                    "top_languages": sorted(langs, key=langs.get, reverse=True)[:5],
                    "recent_stars": [
                        {"full_name": r["full_name"], "language": r["language"],
                         "stars": r["stars"], "description": r["description"][:140]}
                        for _, r in stars[:30]
                    ],
                    "own_repos": [
                        {"full_name": r["full_name"], "language": r["language"],
                         "stars": r["stars"], "description": r["description"][:140]}
                        for r in own[:15]
                    ],
                },
            })
        return {"predictions": out}

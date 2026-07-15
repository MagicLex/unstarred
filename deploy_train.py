"""Register the T1 job `train-towers` as a Hopsworks PYTHON job.

tensorflow-training-pipeline env, 16GB (the joined training frame is large).
Controls run as: hops job run train-towers -- --shuffle-labels / --drop fingerprints
Run: hops job run train-towers
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "train-towers"
ENV_NAME = "tensorflow-training-pipeline"

_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]


def main() -> None:
    project = hopsworks.login()
    ja = project.get_job_api()

    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = f"hdfs:///Projects/{project.name}/{_rel}/train_towers.py"
    cfg["environmentName"] = ENV_NAME
    cfg["files"] = f"hdfs:///Projects/{project.name}/{_rel}/unstarred_features.py"
    cfg["resourceConfig"]["memory"] = 16384

    job = ja.get_job(JOB_NAME)
    if job is not None:  # config property lost its setter; recreate instead
        job.delete()
        print(f"deleted stale {JOB_NAME}", flush=True)
    job = ja.create_job(JOB_NAME, cfg)
    print(f"created job {job.name} on {ENV_NAME}", flush=True)


if __name__ == "__main__":
    main()

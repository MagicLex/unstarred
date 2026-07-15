"""Register the I1 job `embed-candidates` as a Hopsworks PYTHON job.

tensorflow-training-pipeline env (needs TF to run the candidate tower).
Run after every retrain: hops job run embed-candidates
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "embed-candidates"
ENV_NAME = "tensorflow-training-pipeline"

_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]


def main() -> None:
    project = hopsworks.login()
    ja = project.get_job_api()

    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = f"hdfs:///Projects/{project.name}/{_rel}/embed_candidates.py"
    cfg["environmentName"] = ENV_NAME
    cfg["resourceConfig"]["memory"] = 8192

    job = ja.get_job(JOB_NAME)
    if job is not None:  # config property lost its setter; recreate instead
        job.delete()
        print(f"deleted stale {JOB_NAME}", flush=True)
    job = ja.create_job(JOB_NAME, cfg)
    print(f"created job {job.name} on {ENV_NAME}", flush=True)


if __name__ == "__main__":
    main()

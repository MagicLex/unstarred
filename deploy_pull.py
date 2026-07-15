"""Register the F2 job `pull-interactions` as a Hopsworks PYTHON job.

Needs unstarred_features.py alongside the script: shipped via files. The pull is
overnight-sized (rate limit bound); the job is resumable via the checkpoint.
Run: hops job run pull-interactions
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "pull-interactions"
ENV_NAME = "python-feature-pipeline"

_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]
_data = str(_here.parent / "data")


def main() -> None:
    project = hopsworks.login()
    ja = project.get_job_api()

    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = f"hdfs:///Projects/{project.name}/{_rel}/pull_interactions.py"
    cfg["environmentName"] = ENV_NAME
    cfg["defaultArgs"] = f"--data-dir {_data}"
    cfg["files"] = f"hdfs:///Projects/{project.name}/{_rel}/unstarred_features.py"
    cfg["resourceConfig"]["memory"] = 8192

    job = ja.get_job(JOB_NAME)
    if job is not None:  # config property lost its setter; recreate instead
        job.delete()
        print(f"deleted stale {JOB_NAME}", flush=True)
    job = ja.create_job(JOB_NAME, cfg)
    print(f"created job {job.name} on {ENV_NAME}", flush=True)


if __name__ == "__main__":
    main()

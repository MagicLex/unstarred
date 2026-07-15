"""Register the F3 job `fingerprint-readmes` as a Hopsworks PYTHON job.

Runs on the unstarred-pipeline env (sentence-transformers + anthropic).
Run: hops job run fingerprint-readmes
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "fingerprint-readmes"
ENV_NAME = "unstarred-pipeline"

_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]
_data = str(_here.parent / "data")


def main() -> None:
    project = hopsworks.login()
    ja = project.get_job_api()

    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = f"hdfs:///Projects/{project.name}/{_rel}/fingerprint_readmes.py"
    cfg["environmentName"] = ENV_NAME
    cfg["defaultArgs"] = f"--data-dir {_data}"
    cfg["resourceConfig"]["memory"] = 8192

    job = ja.get_job(JOB_NAME)
    if job is not None:  # config property lost its setter; recreate instead
        job.delete()
        print(f"deleted stale {JOB_NAME}", flush=True)
    job = ja.create_job(JOB_NAME, cfg)
    print(f"created job {job.name} on {ENV_NAME}", flush=True)


if __name__ == "__main__":
    main()

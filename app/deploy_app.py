"""Deploy unstarred as a custom (FastAPI) Hopsworks app.

Runs on the unstarred-pipeline env (fastapi + sentence-transformers +
anthropic). The pod reads candidate embeddings once at boot and calls the
unstarredquery deployment per request; the Anthropic key (project secret)
powers dossier + librarian.

Redeploy uses the recovery sequence (stop, purge lingering k8s deployment,
drain, stop zombie executions) since app.stop() returns before the execution
dies.
"""
import subprocess
import time
from pathlib import Path

import hopsworks

APP_NAME = "unstarred"
ENV_NAME = "unstarred-pipeline"

_here = Path(__file__).resolve()
rel = str(_here).split("/hopsfs/", 1)[1]
APP_PATH = str(Path(rel).parent / "server.py")
SERVER = f"/hopsfs/{rel.rsplit('/', 1)[0]}/server.py"
ENTRYPOINT = f'bash -lc "exec python {SERVER}"'


def _is_app(name: str) -> bool:
    # substring match on APP_NAME alone also catches the unstarredquery
    # predictor pods; require the pythonapp prefix + exact app segment
    return name.startswith("pythonapp") and f"--{APP_NAME}-" in name or name == f"pythonapp-deadair--{APP_NAME}"


def _pods():
    out = subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True).stdout
    return [l.split()[0] for l in out.splitlines() if _is_app(l.split()[0])]


def _purge_k8s():
    out = subprocess.run(["kubectl", "get", "deployment"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if _is_app(line.split()[0]):
            name = line.split()[0]
            subprocess.run(["kubectl", "delete", "deployment", name], capture_output=True)
            print(f"purged k8s deployment {name}", flush=True)
    for _ in range(60):
        if not _pods():
            return
        time.sleep(5)
    raise RuntimeError("app pods refused to drain")


def _stop_zombies(project):
    job = project.get_job_api().get_job(APP_NAME)
    if job is None:
        return
    for ex in job.get_executions() or []:
        if ex.final_status in ("UNDEFINED", None):
            try:
                ex.stop()
            except Exception:
                pass


def _create(apps):
    return apps.create_app(
        name=APP_NAME, app_path=APP_PATH, app_kind="CUSTOM",
        entrypoint_command=ENTRYPOINT, app_port=8000,
        environment=ENV_NAME, memory=8192, cores=1.0,
        description="unstarred -- which repos would you have starred already, if you "
                    "had seen them? Two-tower recommender over public star histories, "
                    "an LLM dossier, and a librarian you can talk to.")


def main():
    project = hopsworks.login()
    apps = project.get_app_api()
    print(f"app_path={APP_PATH} env={ENV_NAME}", flush=True)
    app = apps.get_app(APP_NAME)
    if app is not None:
        try:
            app.stop()
        except Exception:
            pass
        _purge_k8s()
        _stop_zombies(project)
        try:
            app.delete()
        except Exception as e:
            print(f"delete: {e}", flush=True)
        for _ in range(24):
            if apps.get_app(APP_NAME) is None:
                break
            time.sleep(5)
        time.sleep(10)
    app = _create(apps)
    app.run(await_serving=True)
    print(f"URL: {app.app_url}", flush=True)


if __name__ == "__main__":
    main()

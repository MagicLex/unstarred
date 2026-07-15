"""Register the I2 deployment `unstarredquery` (KServe, query tower).

Uploads predictor.py next to the latest model's files, deploys on the
tensorflow-inference-pipeline env with platform inference logging, starts it.
Run: python deploy_serving.py
"""
from __future__ import annotations

from pathlib import Path

import hopsworks
from hsml.inference_logger import InferenceLogger
from hsml.resources import PredictorResources, Resources
from hsml.scaling_config import PredictorScalingConfig, ScaleMetric

DEPLOY_NAME = "unstarredquery"
ENV_NAME = "tensorflow-inference-pipeline"

_here = Path(__file__).resolve().parent


def main() -> None:
    project = hopsworks.login()
    mr = project.get_model_registry()
    model = max(mr.get_models("unstarred"), key=lambda m: m.version)
    print(f"deploying unstarred v{model.version}", flush=True)

    script_dir = f"/Projects/{project.name}/Models/{model.name}/{model.version}/Files"
    project.get_dataset_api().upload(str(_here / "predictor.py"), script_dir, overwrite=True)

    ms = project.get_model_serving()
    existing = ms.get_deployment(DEPLOY_NAME)
    if existing is not None:
        existing.stop(await_stopped=180)
        existing.delete()
        print("deleted stale deployment", flush=True)

    deployment = model.deploy(
        name=DEPLOY_NAME,
        description="unstarred query tower: login -> live GitHub pull -> user vector + profile",
        script_file=f"{script_dir}/predictor.py",
        resources=PredictorResources(
            requests=Resources(cores=1, memory=2048, gpus=0),
            limits=Resources(cores=2, memory=4096, gpus=0),
        ),
        scaling_configuration=PredictorScalingConfig(
            min_instances=1, max_instances=2,
            scale_metric=ScaleMetric.CONCURRENCY, target=20,
        ),
        environment=ENV_NAME,
        inference_logger=InferenceLogger(mode="ALL"),
    )
    deployment.start(await_running=600)
    print(f"running: {deployment.get_inference_url()}", flush=True)


if __name__ == "__main__":
    main()

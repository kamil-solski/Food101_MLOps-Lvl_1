# src/utils/onnx_registry.py
import os
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

# Optional flavors used below
import onnx
import mlflow.onnx
import mlflow.pytorch
import torch


AUTO_PROMOTE_IF_NO_CHAMPION = os.getenv("AUTO_PROMOTE_IF_NO_CHAMPION", "true").lower() in {"1","true","yes"}


def _bootstrap_champion_if_absent(client: MlflowClient, model_name: str, version: str) -> None:
    """If no 'champion' alias exists, set it to the given version (first-ever bootstrap)."""
    if not AUTO_PROMOTE_IF_NO_CHAMPION:
        print("[INFO] Bootstrap disabled via AUTO_PROMOTE_IF_NO_CHAMPION=false")
        return
    try:
        client.get_model_version_by_alias(name=model_name, alias="champion")
        print("[INFO] 'champion' already exists → skip bootstrap.")
    except RestException:
        client.set_registered_model_alias(name=model_name, version=version, alias="champion")
        # Tag for auditability (optional)
        client.set_model_version_tag(name=model_name, version=version, key="bootstrap", value="true")
        client.set_model_version_tag(name=model_name, version=version, key="bootstrap_reason", value="no_champion_existing")
        print(f"[INFO] Bootstrapped 'champion' → version {version}.")


def _artifact_exists(run_id: str, artifact_rel_path: str) -> bool:
    try:
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_rel_path)
        return True
    except Exception:
        return False


def _export_onnx_from_pytorch_run(run_id: str, image_size: int, input_name: str, output_name: str, opset: int) -> str:
    """
    Load the PyTorch model logged at runs:/<run_id>/model and export to a local ONNX file.
    Returns the local ONNX path.
    """
    pt_model = mlflow.pytorch.load_model(model_uri=f"runs:/{run_id}/model")
    pt_model.eval()

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    tmp_dir = Path("outputs/checkpoints"); tmp_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = tmp_dir / f"{run_id}_export.onnx"

    torch.onnx.export(
        pt_model,
        dummy,
        onnx_path.as_posix(),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}}
    )
    return onnx_path.as_posix()


def ensure_onnx_and_register(
    run_id: str,
    registry_name: str,
    *,
    image_size: int,
    input_name: str = "images",
    output_name: str = "logits",
    opset: int = 13,
    await_registration_for: int = 300
) -> str:
    """
    Ensures the given run has an MLflow ONNX model logged at 'onnx_model' and registers it
    under `registry_name`. Returns the newly created registered model version (as a string).

    Steps:
      1) If 'onnx_model/MLmodel' exists -> reuse it.
      2) Else if any .onnx artifact exists -> load & re-log as ONNX flavor under 'onnx_model'.
      3) Else export from the logged PyTorch model ('runs:/<run_id>/model'), then log as ONNX flavor.
      4) Register 'runs:/<run_id>/onnx_model' and return version.
    """
    client = MlflowClient()

    # 1) ONNX already logged as an MLflow model?
    if _artifact_exists(run_id, "onnx_model/MLmodel"):
        model_uri = f"runs:/{run_id}/onnx_model"
        print(f"[INFO] Reusing existing MLflow ONNX model at {model_uri}")
        result = mlflow.register_model(model_uri=model_uri, name=registry_name)
        return str(result.version)

    # 2) Look for any raw .onnx in artifacts of this run
    def _list_all_artifacts(rid):
        acc = []
        def walk(p=""):
            for it in client.list_artifacts(rid, p):
                if it.is_dir:
                    walk(it.path)
                else:
                    acc.append(it.path)
        walk("")
        return acc

    all_paths = _list_all_artifacts(run_id)
    raw_onnx_rel = next((p for p in all_paths if p.endswith(".onnx")), None)

    if raw_onnx_rel:
        local_raw = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=raw_onnx_rel)
        with mlflow.start_run(run_id=run_id):
            mlflow.onnx.log_model(
                onnx_model=onnx.load(local_raw),
                artifact_path="onnx_model",
                registered_model_name=None  # register in step below
            )

        model_uri = f"runs:/{run_id}/onnx_model"
        result = mlflow.register_model(model_uri=model_uri, name=registry_name)
        return str(result.version)

    # 3) No ONNX logged yet → export from PyTorch and log as ONNX
    local_export = _export_onnx_from_pytorch_run(
        run_id=run_id, image_size=image_size, input_name=input_name, output_name=output_name, opset=opset
    )
    with mlflow.start_run(run_id=run_id):
        mlflow.onnx.log_model(
            onnx_model=onnx.load(local_export),
            artifact_path="onnx_model",
            registered_model_name=None
        )

    model_uri = f"runs:/{run_id}/onnx_model"
    result = mlflow.register_model(model_uri=model_uri, name=registry_name, await_registration_for=await_registration_for)
    return str(result.version)


def set_aliases_after_register(client: MlflowClient, model_name: str, version: str) -> None:
    """Set 'challenger' to this version and bootstrap 'champion' if absent."""
    client.set_registered_model_alias(name=model_name, version=version, alias="challenger")
    _bootstrap_champion_if_absent(client, model_name, version)

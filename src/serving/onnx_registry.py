# src/utils/onnx_registry.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import numpy as np
import json, hashlib
import torch
import onnx
import mlflow
import mlflow.onnx
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException, RestException


# Small helpers
def _list_artifacts_flat(client: MlflowClient, run_id: str, path: str = "") -> List[str]:
    out: List[str] = []
    for it in client.list_artifacts(run_id, path):
        if it.is_dir:
            out.extend(_list_artifacts_flat(client, run_id, it.path))
        else:
            out.append(it.path)
    return out

def _get_alias_version(client: MlflowClient, name: str, alias: str) -> Optional[str]:
    try:
        return str(client.get_model_version_by_alias(name=name, alias=alias).version)
    except (MlflowException, RestException):
        return None


# ONNX export + register
def _export_onnx_from_pytorch_run(
    run_id: str,
    image_size: int,
    input_name: str = "images",
    output_name: str = "logits",
    opset: int = 13,
    device: str = "cpu",
) -> str:
    """Load PyTorch model from MLflow run and export ONNX locally (CPU by default)."""
    model = mlflow.pytorch.load_model(model_uri=f"runs:/{run_id}/model")
    model.to(device).eval()

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32, device=device)

    out_dir = Path("outputs/checkpoints"); out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{run_id}_export.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path.as_posix(),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset,
            dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
            dynamo=True,
            external_data=True,  # just to ensure that *.data file is created (mlflow requires it when loading)
        )
    return onnx_path.as_posix()


def ensure_onnx_and_register(
    run_id: str,
    registry_name: str,
    *,
    image_size: int,
    class_names: list[str] | None = None,
    input_name: str = "images",
    output_name: str = "logits",
    opset: int = 13,
    await_registration_for: int = 300,
) -> str:
    """
    Ensure the run has an ONNX MLflow model at 'onnx_model', then register it.
    Returns the registered model version (string).
      1) Reuse 'onnx_model' if already logged.
      2) Else, if a raw .onnx exists, re-log it as 'onnx_model'.
      3) Else, export from PyTorch, then log as 'onnx_model'.
    """
    client = MlflowClient()
    paths = _list_artifacts_flat(client, run_id)

    # 1) Reuse 'onnx_model'
    if "onnx_model/MLmodel" in paths:
        result = mlflow.register_model(model_uri=f"runs:/{run_id}/onnx_model",
                                       name=registry_name,
                                       await_registration_for=await_registration_for)
        return str(result.version)

    # 2) Re-log first found raw .onnx (if any)
    raw_onnx_rel = next((p for p in paths if p.endswith(".onnx")), None)
    if raw_onnx_rel:
        local_onnx = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=raw_onnx_rel)
    else:
        # 3) Export from PyTorch
        local_onnx = _export_onnx_from_pytorch_run(
            run_id=run_id,
            image_size=image_size,
            input_name=input_name,
            output_name=output_name,
            opset=opset,
            device="cpu",
        )

    # Log ONNX flavor under 'onnx_model' (with a simple input_example)
    input_example = {input_name: np.zeros((1, 3, image_size, image_size), dtype=np.float32)}
    with mlflow.start_run(run_id=run_id):
        mlflow.onnx.log_model(
            onnx_model=onnx.load(local_onnx),
            name="onnx_model",
            input_example=input_example,
            registered_model_name=None,
        )
        if class_names:
            mlflow.log_dict({"classes": class_names}, artifact_file="onnx_model/labels.json")

    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/onnx_model",
        name=registry_name,
        await_registration_for=await_registration_for
    )
    # TODO: version tags for classes with hashlib and set_model_version_tag   
     
    return str(result.version)


# Aliasing (mutually exclusive)
def ensure_alias_post_register(client: MlflowClient, model_name: str, version: str) -> str:
    """
    Call immediately after registering a new version:
      - If no champion exists -> set champion=version (bootstrap).
      - Else -> set challenger=version.
    Returns the alias that was set.
    """
    champ = _get_alias_version(client, model_name, "champion")
    if champ is None:
        client.set_registered_model_alias(name=model_name, version=str(version), alias="champion")
        return "champion"
    if champ != str(version):
        client.set_registered_model_alias(name=model_name, version=str(version), alias="challenger")
        return "challenger"
    return "champion"  # already champion; nothing to change


def promote_challenger_to_champion(
    client: MlflowClient,
    model_name: str,
    *,
    keep_prev_alias: bool = True,
    clear_challenger: bool = True,
) -> Optional[dict]:
    """
    Promote current challenger -> champion.
    Optionally mark previous champion as 'prev_champion' and clear 'challenger'.
    """
    chal = _get_alias_version(client, model_name, "challenger")
    if chal is None:
        print("[INFO] No challenger to promote.")
        return None

    old = _get_alias_version(client, model_name, "champion")
    client.set_registered_model_alias(name=model_name, version=str(chal), alias="champion")

    if keep_prev_alias and old and old != chal:
        client.set_registered_model_alias(name=model_name, version=str(old), alias="prev_champion")

    if clear_challenger:
        try:
            client.delete_registered_model_alias(name=model_name, alias="challenger")
        except (MlflowException, RestException):
            pass

    return {"old_champion": old, "new_champion": chal}

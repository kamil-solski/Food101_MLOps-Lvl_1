from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import os, json, yaml
import onnxruntime as ort
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException, RestException
from src.utils.paths import MLFLOW_TRACKING_DIR

with open("inference_api/config.yaml", "r") as f:
    SERVE_CFG = yaml.safe_load(f)

mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())

def _get_by_alias(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name=name, alias=alias)
    except (MlflowException, RestException):
        return None

def _latest_version(client: MlflowClient, name: str):
    vers = client.search_model_versions(f"name = '{name}'")
    if not vers:
        return None
    vers.sort(key=lambda v: int(v.last_updated_timestamp or 0), reverse=True)
    return vers[0]

def _try_download(run_id: str, rel_path: str) -> Optional[str]:
    try:
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=rel_path)
    except Exception:
        return None

def _download_run_onnx_dir(run_id: str) -> Tuple[str, str]:
    """
    Download the run's 'onnx_model' directory so external .data files are present.
    Returns (onnx_dir, model_path).
    """
    onnx_dir = _try_download(run_id, "onnx_model")
    if onnx_dir and os.path.isdir(onnx_dir):
        model_path = os.path.join(onnx_dir, "model.onnx")
        if not os.path.exists(model_path):
            # very old runs: pick first .onnx inside
            for root, _, files in os.walk(onnx_dir):
                for f in files:
                    if f.endswith(".onnx"):
                        model_path = os.path.join(root, f)
                        break
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model.onnx found under {onnx_dir}")
        return onnx_dir, model_path

    # optional legacy fallbacks (single file)
    for p in ("onnx_model/model.onnx", "model.onnx", "outputs/checkpoints/model.onnx"):
        mp = _try_download(run_id, p)
        if mp and os.path.exists(mp):
            return os.path.dirname(mp), mp

    raise FileNotFoundError(f"No ONNX found for run {run_id}. Ensure training logs 'onnx_model/model.onnx'.")

def _load_labels_from_run_dir(onnx_dir: str) -> List[str]:
    """Load labels strictly from run artifacts (labels.json → labels.txt)."""
    pj = os.path.join(onnx_dir, "labels.json")
    if os.path.exists(pj):
        try:
            with open(pj, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("classes"), list):
                return [str(x) for x in obj["classes"]]
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
    pt = os.path.join(onnx_dir, "labels.txt")
    if os.path.exists(pt):
        try:
            with open(pt, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        except Exception:
            pass
    return []

def resolve_and_load() -> Tuple[ort.InferenceSession, Dict[str, Any]]:
    client = MlflowClient()
    model_name = os.getenv("MODEL_REGISTRY_NAME", SERVE_CFG.get("model_registry_name", "best_model"))
    pref_alias = os.getenv("PREFERRED_ALIAS", SERVE_CFG.get("preferred_alias", "champion"))
    fb_alias   = os.getenv("FALLBACK_ALIAS",  SERVE_CFG.get("fallback_alias", "challenger"))

    cand_pref = _get_by_alias(client, model_name, pref_alias)
    cand_fb   = _get_by_alias(client, model_name, fb_alias)
    chosen    = cand_pref or cand_fb or _latest_version(client, model_name)
    if not chosen:
        raise RuntimeError(f"No versions found in registry '{model_name}'. Train & register a model first.")

    alias = "champion" if (cand_pref and chosen.version == cand_pref.version) else \
            "challenger" if (cand_fb and chosen.version == cand_fb.version) else "latest"

    # Always pull artifacts from the SOURCE RUN (labels live there)
    onnx_dir, onnx_path = _download_run_onnx_dir(chosen.run_id)

    providers = SERVE_CFG.get("providers") or ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    classes = _load_labels_from_run_dir(onnx_dir)  # <- only from source run

    meta = {
        "runtime": "ort",
        "model_name": model_name,
        "version": str(chosen.version),
        "alias": alias,
        "run_id": chosen.run_id,
        "onnx_path": onnx_path,
        "image_size": int(SERVE_CFG.get("image_size", 224)),
        "top_k": int(SERVE_CFG.get("top_k", 5)),
        "output_is_logits": bool(SERVE_CFG.get("output_is_logits", True)),
        "preprocess": SERVE_CFG.get("preprocess", {}),
        "channels": SERVE_CFG.get("channels", "rgb"),
        "input_name": session.get_inputs()[0].name,
        "classes": classes,
    }
    return session, meta
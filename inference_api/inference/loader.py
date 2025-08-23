from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os, json, yaml

import onnx
import onnxruntime as ort
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.paths import MLFLOW_TRACKING_DIR
from inference_api.utils.helpers import _get_by_alias, _latest_version

# Load serving config (providers, image_size, top_k, etc.)
with open("inference_api/config.yaml", "r") as f:
    SERVE_CFG = yaml.safe_load(f)

mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())



# ---------- Artifacts ----------
def _try_download(run_id: str, path: str) -> str | None:
    try:
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
    except Exception:
        return None

def _download_onnx(client: MlflowClient, run_id: str) -> str:
    # 1) Preferred: download the whole MLflow ONNX flavor directory
    onnx_dir = _try_download(run_id, "onnx_model")
    if onnx_dir and os.path.isdir(onnx_dir):
        # standard name from mlflow.onnx flavor
        candidate = os.path.join(onnx_dir, "model.onnx")
        if os.path.exists(candidate):
            return candidate
        # fallback: first .onnx inside onnx_dir (rarely needed)
        for root, _, files in os.walk(onnx_dir):
            for f in files:
                if f.endswith(".onnx"):
                    return os.path.join(root, f)

    # 2) Legacy fallbacks (if you ever had old runs)
    for p in ("onnx_model/model.onnx", "model.onnx", "outputs/checkpoints/model.onnx"):
        local = _try_download(run_id, p)
        if local and os.path.exists(local):
            return local

    raise FileNotFoundError(
        f"No ONNX found at expected locations for run {run_id}. "
        f"Ensure the run logs 'onnx_model/model.onnx' with mlflow.onnx.log_model."
    )



# ---------- ONNX metadata (labels) ----------
def _classes_from_onnx_metadata(onnx_path: str) -> List[str]:
    """
    Extract class names from ONNX metadata_props.
    Supported keys (string values):
      - 'classes_json' -> JSON list or {"classes": [...]}
      - 'classes'      -> JSON list OR comma/semicolon-separated string
      - 'labels_json'  -> JSON list
    Returns [] if not found or unparsable.
    """
    try:
        m = onnx.load(onnx_path)
    except Exception:
        return []

    props = {p.key: p.value for p in m.metadata_props}

    # 1) JSON payloads
    for k in ("classes_json", "labels_json"):
        if k in props:
            try:
                obj = json.loads(props[k])
                if isinstance(obj, dict) and "classes" in obj and isinstance(obj["classes"], list):
                    return [str(x) for x in obj["classes"]]
                if isinstance(obj, list):
                    return [str(x) for x in obj]
            except Exception:
                pass

    # 2) Simple 'classes' key: JSON list or delimited string
    if "classes" in props:
        val = props["classes"]
        # try JSON
        try:
            obj = json.loads(val)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            # split by common delimiters
            for sep in (",", ";", "|"):
                if sep in val:
                    return [s.strip() for s in val.split(sep) if s.strip()]

    return []


# ---------- Public API ----------
def resolve_and_load() -> Tuple[ort.InferenceSession, Dict[str, Any]]:
    """
    Resolve model (champion -> challenger -> latest), download ONNX, build ORT session,
    and extract class names from ONNX metadata only.
    """
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

    onnx_path = _download_onnx(client, chosen.run_id)

    providers = SERVE_CFG.get("providers") or ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    classes = _classes_from_onnx_metadata(onnx_path)

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
        "classes": classes,              # <- labels from ONNX metadata only
    }
    return session, meta

import torch
import argparse
from src.models import architectures
from src.utils.paths import get_paths

def export_model_to_onnx(model_name: str, arch_name: str, hidden_units: int, num_classes: int, image_size: int, fold: str = None):
    paths = get_paths(fold=fold, model_name=model_name)
    checkpoint_path = paths["MODEL_CHECKPOINT_PATH"]
    onnx_export_path = checkpoint_path.with_suffix(".onnx")

    model_class = architectures[arch_name]
    model = model_class(input_shape=3, hidden_units=hidden_units, output_shape=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(model, dummy_input, onnx_export_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)

    print(f"[✔] Exported ONNX model to: {onnx_export_path}")
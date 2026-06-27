"""
Run this script once to export DINOv2 to ONNX format.
Requires: pip install torch transformers
"""
import torch
from transformers import AutoModel

MODEL_PATH = r"f:\E\SourceDuplicateImages\DinoDuplicateSearch\models--facebook--dinov2-base"
OUTPUT = "Models/dinov2-base.onnx"

model = AutoModel.from_pretrained(MODEL_PATH)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    OUTPUT,
    opset_version=17,
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "last_hidden_state": {0: "batch_size"}}
)

print(f"Model exported to {OUTPUT}")

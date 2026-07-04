"""
Run this script once to export DINOv2 to ONNX format with register tokens.
Register tokens fix the high-norm outlier token problem, improving retrieval accuracy.
Requires: pip install torch transformers
"""
import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_PATH = r"f:\E\SourceDuplicateImages\DinoDuplicateSearch\models--facebook--dinov2-base"
OUTPUT = "Models/dinov2-base.onnx"
NUM_REGISTER_TOKENS = 4


class DINOv2WithRegisters(nn.Module):
    """DINOv2 with register tokens prepended to the sequence."""

    def __init__(self, base_model, num_register_tokens=4):
        super().__init__()
        self.base_model = base_model
        self.num_register_tokens = num_register_tokens
        embed_dim = base_model.config.hidden_size
        self.register_tokens = nn.Parameter(
            torch.randn(1, num_register_tokens, embed_dim) * 0.02
        )

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values, output_hidden_states=False)
        cls_token = outputs.last_hidden_state[:, :1, :]
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        batch_size = pixel_values.shape[0]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        full_sequence = torch.cat([cls_token, register_tokens, patch_tokens], dim=1)
        return full_sequence


model = AutoModel.from_pretrained(MODEL_PATH)
model.eval()

wrapper = DINOv2WithRegisters(model, NUM_REGISTER_TOKENS)
wrapper.eval()

dummy = torch.randn(1, 3, 224, 224)

# Verify output shape: CLS(1) + registers(4) + patches(196) = 201 tokens
with torch.no_grad():
    out = wrapper(dummy)
    assert out.shape == (1, 201, 768), f"Unexpected shape: {out.shape}"
    print(f"Output shape: {out.shape} (CLS + {NUM_REGISTER_TOKENS} registers + 196 patches)")

torch.onnx.export(
    wrapper,
    dummy,
    OUTPUT,
    opset_version=17,
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "last_hidden_state": {0: "batch_size"}}
)

print(f"Model exported to {OUTPUT}")
print(f"Register tokens: {NUM_REGISTER_TOKENS}")
print(f"CLS token index: 0, Register tokens: 1-{NUM_REGISTER_TOKENS}, Patches: {NUM_REGISTER_TOKENS+1}-{NUM_REGISTER_TOKENS+196}")

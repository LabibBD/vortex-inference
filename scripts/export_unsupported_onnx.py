from pathlib import Path

import torch
import torch.nn as nn


class UnsupportedModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(x)


model = UnsupportedModel().eval()

sample = torch.randn(
    2,
    8,
)

path = Path(
    "models/phase16_unsupported.onnx"
)

torch.onnx.export(
    model,
    sample,
    path,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    dynamo=False,
)

print(
    f"[VORTEX] Exported {path}"
)

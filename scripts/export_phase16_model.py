from pathlib import Path

import torch
import torch.nn as nn


class Phase16MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


def main() -> None:
    torch.manual_seed(42)

    model = Phase16MLP()
    model.eval()

    sample = torch.randn(
        2,
        8,
        dtype=torch.float32,
    )

    output_path = Path(
        "models/phase16_mlp.onnx"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["input"],
        output_names=["probabilities"],
        opset_version=18,
        dynamo=False,
    )

    print(
        f"[VORTEX] Exported {output_path}"
    )


if __name__ == "__main__":
    main()

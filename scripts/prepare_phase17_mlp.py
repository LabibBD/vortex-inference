from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


BATCH = 128
INPUT = 784


class VortexMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 10)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return self.softmax(x)


def main() -> None:
    torch.manual_seed(1701)

    model = VortexMLP()
    model.eval()

    # Deterministic input.
    x = torch.linspace(
        -1.0,
        1.0,
        steps=BATCH * INPUT,
        dtype=torch.float32,
    ).reshape(BATCH, INPUT)

    with torch.inference_mode():
        reference = model(x)

    models = Path("models")
    models.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        models / "phase17_mlp.pt",
    )

    x.numpy().astype(
        np.float32
    ).tofile(
        models / "phase17_input.bin"
    )

    reference.numpy().astype(
        np.float32
    ).tofile(
        models / "phase17_reference.bin"
    )

    predictions = (
        reference.argmax(dim=1)
        .cpu()
        .numpy()
    )

    np.savetxt(
        models / "phase17_predictions.txt",
        predictions,
        fmt="%d",
    )

    torch.onnx.export(
        model,
        x,
        models / "phase17_mlp.onnx",
        input_names=["input"],
        output_names=["probabilities"],
        opset_version=18,
        dynamo=False,
    )

    print(
        "================================="
    )
    print(
        " VORTEX Phase-17 Model"
    )
    print(
        "=================================\n"
    )

    print(
        f"Input:  {tuple(x.shape)}"
    )

    print(
        f"Output: {tuple(reference.shape)}"
    )

    print(
        "Architecture: "
        "784 -> 512 -> 256 -> 10"
    )

    print(
        "First 10 predictions:",
        predictions[:10].tolist(),
    )

    print(
        "Max softmax row error:",
        float(
            torch.max(
                torch.abs(
                    reference.sum(dim=1)
                    - 1.0
                )
            )
        ),
    )

    print(
        "\n[VORTEX] Phase-17 model exported"
    )


if __name__ == "__main__":
    main()

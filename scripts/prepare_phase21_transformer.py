from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


BATCH = 4
TOKENS = 16
MODEL_DIM = 32
FF_DIM = 64


class TinyTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.q_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )

        self.k_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )

        self.v_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )

        self.out_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )

        self.norm1 = nn.LayerNorm(
            MODEL_DIM,
            eps=1e-5,
        )

        self.ff1 = nn.Linear(
            MODEL_DIM,
            FF_DIM,
        )

        self.ff2 = nn.Linear(
            FF_DIM,
            MODEL_DIM,
        )

        self.norm2 = nn.LayerNorm(
            MODEL_DIM,
            eps=1e-5,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        scores = (
            query @ key.transpose(-2, -1)
        ) / (MODEL_DIM ** 0.5)

        probabilities = torch.softmax(
            scores,
            dim=-1,
        )

        context = probabilities @ value

        attention_output = self.out_proj(
            context
        )

        normalized = self.norm1(
            x + attention_output
        )

        feed_forward = self.ff2(
            functional.gelu(
                self.ff1(normalized),
                approximate="tanh",
            )
        )

        return self.norm2(
            normalized +
            feed_forward
        )


def save_tensor(
    path: Path,
    tensor: torch.Tensor,
    transpose: bool = False,
) -> None:
    value = tensor.detach().cpu()

    if transpose:
        value = value.transpose(
            0,
            1,
        )

    array = (
        value.contiguous()
        .numpy()
        .astype(np.float32)
    )

    array.tofile(path)


def main() -> None:
    torch.manual_seed(2101)

    model = TinyTransformerBlock()
    model.eval()

    input_tensor = torch.linspace(
        -1.0,
        1.0,
        steps=(
            BATCH *
            TOKENS *
            MODEL_DIM
        ),
        dtype=torch.float32,
    ).reshape(
        BATCH,
        TOKENS,
        MODEL_DIM,
    )

    with torch.inference_mode():
        reference = model(
            input_tensor
        )

    root = Path(
        "models/phase21_transformer"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_tensor(
        root / "input.bin",
        input_tensor,
    )

    save_tensor(
        root / "reference.bin",
        reference,
    )

    linear_layers: Dict[
        str,
        nn.Linear,
    ] = {
        "q": model.q_proj,
        "k": model.k_proj,
        "v": model.v_proj,
        "out": model.out_proj,
        "ff1": model.ff1,
        "ff2": model.ff2,
    }

    for name, layer in linear_layers.items():
        # PyTorch: [out_features, in_features]
        # VORTEX:  [in_features, out_features]
        save_tensor(
            root / f"{name}_weight.bin",
            layer.weight,
            transpose=True,
        )

        save_tensor(
            root / f"{name}_bias.bin",
            layer.bias,
        )

    save_tensor(
        root / "norm1_gamma.bin",
        model.norm1.weight,
    )

    save_tensor(
        root / "norm1_beta.bin",
        model.norm1.bias,
    )

    save_tensor(
        root / "norm2_gamma.bin",
        model.norm2.weight,
    )

    save_tensor(
        root / "norm2_beta.bin",
        model.norm2.bias,
    )

    print(
        "================================="
    )

    print(
        " VORTEX Phase-21 Transformer"
    )

    print(
        "=================================\n"
    )

    print(
        f"Input shape:  "
        f"{tuple(input_tensor.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(reference.shape)}"
    )

    print(
        f"Model dimension: {MODEL_DIM}"
    )

    print(
        f"FFN dimension:   {FF_DIM}"
    )

    print(
        "All values finite:",
        bool(
            torch.isfinite(reference)
            .all()
        ),
    )

    print(
        "Output mean:",
        float(reference.mean()),
    )

    print(
        "Output std:",
        float(
            reference.std(
                unbiased=False
            )
        ),
    )

    print(
        "\n[VORTEX] Transformer reference generated"
    )


if __name__ == "__main__":
    main()

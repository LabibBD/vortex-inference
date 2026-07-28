#!/usr/bin/env python3

from pathlib import Path
from typing import Callable

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

        self.q_proj = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.k_proj = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.v_proj = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.out_proj = nn.Linear(MODEL_DIM, MODEL_DIM)

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

        hidden = self.ff1(
            normalized
        )

        hidden = functional.gelu(
            hidden,
            approximate="tanh",
        )

        feed_forward = self.ff2(
            hidden
        )

        return self.norm2(
            normalized + feed_forward
        )


def save_tensor(
    path: Path,
    tensor: torch.Tensor,
    transpose: bool = False,
) -> None:
    value = (
        tensor.detach()
        .cpu()
    )

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


def save_linear(
    directory: Path,
    name: str,
    layer: nn.Linear,
) -> None:
    # PyTorch: [output, input]
    # VORTEX:  [input, output]
    save_tensor(
        directory / f"{name}_weight.bin",
        layer.weight,
        transpose=True,
    )

    save_tensor(
        directory / f"{name}_bias.bin",
        layer.bias,
    )


def normal_small(
    generator: torch.Generator,
) -> torch.Tensor:
    return 0.1 * torch.randn(
        BATCH,
        TOKENS,
        MODEL_DIM,
        generator=generator,
    )


def normal_standard(
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        BATCH,
        TOKENS,
        MODEL_DIM,
        generator=generator,
    )


def normal_large(
    generator: torch.Generator,
) -> torch.Tensor:
    return 4.0 * torch.randn(
        BATCH,
        TOKENS,
        MODEL_DIM,
        generator=generator,
    )


def uniform_standard(
    generator: torch.Generator,
) -> torch.Tensor:
    return (
        2.0 *
        torch.rand(
            BATCH,
            TOKENS,
            MODEL_DIM,
            generator=generator,
        )
        - 1.0
    )


def uniform_large(
    generator: torch.Generator,
) -> torch.Tensor:
    return (
        16.0 *
        torch.rand(
            BATCH,
            TOKENS,
            MODEL_DIM,
            generator=generator,
        )
        - 8.0
    )


def near_constant(
    generator: torch.Generator,
) -> torch.Tensor:
    return (
        torch.full(
            (
                BATCH,
                TOKENS,
                MODEL_DIM,
            ),
            10.0,
        )
        +
        0.01 *
        torch.randn(
            BATCH,
            TOKENS,
            MODEL_DIM,
            generator=generator,
        )
    )


def sparse(
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.randn(
        BATCH,
        TOKENS,
        MODEL_DIM,
        generator=generator,
    ) * 3.0

    mask = (
        torch.rand(
            BATCH,
            TOKENS,
            MODEL_DIM,
            generator=generator,
        )
        < 0.10
    )

    return torch.where(
        mask,
        values,
        torch.zeros_like(values),
    )


def alternating(
    generator: torch.Generator,
) -> torch.Tensor:
    del generator

    indices = torch.arange(
        BATCH *
        TOKENS *
        MODEL_DIM,
        dtype=torch.int64,
    )

    signs = torch.where(
        indices % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )

    return (
        8.0 * signs.float()
    ).reshape(
        BATCH,
        TOKENS,
        MODEL_DIM,
    )


CASES: list[
    tuple[
        str,
        int,
        Callable[
            [torch.Generator],
            torch.Tensor,
        ],
    ]
] = [
    (
        "normal_small",
        25001,
        normal_small,
    ),
    (
        "normal_standard",
        25002,
        normal_standard,
    ),
    (
        "normal_large",
        25003,
        normal_large,
    ),
    (
        "uniform_standard",
        25004,
        uniform_standard,
    ),
    (
        "uniform_large",
        25005,
        uniform_large,
    ),
    (
        "near_constant",
        25006,
        near_constant,
    ),
    (
        "sparse",
        25007,
        sparse,
    ),
    (
        "alternating",
        25008,
        alternating,
    ),
]


def main() -> None:
    torch.set_num_threads(1)

    root = Path(
        "models/phase25_transformer_cases"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_lines = [
        (
            "case\tseed\tdistribution\t"
            "input_min\tinput_max\t"
            "reference_abs_max"
        )
    ]

    for case_name, seed, make_input in CASES:
        torch.manual_seed(
            seed
        )

        generator = (
            torch.Generator()
            .manual_seed(seed + 1000)
        )

        model = TinyTransformerBlock()
        model.eval()

        input_tensor = make_input(
            generator
        ).float()

        with torch.inference_mode():
            reference = model(
                input_tensor
            )

        if not torch.isfinite(
            reference
        ).all():
            raise RuntimeError(
                f"{case_name} produced non-finite reference output"
            )

        directory = (
            root /
            case_name
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_tensor(
            directory / "input.bin",
            input_tensor,
        )

        save_tensor(
            directory / "reference.bin",
            reference,
        )

        save_linear(
            directory,
            "q",
            model.q_proj,
        )

        save_linear(
            directory,
            "k",
            model.k_proj,
        )

        save_linear(
            directory,
            "v",
            model.v_proj,
        )

        save_linear(
            directory,
            "out",
            model.out_proj,
        )

        save_linear(
            directory,
            "ff1",
            model.ff1,
        )

        save_linear(
            directory,
            "ff2",
            model.ff2,
        )

        save_tensor(
            directory / "norm1_gamma.bin",
            model.norm1.weight,
        )

        save_tensor(
            directory / "norm1_beta.bin",
            model.norm1.bias,
        )

        save_tensor(
            directory / "norm2_gamma.bin",
            model.norm2.weight,
        )

        save_tensor(
            directory / "norm2_beta.bin",
            model.norm2.bias,
        )

        input_min = float(
            input_tensor.min()
        )

        input_max = float(
            input_tensor.max()
        )

        reference_abs_max = float(
            reference.abs().max()
        )

        manifest_lines.append(
            "\t".join(
                [
                    case_name,
                    str(seed),
                    case_name,
                    f"{input_min:.9g}",
                    f"{input_max:.9g}",
                    f"{reference_abs_max:.9g}",
                ]
            )
        )

        print(
            f"{case_name:18s} "
            f"seed={seed} "
            f"input=[{input_min:.3f}, "
            f"{input_max:.3f}] "
            f"output_abs_max="
            f"{reference_abs_max:.3f}"
        )

    (
        root /
        "manifest.tsv"
    ).write_text(
        "\n".join(
            manifest_lines
        )
        + "\n"
    )

    print(
        "\n[VORTEX] Generated "
        f"{len(CASES)} randomized transformer cases"
    )


if __name__ == "__main__":
    main()

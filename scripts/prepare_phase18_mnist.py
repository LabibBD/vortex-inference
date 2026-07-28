from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


BATCH_EXPORT = 1024
TRAIN_BATCH = 512
EPOCHS = 3


class MNISTMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 10)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.relu1(
            self.fc1(x)
        )

        x = self.relu2(
            self.fc2(x)
        )

        return self.fc3(x)


class ExportModel(nn.Module):
    def __init__(
        self,
        model: MNISTMLP,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.softmax(
            self.model(x),
            dim=-1,
        )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = (
                images.flatten(1)
                .to(device)
            )

            labels = labels.to(
                device
            )

            predictions = (
                model(images)
                .argmax(dim=1)
            )

            correct += int(
                (
                    predictions == labels
                ).sum()
            )

            total += labels.numel()

    return 100.0 * correct / total


def main() -> None:
    torch.manual_seed(1801)

    root = Path("data")
    models = Path("models")

    models.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = MNIST(
        root=root,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_dataset = MNIST(
        root=root,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[VORTEX] Training on {device}"
    )

    model = MNISTMLP().to(
        device
    )

    checkpoint = (
        models /
        "phase18_mnist.pt"
    )

    if checkpoint.exists():
        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device,
            )
        )

        print(
            "[VORTEX] Existing checkpoint loaded."
        )

    else:
        criterion = (
            nn.CrossEntropyLoss()
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        model.train()

        for epoch in range(EPOCHS):
            running_loss = 0.0

            for images, labels in train_loader:
                images = (
                    images.flatten(1)
                    .to(
                        device,
                        non_blocking=True,
                    )
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    labels
                )

                loss.backward()
                optimizer.step()

                running_loss += float(
                    loss
                )

            accuracy = evaluate(
                model,
                test_loader,
                device,
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"loss={running_loss / len(train_loader):.4f} "
                f"test_accuracy={accuracy:.2f}%"
            )

        torch.save(
            model.state_dict(),
            checkpoint,
        )

    accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print(
        f"\nFinal test accuracy: "
        f"{accuracy:.2f}%"
    )

    # ==================================================
    # Fixed 1024-image evaluation batch
    # ==================================================

    test_images = (
        test_dataset.data[
            :BATCH_EXPORT
        ]
        .float()
        .div(255.0)
        .reshape(
            BATCH_EXPORT,
            784,
        )
    )

    test_labels = (
        test_dataset.targets[
            :BATCH_EXPORT
        ]
    )

    model.eval()

    export_model = ExportModel(
        model
    ).to(device)

    export_model.eval()

    sample = test_images.to(
        device
    )

    with torch.inference_mode():
        reference = (
            export_model(sample)
            .cpu()
        )

    images_np = (
        test_images
        .numpy()
        .astype(np.float32)
    )

    labels_np = (
        test_labels
        .numpy()
        .astype(np.int64)
    )

    reference_np = (
        reference
        .numpy()
        .astype(np.float32)
    )

    images_np.tofile(
        models /
        "phase18_input.bin"
    )

    labels_np.tofile(
        models /
        "phase18_labels.bin"
    )

    reference_np.tofile(
        models /
        "phase18_reference.bin"
    )

    subset_predictions = (
        reference.argmax(dim=1)
    )

    subset_accuracy = (
        100.0 *
        (
            subset_predictions ==
            test_labels
        )
        .float()
        .mean()
        .item()
    )

    # Export from CPU for a portable ONNX file.
    cpu_export_model = ExportModel(
        MNISTMLP()
    )

    cpu_export_model.model.load_state_dict(
        model.cpu().state_dict()
    )

    cpu_export_model.eval()

    torch.onnx.export(
        cpu_export_model,
        test_images,
        models /
        "phase18_mnist.onnx",
        input_names=[
            "image_pixels"
        ],
        output_names=[
            "probabilities"
        ],
        opset_version=18,
        dynamo=False,
    )

    print(
        "\n================================="
    )
    print(
        " VORTEX Phase-18 Vision Model"
    )
    print(
        "=================================\n"
    )

    print(
        "Architecture: "
        "784 -> 256 -> 128 -> 10"
    )

    print(
        f"Evaluation images: "
        f"{BATCH_EXPORT}"
    )

    print(
        f"Subset accuracy: "
        f"{subset_accuracy:.2f}%"
    )

    print(
        "Output row-sum error:",
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
        "\n[VORTEX] MNIST model exported"
    )


if __name__ == "__main__":
    main()

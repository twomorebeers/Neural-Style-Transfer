import argparse
import csv
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.utils import save_image


# -----------------------------
# Utility functions
# -----------------------------

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ModelSpec:
    name: str
    input_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    content_layer: str
    style_layers: List[str]


MODEL_SPECS: Dict[str, ModelSpec] = {
    "vgg16": ModelSpec(
        name="vgg16",
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        content_layer="features.21",  # relu4_2 equivalent in torchvision indexing
        style_layers=["features.0", "features.5", "features.10", "features.19", "features.28"],
    ),
    "vgg19": ModelSpec(
        name="vgg19",
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        content_layer="features.21",  # conv4_2 / relu4_2 area commonly used in NST
        style_layers=["features.0", "features.5", "features.10", "features.19", "features.28"],
    ),
    "resnet50": ModelSpec(
        name="resnet50",
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        content_layer="layer3",
        style_layers=["relu", "layer1", "layer2", "layer3", "layer4"],
    ),
    "inception_v3": ModelSpec(
        name="inception_v3",
        input_size=299,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        content_layer="Mixed_6e",
        style_layers=["Conv2d_2b_3x3", "Mixed_5b", "Mixed_5d", "Mixed_6a", "Mixed_6e"],
    ),
    "squeezenet1_1": ModelSpec(
        name="squeezenet1_1",
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        content_layer="features.8",
        style_layers=["features.0", "features.3", "features.5", "features.8", "features.12"],
    ),
}


def build_model(model_name: str) -> nn.Module:
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif model_name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == "inception_v3":
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
            )
        model.aux_logits = False
        model.AuxLogits = None
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.detach().cpu().clamp(0, 1), str(path))


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


class Normalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_names: List[str], model_name: str):
        super().__init__()
        self.model = model
        self.layer_names = set(layer_names)
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.model_name.startswith("vgg") or self.model_name.startswith("squeezenet"):
            return self._forward_named_children(x)
        if self.model_name == "resnet50":
            return self._forward_resnet(x)
        if self.model_name == "inception_v3":
            return self._forward_inception(x)
        raise ValueError(f"Unsupported model_name: {self.model_name}")

    def _forward_named_children(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        for top_name, module in self.model.named_children():
            if top_name == "classifier":
                break
            # VGG / SqueezeNet store feature stack in .features
            if isinstance(module, nn.Sequential):
                for idx, layer in module.named_children():
                    x = layer(x)
                    layer_name = f"{top_name}.{idx}"
                    if layer_name in self.layer_names:
                        outputs[layer_name] = x
            else:
                x = module(x)
                if top_name in self.layer_names:
                    outputs[top_name] = x
        return outputs

    def _forward_resnet(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        x = self.model.conv1(x)
        if "conv1" in self.layer_names:
            outputs["conv1"] = x
        x = self.model.bn1(x)
        x = self.model.relu(x)
        if "relu" in self.layer_names:
            outputs["relu"] = x
        x = self.model.maxpool(x)
        if "maxpool" in self.layer_names:
            outputs["maxpool"] = x
        x = self.model.layer1(x)
        if "layer1" in self.layer_names:
            outputs["layer1"] = x
        x = self.model.layer2(x)
        if "layer2" in self.layer_names:
            outputs["layer2"] = x
        x = self.model.layer3(x)
        if "layer3" in self.layer_names:
            outputs["layer3"] = x
        x = self.model.layer4(x)
        if "layer4" in self.layer_names:
            outputs["layer4"] = x
        return outputs

    def _forward_inception(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        # Mirror torchvision.models.inception.Inception3._forward without classifier head.
        x = self.model.Conv2d_1a_3x3(x)
        if "Conv2d_1a_3x3" in self.layer_names:
            outputs["Conv2d_1a_3x3"] = x
        x = self.model.Conv2d_2a_3x3(x)
        if "Conv2d_2a_3x3" in self.layer_names:
            outputs["Conv2d_2a_3x3"] = x
        x = self.model.Conv2d_2b_3x3(x)
        if "Conv2d_2b_3x3" in self.layer_names:
            outputs["Conv2d_2b_3x3"] = x
        x = self.model.maxpool1(x)
        if "maxpool1" in self.layer_names:
            outputs["maxpool1"] = x
        x = self.model.Conv2d_3b_1x1(x)
        if "Conv2d_3b_1x1" in self.layer_names:
            outputs["Conv2d_3b_1x1"] = x
        x = self.model.Conv2d_4a_3x3(x)
        if "Conv2d_4a_3x3" in self.layer_names:
            outputs["Conv2d_4a_3x3"] = x
        x = self.model.maxpool2(x)
        if "maxpool2" in self.layer_names:
            outputs["maxpool2"] = x
        x = self.model.Mixed_5b(x)
        if "Mixed_5b" in self.layer_names:
            outputs["Mixed_5b"] = x
        x = self.model.Mixed_5c(x)
        if "Mixed_5c" in self.layer_names:
            outputs["Mixed_5c"] = x
        x = self.model.Mixed_5d(x)
        if "Mixed_5d" in self.layer_names:
            outputs["Mixed_5d"] = x
        x = self.model.Mixed_6a(x)
        if "Mixed_6a" in self.layer_names:
            outputs["Mixed_6a"] = x
        x = self.model.Mixed_6b(x)
        if "Mixed_6b" in self.layer_names:
            outputs["Mixed_6b"] = x
        x = self.model.Mixed_6c(x)
        if "Mixed_6c" in self.layer_names:
            outputs["Mixed_6c"] = x
        x = self.model.Mixed_6d(x)
        if "Mixed_6d" in self.layer_names:
            outputs["Mixed_6d"] = x
        x = self.model.Mixed_6e(x)
        if "Mixed_6e" in self.layer_names:
            outputs["Mixed_6e"] = x
        # Stop here intentionally; deeper inception blocks become very abstract for NST.
        return outputs


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    loss_h = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    loss_w = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return loss_h + loss_w


def compute_style_targets(
    extractor: FeatureExtractor,
    normalizer: Normalizer,
    style_img: torch.Tensor,
    style_layers: List[str],
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        style_features = extractor(normalizer(style_img))
        return {name: gram_matrix(style_features[name]).detach() for name in style_layers}


def compute_content_target(
    extractor: FeatureExtractor,
    normalizer: Normalizer,
    content_img: torch.Tensor,
    content_layer: str,
) -> torch.Tensor:
    with torch.no_grad():
        content_features = extractor(normalizer(content_img))
        return content_features[content_layer].detach()


def style_transfer(
    model_name: str,
    content_path: str,
    style_path: str,
    output_dir: str,
    num_steps: int = 300,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    tv_weight: float = 1e-4,
    init_mode: str = "content",
    log_interval: int = 50,
    device: str = "auto",
    seed: int = 42,
) -> Dict[str, float]:
    set_seed(seed)
    chosen_device = get_default_device() if device == "auto" else torch.device(device)
    spec = MODEL_SPECS[model_name]
    model = build_model(model_name).to(chosen_device)
    normalizer = Normalizer(spec.mean, spec.std).to(chosen_device)

    content_img = load_image(content_path, spec.input_size, chosen_device)
    style_img = load_image(style_path, spec.input_size, chosen_device)

    if init_mode == "content":
        input_img = content_img.clone()
    elif init_mode == "noise":
        input_img = torch.rand_like(content_img)
    elif init_mode == "blend":
        input_img = (0.6 * content_img + 0.4 * style_img).clamp(0, 1)
    else:
        raise ValueError("init_mode must be one of: content, noise, blend")

    input_img.requires_grad_(True)

    requested_layers = [spec.content_layer] + spec.style_layers
    extractor = FeatureExtractor(model, requested_layers, model_name)

    style_targets = compute_style_targets(extractor, normalizer, style_img, spec.style_layers)
    content_target = compute_content_target(extractor, normalizer, content_img, spec.content_layer)

    optimizer = torch.optim.LBFGS([input_img])

    step = [0]
    history = []
    t0 = time.perf_counter()
    last_metrics = {
        "content_loss": math.nan,
        "style_loss": math.nan,
        "tv_loss": math.nan,
        "total_loss": math.nan,
    }

    while step[0] <= num_steps:
        def closure():
            optimizer.zero_grad(set_to_none=True)
            input_img.data.clamp_(0, 1)

            features = extractor(normalizer(input_img))
            content_loss = F.mse_loss(features[spec.content_layer], content_target)

            style_loss = torch.tensor(0.0, device=chosen_device)
            for layer_name in spec.style_layers:
                gm_input = gram_matrix(features[layer_name])
                gm_target = style_targets[layer_name]
                style_loss = style_loss + F.mse_loss(gm_input, gm_target)
            style_loss = style_loss / len(spec.style_layers)

            tv_loss = total_variation_loss(input_img)
            total_loss = (
                content_weight * content_loss
                + style_weight * style_loss
                + tv_weight * tv_loss
            )
            total_loss.backward()

            last_metrics["content_loss"] = float(content_loss.detach().item())
            last_metrics["style_loss"] = float(style_loss.detach().item())
            last_metrics["tv_loss"] = float(tv_loss.detach().item())
            last_metrics["total_loss"] = float(total_loss.detach().item())

            if step[0] % log_interval == 0 or step[0] == num_steps:
                history.append({"step": step[0], **last_metrics})
                print(
                    f"[{model_name}] step={step[0]:4d} "
                    f"content={last_metrics['content_loss']:.6f} "
                    f"style={last_metrics['style_loss']:.6f} "
                    f"tv={last_metrics['tv_loss']:.6f} "
                    f"total={last_metrics['total_loss']:.6f}"
                )

            step[0] += 1
            return total_loss

        optimizer.step(closure)

    runtime_sec = time.perf_counter() - t0
    input_img.data.clamp_(0, 1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{model_name}_stylized.png"
    history_path = out_dir / f"{model_name}_history.csv"
    save_tensor_image(input_img, image_path)

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "content_loss", "style_loss", "tv_loss", "total_loss"])
        writer.writeheader()
        writer.writerows(history)

    metrics = OrderedDict(
        model=model_name,
        input_size=spec.input_size,
        steps=num_steps,
        content_weight=content_weight,
        style_weight=style_weight,
        tv_weight=tv_weight,
        runtime_sec=runtime_sec,
        content_loss=last_metrics["content_loss"],
        style_loss=last_metrics["style_loss"],
        tv_loss=last_metrics["tv_loss"],
        total_loss=last_metrics["total_loss"],
        parameters=sum(p.numel() for p in model.parameters()),
        output_image=str(image_path),
        loss_history_csv=str(history_path),
        device=str(chosen_device),
    )
    return metrics


def compare_models(
    models_to_run: List[str],
    content_path: str,
    style_path: str,
    output_dir: str,
    num_steps: int,
    style_weight: float,
    content_weight: float,
    tv_weight: float,
    init_mode: str,
    log_interval: int,
    device: str,
    seed: int,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in models_to_run:
        print(f"\n=== Running style transfer with {model_name} ===")
        metrics = style_transfer(
            model_name=model_name,
            content_path=content_path,
            style_path=style_path,
            output_dir=str(output),
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
            tv_weight=tv_weight,
            init_mode=init_mode,
            log_interval=log_interval,
            device=device,
            seed=seed,
        )
        rows.append(metrics)

    summary_path = output / "comparison_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Comparison summary ===")
    for row in rows:
        print(
            f"{row['model']:15s} | runtime={row['runtime_sec']:.2f}s | "
            f"content={row['content_loss']:.6f} | style={row['style_loss']:.6f} | "
            f"tv={row['tv_loss']:.6f} | total={row['total_loss']:.6f}"
        )
    print(f"\nSaved summary to: {summary_path}")
    print("Note: loss magnitudes are most reliable for comparing runs within the same model family.")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pretrained CNN backbones for neural style transfer.")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for results")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["vgg19"],
        choices=list(MODEL_SPECS.keys()),
        help="One or more backbones to run",
    )
    parser.add_argument("--steps", type=int, default=300, help="Number of LBFGS steps")
    parser.add_argument("--style-weight", type=float, default=1e6, help="Weight for style loss")
    parser.add_argument("--content-weight", type=float, default=1.0, help="Weight for content loss")
    parser.add_argument("--tv-weight", type=float, default=1e-4, help="Weight for total variation regularization")
    parser.add_argument("--init", type=str, default="content", choices=["content", "noise", "blend"], help="Initialization mode")
    parser.add_argument("--log-interval", type=int, default=50, help="How often to print and record losses")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_models(
        models_to_run=args.models,
        content_path=args.content,
        style_path=args.style,
        output_dir=args.output_dir,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        init_mode=args.init,
        log_interval=args.log_interval,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

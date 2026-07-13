"""Command-line interface for stemseg.

Subcommands:
    stemseg simulate    simulate a material to .npz and/or a .png preview
    stemseg segment     segment an .npz sample or a real image with a method
    stemseg train       train the U-Net and fit the random forest
    stemseg benchmark   run a YAML benchmark config, save JSON + figure
    stemseg samples     regenerate the committed sample images
    stemseg gallery     render the segmentation-overlay gallery
    stemseg demo        segment every committed sample, print + save metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from stemseg.benchmark import build_methods, run_config
from stemseg.io import load_sample, save_rf, save_sample
from stemseg.metrics import score_segmentation
from stemseg.plots import gallery, plot_payload, show_image, show_labels
from stemseg.real import prepare_real
from stemseg.sim import material_config, simulate_image
from stemseg.train import TrainSettings, sample_pixels, train_unet

DEFAULT_MODELS = {"unet": "models/unet.pt", "rf": "models/rf.pkl"}

# (name, material, dose, disorder, seed): the committed reference samples.
SAMPLES = (
    ("graphene_d150", "graphene", 150.0, 0.12, 0),
    ("hbn_d150", "hbn", 150.0, 0.14, 1),
    ("mos2_d200", "mos2", 200.0, 0.10, 2),
    ("oxide_d250", "oxide", 250.0, 0.10, 3),
)


def _cmd_simulate(args: argparse.Namespace) -> None:
    config = material_config(
        args.material, size=args.size, dose=args.dose, disorder_fraction=args.disorder
    )
    result = simulate_image(config, np.random.default_rng(args.seed))
    if args.out:
        save_sample(args.out, result)
        print(f"wrote {args.out}")
    if args.figure:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
        show_image(axes[0], result.image)
        axes[0].set_title(f"{args.material}, dose {args.dose:g}")
        show_labels(axes[1], result.labels)
        axes[1].set_title("ground-truth labels")
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=140)
        print(f"saved {args.figure}")
    if not args.out and not args.figure:
        print("nothing to do: pass --out and/or --figure")


def _cmd_segment(args: argparse.Namespace) -> None:
    path = Path(args.image)
    methods = build_methods([args.method], DEFAULT_MODELS)
    method = methods[args.method]

    if path.suffix == ".npz":
        sample = load_sample(path)
        image, truth = sample.image, sample.labels
    else:
        image = prepare_real(path, invert=args.invert, downsample_factor=args.downsample)
        truth = None

    pred = method.predict(image)
    if truth is not None:
        score = score_segmentation(truth, pred)
        print(
            f"{method.label}: pixel acc {score.pixel_accuracy:.3f}  mean IoU {score.mean_iou:.3f}"
        )
        for name, iou in score.as_dict()["iou"].items():
            print(f"  IoU {name:11s} {iou if iou is None else round(iou, 3)}")
        print(f"  boundary error {score.boundary_error_px:.2f} px")

    if args.figure:
        ncol = 3 if truth is not None else 2
        fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 4.4))
        show_image(axes[0], image)
        axes[0].set_title("image")
        show_labels(axes[1], pred)
        axes[1].set_title(f"{method.label} prediction")
        if truth is not None:
            show_labels(axes[2], truth)
            axes[2].set_title("ground truth")
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=140)
        print(f"saved {args.figure}")


def _cmd_train(args: argparse.Namespace) -> None:
    variants = {
        "full": {},
        "fixed_dose": {"randomize_dose": False},
        "no_disorder": {"include_disorder": False},
    }
    if args.ablation:
        for variant, overrides in variants.items():
            settings = TrainSettings(steps=args.steps, seed=args.seed, **overrides)
            model, history = train_unet(settings, log_every=args.steps // 4 or 1)
            out = Path("models/ablation") / f"{variant}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out)
            print(f"saved {out}  (final loss {history[-1]:.4f})")
        return

    settings = TrainSettings(steps=args.steps, seed=args.seed)
    if args.model in ("unet", "both"):
        model, history = train_unet(settings, log_every=args.steps // 8 or 1)
        out = Path(args.out or "models/unet.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"saved {out}  ({n_params} parameters, final loss {history[-1]:.4f})")
        if args.figure:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history, lw=0.8)
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.set_title("U-Net training loss")
            fig.tight_layout()
            Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.figure, dpi=140)
            print(f"saved {args.figure}")
    if args.model in ("rf", "both"):
        from stemseg.classical import RandomForestPixelClassifier

        x, y = sample_pixels(settings, n_images=args.rf_images, per_image=args.rf_pixels)
        rf = RandomForestPixelClassifier(seed=args.seed)
        rf.fit(x, y)
        save_rf("models/rf.pkl", rf)
        print(f"saved models/rf.pkl  (fit on {len(y)} pixels)")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    for config in args.configs:
        payload = run_config(config)
        name = payload["config"].get("name", Path(config).stem)
        if payload["mode"] != "materials":
            plot_payload(payload, Path("figures") / f"{name}.png")


def _cmd_samples(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, material, dose, disorder, seed in SAMPLES:
        config = material_config(
            material, size=192, dose=dose, disorder_fraction=disorder, rotation_deg=15.0
        )
        result = simulate_image(config, np.random.default_rng(seed))
        save_sample(out / f"{name}.npz", result)
        print(f"wrote {out / (name + '.npz')}")


def _cmd_gallery(args: argparse.Namespace) -> None:
    methods = build_methods(["unet"], DEFAULT_MODELS)
    predict = methods["unet"].predict
    rows = []
    for name, *_ in SAMPLES:
        sample = load_sample(Path(args.data) / f"{name}.npz")
        rows.append((name, sample.image, sample.labels, predict(sample.image)))
    gallery(rows, args.figure)
    print(f"saved {args.figure}")


def _cmd_demo(args: argparse.Namespace) -> None:
    method_names = ["threshold", "rf", "unet"]
    methods = build_methods(method_names, DEFAULT_MODELS)
    all_metrics: dict[str, dict] = {}
    for name, *_ in SAMPLES:
        sample = load_sample(Path(args.data) / f"{name}.npz")
        all_metrics[name] = {}
        for mname in method_names:
            score = score_segmentation(sample.labels, methods[mname].predict(sample.image))
            all_metrics[name][mname] = {
                "pixel_accuracy": round(score.pixel_accuracy, 4),
                "mean_iou": round(score.mean_iou, 4),
                "boundary_error_px": (
                    None if np.isnan(score.boundary_error_px) else round(score.boundary_error_px, 3)
                ),
                "iou": {
                    k: (None if v is None else round(v, 4))
                    for k, v in score.as_dict()["iou"].items()
                },
            }
            print(
                f"{name:14s} {mname:9s} acc {score.pixel_accuracy:.3f}  mIoU {score.mean_iou:.3f}"
            )

    metrics = Path(args.metrics)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps({"samples": all_metrics}, indent=2) + "\n")
    print(f"saved {metrics}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the stemseg console command."""
    parser = argparse.ArgumentParser(prog="stemseg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="simulate a material preset")
    p.add_argument("--material", default="graphene")
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--dose", type=float, default=200.0)
    p.add_argument("--disorder", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help=".npz output with labels")
    p.add_argument("--figure", default=None, help=".png preview")
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("segment", help="segment an .npz sample or real image")
    p.add_argument("image", help=".npz sample or real .png/.tif/.jpg")
    p.add_argument("--method", default="unet", choices=["threshold", "rf", "unet"])
    p.add_argument("--invert", action="store_true", help="invert real-image contrast")
    p.add_argument("--downsample", type=int, default=1, help="block-average a real image")
    p.add_argument("--figure", default=None, help="overlay .png")
    p.set_defaults(func=_cmd_segment)

    p = sub.add_parser("train", help="train the U-Net and fit the random forest")
    p.add_argument("--model", default="both", choices=["unet", "rf", "both"])
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rf-images", type=int, default=30)
    p.add_argument("--rf-pixels", type=int, default=1200)
    p.add_argument("--out", default=None)
    p.add_argument("--figure", default=None)
    p.add_argument("--ablation", action="store_true", help="train U-Net ablation variants")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("benchmark", help="run YAML benchmark configs")
    p.add_argument("configs", nargs="+")
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("samples", help="regenerate the committed samples")
    p.add_argument("--out", default="data/sample")
    p.set_defaults(func=_cmd_samples)

    p = sub.add_parser("gallery", help="render the segmentation-overlay gallery")
    p.add_argument("--data", default="data/sample")
    p.add_argument("--figure", default="figures/gallery.png")
    p.set_defaults(func=_cmd_gallery)

    p = sub.add_parser("demo", help="segment every committed sample")
    p.add_argument("--data", default="data/sample")
    p.add_argument("--metrics", default="results/metrics.json")
    p.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

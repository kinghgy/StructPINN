from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .benchmark import benchmark_devices
from .config import load_config
from .experiments import evaluate_checkpoint_sweep, run_finetune_seed_sweep, run_seed_sweep
from .solver import predict_checkpoint, solve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structpinn",
        description="Train configurable PINN solvers for beams and plates.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="Train a problem from a YAML configuration.")
    train.add_argument("config", type=Path)
    train.add_argument("--output", type=Path, default=None)
    train.add_argument(
        "--init", type=Path, default=None, help="Warm-start from a compatible final model.pt."
    )
    train.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume Adam state, collocation points, loss weights, and RNG state.",
    )

    check = subcommands.add_parser("check", help="Validate a YAML configuration without training.")
    check.add_argument("config", type=Path)

    predict = subcommands.add_parser("predict", help="Export fields from a trained checkpoint.")
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="auto")
    predict.add_argument(
        "--plate-field-source", choices=["mixed", "displacement"], default=None
    )

    sweep = subcommands.add_parser(
        "sweep", help="Train independent runs for several seeds and aggregate metrics."
    )
    sweep.add_argument("config", type=Path)
    sweep.add_argument("--seeds", type=int, nargs="+", required=True)
    sweep.add_argument("--output-root", type=Path, required=True)
    sweep.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Retrain even when a complete run with the same seed is present.",
    )
    sweep.set_defaults(resume=True)

    benchmark = subcommands.add_parser(
        "benchmark", help="Compare training throughput on CPU and CUDA devices."
    )
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    benchmark.add_argument("--repetitions", type=int, default=3)
    benchmark.add_argument("--adam-epochs", type=int, default=300)
    benchmark.add_argument("--lbfgs-max-iter", type=int, default=0)
    benchmark.add_argument("--dtype", choices=["float32", "float64"], default=None)
    benchmark.add_argument("--output", type=Path, required=True)

    finetune = subcommands.add_parser(
        "finetune-sweep", help="Fine-tune matching seed checkpoints and aggregate metrics."
    )
    finetune.add_argument("config", type=Path)
    finetune.add_argument("--checkpoint-root", type=Path, required=True)
    finetune.add_argument("--seeds", type=int, nargs="+", required=True)
    finetune.add_argument("--output-root", type=Path, required=True)
    finetune.add_argument("--no-resume", action="store_false", dest="resume")
    finetune.set_defaults(resume=True)

    evaluate = subcommands.add_parser(
        "evaluate-sweep", help="Evaluate matching seed checkpoints with one field source."
    )
    evaluate.add_argument("--checkpoint-root", type=Path, required=True)
    evaluate.add_argument("--seeds", type=int, nargs="+", required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument(
        "--plate-field-source", choices=["mixed", "displacement"], default="mixed"
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "check":
        config = load_config(arguments.config)
        print(f"Configuration is valid: {config['problem']['type']}")
        return
    if arguments.command == "predict":
        predict_checkpoint(
            arguments.checkpoint,
            arguments.output,
            arguments.device,
            arguments.plate_field_source,
        )
        return
    if arguments.command == "sweep":
        run_seed_sweep(
            arguments.config,
            arguments.seeds,
            arguments.output_root,
            arguments.resume,
        )
        return
    if arguments.command == "benchmark":
        benchmark_devices(
            arguments.config,
            arguments.devices,
            arguments.repetitions,
            arguments.adam_epochs,
            arguments.output,
            arguments.dtype,
            arguments.lbfgs_max_iter,
        )
        return
    if arguments.command == "finetune-sweep":
        run_finetune_seed_sweep(
            arguments.config,
            arguments.checkpoint_root,
            arguments.seeds,
            arguments.output_root,
            arguments.resume,
        )
        return
    if arguments.command == "evaluate-sweep":
        evaluate_checkpoint_sweep(
            arguments.checkpoint_root,
            arguments.seeds,
            arguments.output_root,
            arguments.device,
            arguments.plate_field_source,
        )
        return
    solve(
        arguments.config,
        arguments.output,
        initialization_checkpoint=arguments.init,
        resume_checkpoint=arguments.resume,
    )


if __name__ == "__main__":
    main()

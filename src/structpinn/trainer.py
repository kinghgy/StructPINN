from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from .problems.base import PINNProblem
from .sampling import residual_adaptive_sample, sample_interior
from .utils import capture_rng_state, restore_rng_state, tensor_to_float
from .weighting import GradientLossBalancer


@dataclass
class TrainingResult:
    history: list[dict[str, Any]]
    elapsed_seconds: float
    final_weights: dict[str, float]
    stopped_early: bool
    resumed_from_epoch: int | None = None


def _snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    # Keep the best-state copy on the active device.  Moving every improving
    # state to CPU forces a CUDA synchronization on almost every early epoch.
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _training_checkpoint(
    problem: PINNProblem,
    optimizer: torch.optim.Optimizer,
    balancer: GradientLossBalancer,
    interior: torch.Tensor,
    anchor_targets: dict[str, torch.Tensor] | None,
    history: list[dict[str, Any]],
    best_loss: float,
    best_state: dict[str, torch.Tensor] | None,
    patience: int,
    stopped_early: bool,
    epoch: int,
    elapsed_seconds: float,
    metadata: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "structpinn_training_state",
        "phase": phase,
        "adam_epoch": epoch,
        "model_state_dict": problem.state_dict(),
        "optimizer_name": "adam",
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_balancer_state_dict": balancer.state_dict(),
        "interior": interior.detach(),
        "anchor_targets": (
            {name: value.detach() for name, value in anchor_targets.items()}
            if anchor_targets is not None
            else None
        ),
        "history": history,
        "best_loss": best_loss,
        "best_state_dict": best_state,
        "patience": patience,
        "stopped_early": stopped_early,
        "elapsed_seconds": elapsed_seconds,
        "rng_state": capture_rng_state(),
        "metadata": metadata,
    }


def _save_training_checkpoint(
    state: dict[str, Any],
    checkpoint_directory: Path,
    epoch: int,
    keep_last: int,
    save_snapshot: bool,
) -> None:
    _atomic_torch_save(state, checkpoint_directory / "latest.pt")
    if not save_snapshot or keep_last == 0:
        return
    _atomic_torch_save(state, checkpoint_directory / f"adam_{epoch:08d}.pt")
    snapshots = sorted(checkpoint_directory.glob("adam_*.pt"))
    for expired in snapshots[:-keep_last]:
        expired.unlink()


def _reference_anchor_targets(
    problem: PINNProblem,
    interior: torch.Tensor,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Cache displacement-derived fields before fine-tuning starts.

    The cache is a functional teacher: it constrains the solution rather than
    individual parameters, which is important because small parameter changes
    can cause large changes in second derivatives.
    """
    if not hasattr(problem, "displacement_derived_fields"):
        raise ValueError("Reference anchoring is currently supported only for plates.")
    coordinates = interior.detach().clone().requires_grad_(True)
    fields = problem.displacement_derived_fields(coordinates)  # type: ignore[attr-defined]
    return {name: fields[name].detach() for name in settings["fields"]}


def _reference_anchor_losses(
    problem: PINNProblem,
    interior: torch.Tensor,
    targets: dict[str, torch.Tensor],
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    fields = problem.displacement_derived_fields(interior)  # type: ignore[attr-defined]
    global_weight = float(settings.get("weight", 1.0))
    field_weights = settings.get("field_weights", {})
    losses: dict[str, torch.Tensor] = {}
    for name, target in targets.items():
        # Normalize by the teacher field itself.  PDE characteristic scales can
        # be orders of magnitude larger than the realized solution and would
        # make an apparently large anchor weight ineffective.
        scale = torch.sqrt(torch.mean(target**2)).clamp_min(
            100.0 * torch.finfo(target.dtype).eps
        )
        losses[f"anchor/{name}"] = (
            global_weight
            * float(field_weights.get(name, 1.0))
            * torch.mean(((fields[name] - target) / scale) ** 2)
        )
    return losses


def train_problem(
    problem: PINNProblem,
    training: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_directory: str | Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    resume_state: dict[str, Any] | None = None,
) -> TrainingResult:
    collocation_count = int(training["collocation_points"])
    sampler = training["sampler"]
    exclusion_radius = float(training.get("point_exclusion_radius", 1.0e-4))
    interior = sample_interior(
        problem,
        collocation_count,
        sampler,
        device,
        dtype,
        exclusion_radius,
    )
    boundary_points = problem.make_boundary_points(int(training.get("boundary_points", 64)))
    trainable_parameters = [parameter for parameter in problem.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The problem has no trainable parameters.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(training["learning_rate"]))
    weighting_settings = training["adaptive_weighting"]
    balancer = GradientLossBalancer(weighting_settings)
    weight_every = max(int(weighting_settings.get("update_every", 100)), 1)
    adaptive_settings = training["adaptive_sampling"]
    adaptive_every = max(int(adaptive_settings.get("resample_every", 500)), 1)
    early = training["early_stopping"]
    log_every = max(int(training.get("log_every", 100)), 1)
    adam_epochs = int(training["adam_epochs"])
    anchor_settings = training.get("reference_anchor", {})
    anchor_targets = None
    if anchor_settings.get("enabled", False):
        anchor_targets = _reference_anchor_targets(problem, interior, anchor_settings)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    stopped_early = False
    start_epoch = 1
    resumed_from_epoch: int | None = None
    elapsed_offset = 0.0

    if resume_state is not None:
        if resume_state.get("kind") != "structpinn_training_state":
            raise ValueError("The resume file is not a StructPINN training-state checkpoint.")
        if int(resume_state.get("format_version", -1)) != 1:
            raise ValueError("Unsupported StructPINN training-state checkpoint version.")
        if resume_state.get("optimizer_name") != "adam":
            raise ValueError("This release can resume only Adam-phase training checkpoints.")
        problem.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        balancer.load_state_dict(resume_state["loss_balancer_state_dict"])
        interior = resume_state["interior"].to(device=device, dtype=dtype).requires_grad_(True)
        saved_targets = resume_state.get("anchor_targets")
        anchor_targets = (
            {
                name: value.to(device=device, dtype=dtype)
                for name, value in saved_targets.items()
            }
            if saved_targets is not None
            else None
        )
        history = list(resume_state.get("history", []))
        best_loss = float(resume_state.get("best_loss", float("inf")))
        saved_best = resume_state.get("best_state_dict")
        best_state = (
            {name: value.to(device=device, dtype=dtype) for name, value in saved_best.items()}
            if saved_best is not None
            else None
        )
        patience = int(resume_state.get("patience", 0))
        stopped_early = bool(resume_state.get("stopped_early", False))
        resumed_from_epoch = int(resume_state["adam_epoch"])
        start_epoch = resumed_from_epoch + 1
        elapsed_offset = float(resume_state.get("elapsed_seconds", 0.0))
        restore_rng_state(resume_state["rng_state"])

    checkpoint_settings = training.get("checkpointing", {})
    checkpoint_enabled = bool(checkpoint_settings.get("enabled", True))
    checkpoint_every = max(int(checkpoint_settings.get("every", 500)), 1)
    checkpoint_keep_last = max(int(checkpoint_settings.get("keep_last", 2)), 0)
    checkpoint_path = Path(checkpoint_directory) if checkpoint_directory is not None else None
    metadata = checkpoint_metadata or {}
    start_time = perf_counter()

    problem.train()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, adam_epochs + 1):
        last_epoch = epoch
        optimizer.zero_grad(set_to_none=True)
        interior.grad = None
        losses = problem.loss_terms(interior, boundary_points)
        if anchor_targets is not None:
            losses.update(
                _reference_anchor_losses(problem, interior, anchor_targets, anchor_settings)
            )
        if balancer.enabled and (epoch == 1 or epoch % weight_every == 0):
            balancer.update(losses, problem)
        total = balancer.combine(losses)
        total.backward()
        optimizer.step()
        current = tensor_to_float(total)

        if current < best_loss:
            relative = (best_loss - current) / max(abs(best_loss), 1.0e-12)
            if best_state is None or relative >= float(early["min_relative_improvement"]):
                best_loss = current
                best_state = _snapshot_state(problem)
                patience = 0
            else:
                patience += 1
        else:
            patience += 1

        if epoch == 1 or epoch % log_every == 0 or epoch == adam_epochs:
            row: dict[str, Any] = {
                "phase": "adam",
                "step": epoch,
                "total": current,
                "elapsed_seconds": elapsed_offset + perf_counter() - start_time,
            }
            row.update({name: tensor_to_float(value) for name, value in losses.items()})
            row.update({f"weight/{name}": value for name, value in balancer.snapshot().items()})
            history.append(row)
            print(f"Adam {epoch:6d}/{adam_epochs}: loss={current:.6e}")

        if adaptive_settings.get("enabled", True) and epoch % adaptive_every == 0:
            problem.eval()
            interior = residual_adaptive_sample(
                problem,
                collocation_count,
                adaptive_settings,
                sampler,
                device,
                dtype,
                exclusion_radius,
            )
            problem.train()

        should_stop = early.get("enabled", True) and patience >= int(early["patience"])
        if checkpoint_enabled and checkpoint_path is not None and (
            epoch % checkpoint_every == 0 or should_stop
        ):
            state = _training_checkpoint(
                problem,
                optimizer,
                balancer,
                interior,
                anchor_targets,
                history,
                best_loss,
                best_state,
                patience,
                bool(should_stop),
                epoch,
                elapsed_offset + perf_counter() - start_time,
                metadata,
                "adam",
            )
            _save_training_checkpoint(
                state,
                checkpoint_path,
                epoch,
                checkpoint_keep_last,
                save_snapshot=True,
            )

        if should_stop:
            print(f"Adam early stopping at epoch {epoch}; best loss={best_loss:.6e}")
            stopped_early = True
            break

    if checkpoint_enabled and checkpoint_path is not None:
        state = _training_checkpoint(
            problem,
            optimizer,
            balancer,
            interior,
            anchor_targets,
            history,
            best_loss,
            best_state,
            patience,
            stopped_early,
            last_epoch,
            elapsed_offset + perf_counter() - start_time,
            metadata,
            "adam_complete",
        )
        _save_training_checkpoint(
            state,
            checkpoint_path,
            last_epoch,
            checkpoint_keep_last,
            save_snapshot=False,
        )

    if best_state is not None:
        problem.load_state_dict(best_state)

    lbfgs_iterations = int(training.get("lbfgs_max_iter", 0))
    if lbfgs_iterations > 0:
        problem.train()
        optimizer_lbfgs = torch.optim.LBFGS(
            trainable_parameters,
            max_iter=lbfgs_iterations,
            max_eval=max(lbfgs_iterations * 5 // 4, lbfgs_iterations),
            history_size=int(training.get("lbfgs_history_size", 100)),
            tolerance_grad=float(training.get("lbfgs_tolerance_grad", 1.0e-11)),
            tolerance_change=float(training.get("lbfgs_tolerance_change", 1.0e-12)),
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            optimizer_lbfgs.zero_grad(set_to_none=True)
            interior.grad = None
            losses = problem.loss_terms(interior, boundary_points)
            if anchor_targets is not None:
                losses.update(
                    _reference_anchor_losses(problem, interior, anchor_targets, anchor_settings)
                )
            total = balancer.combine(losses)
            total.backward()
            closure_calls += 1
            if closure_calls == 1 or closure_calls % log_every == 0:
                row: dict[str, Any] = {
                    "phase": "lbfgs",
                    "step": len(history) + closure_calls,
                    "total": tensor_to_float(total),
                    "elapsed_seconds": elapsed_offset + perf_counter() - start_time,
                }
                row.update({name: tensor_to_float(value) for name, value in losses.items()})
                history.append(row)
                print(f"L-BFGS call {closure_calls:6d}: loss={tensor_to_float(total):.6e}")
            return total

        optimizer_lbfgs.step(closure)

    return TrainingResult(
        history=history,
        elapsed_seconds=elapsed_offset + perf_counter() - start_time,
        final_weights=deepcopy(balancer.snapshot()),
        stopped_early=stopped_early,
        resumed_from_epoch=resumed_from_epoch,
    )

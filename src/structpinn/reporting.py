from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_history(history: list[dict[str, Any]], path: str | Path) -> None:
    if not history:
        return
    columns: list[str] = []
    for row in history:
        for key in row:
            if key not in columns:
                columns.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def save_table(data: dict[str, np.ndarray], path: str | Path) -> None:
    columns = list(data)
    arrays = [np.asarray(data[column]).reshape(-1) for column in columns]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*arrays, strict=True))


def plot_loss(history: list[dict[str, Any]], path: str | Path) -> None:
    if not history:
        return
    epochs = [row["step"] for row in history if "total" in row]
    totals = [row["total"] for row in history if "total" in row]
    if not totals:
        return
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    axis.plot(epochs, totals, color="#d55e00")
    axis.set_yscale("log")
    axis.set_xlabel("Optimization step")
    axis.set_ylabel("Weighted loss")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_beam(
    prediction: dict[str, np.ndarray],
    reference: dict[str, np.ndarray] | None,
    path: str | Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150, sharex=True)
    fields = (("u", "Deflection"), ("theta", "Rotation"), ("moment", "Moment"), ("shear", "Shear"))
    x = prediction["x"]
    for axis, (field, title) in zip(axes.flat, fields, strict=True):
        axis.plot(x, prediction[field], label="PINN", linewidth=2)
        if reference is not None:
            axis.plot(reference["x"], reference[field], "--", label="FEM", linewidth=1.5)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend()
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_plate(
    prediction: dict[str, np.ndarray],
    grid_size: int,
    path: str | Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    x = prediction["x"].reshape(grid_size, grid_size)
    y = prediction["y"].reshape(grid_size, grid_size)
    for axis, field, title in zip(
        axes,
        ("w", "Mx", "My"),
        ("Deflection", "Moment Mx", "Moment My"),
        strict=True,
    ):
        contour = axis.contourf(x, y, prediction[field].reshape(grid_size, grid_size), 40, cmap="coolwarm")
        axis.set_aspect("equal")
        axis.set_title(title)
        figure.colorbar(contour, ax=axis)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: str,
    ) -> None:
        super().__init__()
        dimensions = [input_dim, *hidden_layers, output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )
        self.activation = make_activation(activation)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for layer in self.layers[:-1]:
            values = self.activation(layer(values))
        return self.layers[-1](values)


class FieldNetwork(nn.Module):
    """A shared network or a group of independent physical-field networks."""

    def __init__(
        self,
        input_dim: int,
        field_dims: Mapping[str, int],
        hidden_layers: list[int],
        activation: str,
        architecture: str = "shared",
    ) -> None:
        super().__init__()
        self.field_dims = dict(field_dims)
        self.architecture = architecture
        if architecture == "shared":
            self.shared = MLP(
                input_dim,
                sum(self.field_dims.values()),
                hidden_layers,
                activation,
            )
            self.networks = None
        elif architecture == "split":
            self.shared = None
            self.networks = nn.ModuleDict(
                {
                    name: MLP(input_dim, dimension, hidden_layers, activation)
                    for name, dimension in self.field_dims.items()
                }
            )
        else:
            raise ValueError("model.architecture must be 'shared' or 'split'.")

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.shared is not None:
            combined = self.shared(inputs)
            outputs: dict[str, torch.Tensor] = {}
            start = 0
            for name, dimension in self.field_dims.items():
                outputs[name] = combined[:, start : start + dimension]
                start += dimension
            return outputs
        assert self.networks is not None
        return {name: network(inputs) for name, network in self.networks.items()}

    def freeze_fields(self, fields: list[str]) -> None:
        if not fields:
            return
        if self.networks is None:
            raise ValueError("Individual fields can only be frozen with split architecture.")
        missing = sorted(set(fields) - set(self.networks))
        if missing:
            raise ValueError(f"Unknown fields requested for freezing: {missing}")
        for field in fields:
            for parameter in self.networks[field].parameters():
                parameter.requires_grad_(False)

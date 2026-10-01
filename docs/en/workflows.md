# Efficient retraining and recovery

## Changed conditions

There is no valid way to guarantee fast PINN convergence after an arbitrary geometry,
material, load, or boundary change. StructPINN therefore implements a measurable
continuation workflow instead of claiming one-shot generalization:

1. start from a nearby solved condition with `train TARGET.yaml --init SOURCE/model.pt`;
2. change one parameter family at a time and prefer several small continuation steps;
3. keep the network layout fixed and compare wall time and error against a from-scratch run;
4. fall back to a fresh initialization when the transfer run stalls or degrades validation.

The loader rejects incompatible problem types, network layouts, beam point-load feature
counts, and plate constitutive output layouts. Geometry, material, load values, and boundary
kinds may change because normalized coordinates keep the network input domain fixed. That
compatibility only means the tensors can be reused; it does not imply positive transfer.

For repeated queries over a wide parameter range, individual warm starts are not the final
solution. A genuinely parametric PINN or operator learner must include the changing condition
as an input and be tested on held-out parameter combinations.

## Reference solutions

`validation.plate_reference: auto` selects the independent Navier series when its assumptions
hold and otherwise uses the NumPy Rayleigh--Ritz energy solver. The Ritz trial space embeds
homogeneous essential conditions: power 1 at a simply supported edge, power 2 at a clamped
edge, and no multiplier at a free edge. Natural moment/effective-shear conditions arise weakly.

The high-order result is compared with a lower-order result and the differences are written to
`metrics.json` under `validation_reference.self_convergence`. This is a discretization
sensitivity check, not a proof of correctness. A strong validation claim needs all of:

- reference-order/quadrature convergence;
- PDE, boundary, geometry, and constitutive residuals on an independent grid;
- agreement with Navier where both methods apply;
- multiple random seeds;
- for consequential use, comparison with a mature plate FEM implementation and a mesh study.

A fully free static plate is rejected because its stiffness matrix contains rigid-body modes.
Free-edge corner forces and curved boundaries remain outside the implemented model.

## Interrupted training

With checkpointing enabled, `RUN/checkpoints/latest.pt` is written atomically and contains the
model, Adam optimizer state, collocation points, adaptive loss weights, best-so-far state,
early-stopping counters, history, elapsed time, and Python/NumPy/Torch RNG states. Resume with:

```bash
python -m structpinn.cli train CONFIG.yaml \
  --resume RUN/checkpoints/latest.pt --output RUN_RESUMED
```

`training.adam_epochs` may be increased and device/logging/checkpoint/output/validation settings
may change. Physics, architecture, optimizer-relevant training settings, and seed must match.
Periodic snapshots are controlled by:

```yaml
training:
  checkpointing:
    enabled: true
    every: 500
    keep_last: 2
```

Recovery is exact in the repository's deterministic CPU regression test. It is not promised to
be bitwise identical across PyTorch/CUDA versions or different devices. In-progress L-BFGS
state is not checkpointed; after interruption, resume from the last Adam checkpoint and rerun
L-BFGS.

# S4 CUDA Training Design

## Goal

Run S4 model training and model evaluation on CUDA when available, while retaining a deterministic CPU fallback for local development and CI.

## Device policy

- `auto` is the default: select `cuda` when `torch.cuda.is_available()` is true; otherwise select `cpu`.
- `cuda` requires an available CUDA device and fails clearly when none is available.
- `cpu` always selects CPU.
- CPU data generation, protocol degradation, and sample encoding remain on CPU. Each mini-batch moves to the selected device immediately before model execution.

## Data flow

`cloud_train_s4` resolves one device, passes it into both belief and policy epoch training, and records the resolved device in its reports and checkpoints. Training epochs move models to that device and move each batch there. Evaluation infers the model device and moves its temporary tensors or batches to the same device; reported metrics are moved back to CPU scalars.

## Error handling

Explicit CUDA requests on a host without CUDA raise `ValueError`. The default auto mode never fails solely because CUDA is absent.

## Tests

- A CPU device test proves default batch/model behavior still works.
- A CUDA test, skipped when CUDA is unavailable, proves a training step and evaluation use CUDA.
- A resolver test proves `auto`, explicit CPU, and unavailable explicit CUDA behavior.

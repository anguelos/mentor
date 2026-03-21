# mentor — Design Patterns and Conventions

This document captures the architectural decisions made for the `mentor`
framework.  It is intended as a prompt / context document for future
development sessions.

---

## 1. Core idea

`Mentee` is a `torch.nn.Module` subclass that carries its own training
history, validation history, software provenance, and constructor parameters
inside a single `.pt` checkpoint.  A model knows everything needed to
resume, report, or roll back without any external bookkeeping.

---

## 2. Class hierarchy

```
nn.Module
  └── Mentee                  # base: checkpointing, history, provenance
        └── MentorModality    # abstract: declares training_step @abstractmethod
              ├── Classifier  # cross-entropy + accuracy
              └── Regressor   # MSE + RMSE
```

Concrete user models subclass a modality (or `Mentee` directly if no
standard modality fits):

```python
class MyNet(Classifier):
    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        return self.fc(x.flatten(1))
```

`MyNet` needs only `__init__` and `forward`.  Everything else is inherited.

---

## 3. Constructor parameter capture

`Mentee.__init__` always walks the call stack upward through every frame
whose `co_name == "__init__"` and whose `self` is the same object being
constructed.  It captures the locals of the **topmost** such frame (the
most-derived concrete class).

- Works with zero-argument `super().__init__()` (implicit capture).
- Works even when an intermediate base passes explicit kwargs to
  `Mentee.__init__` — the walk is unconditional.
- Falls back to whatever explicit `**constructor_params` were passed when
  no `__init__` frame is found (factory functions, module-level calls).
- Guard: capture is skipped when `type(self) is Mentee` (direct
  instantiation of the base class is allowed).

This means subclasses never need to write `super().__init__(a=a, b=b)`;
a bare `super().__init__()` is sufficient and preferred.

---

## 4. Method naming and signatures

| Method | Owner | Returns |
|---|---|---|
| `forward(*args, **kwargs)` | concrete model | model output |
| `training_step(batch, loss_fn=None)` | modality / concrete model | `(loss_tensor, dict[str, float])` |
| `validation_step(batch, loss_fn=None)` | modality / concrete model | `dict[str, float]` |
| `create_train_objects(lr, step_size, gamma, loss_fn, overwrite_default_loss)` | modality / Mentee | `dict` |

`validation_step` default implementation in `Mentee` calls `training_step`
and unpacks the result, returning only the metrics dict (not the loss
tensor — no gradient needed during validation).

The **first key** of the metrics dict is the *principal metric* used for
best-checkpoint selection.

---

## 5. create_train_objects returns a dict

```python
result = model.create_train_objects(lr=1e-3)
# result == {"optimizer": Adam, "lr_scheduler": StepLR, "loss_fn": <fn or None>}
opt   = result["optimizer"]
sched = result["lr_scheduler"]
```

Rationale: a tuple is positionally fragile; a dict is self-documenting and
extensible without breaking callers.

---

## 6. Default loss function mechanism

`Mentee` carries `self._default_loss_fn = None`.

Resolution chain inside `training_step` (or via `_resolve_loss_fn`):

1. `loss_fn` argument passed directly to `training_step` — highest priority.
2. `self._default_loss_fn` — set as a side-effect of `create_train_objects`.
3. Modality-specific stateless fallback (e.g. `F.cross_entropy` in
   `Classifier`, `F.mse_loss` in `Regressor`) — lowest priority, no error.
   `Mentee.training_step` raises `RuntimeError` if none of the above apply.

`create_train_objects(overwrite_default_loss=False)`:
- If `loss_fn` is supplied and no default exists yet -> register it.
- If `loss_fn` is supplied and a default exists and `overwrite_default_loss=True` -> replace it.
- If `loss_fn` is supplied and a default exists and `overwrite_default_loss=False` -> keep existing.
- This preserves a parametric (trainable) loss that has already been
  partially optimised when the user resets the optimizer mid-training.

`_default_loss_fn` is saved in the checkpoint and restored on resume.

---

## 7. MentorModality — abstract mixin pattern

`MentorModality(Mentee, ABC)` marks `training_step` as `@abstractmethod`.
Concrete modalities (`Classifier`, `Regressor`) implement it and optionally
override `create_train_objects` to auto-register a sensible default loss.

The mixin sits in the MRO between the user's class and `Mentee`:

```
MyNet -> Classifier -> MentorModality -> Mentee -> nn.Module -> object
```

`super().__init__()` in `MyNet.__init__` propagates cooperatively through
this chain.  `Mentee.__init__` is always reached; its stack walk captures
`MyNet.__init__`'s locals.

Adding a new modality:
- Subclass `MentorModality`.
- Implement `training_step(self, batch, loss_fn=None) -> (Tensor, dict)`.
- Optionally override `create_train_objects` to auto-set a default loss.
- Optionally override `validation_step` if val differs from train.

---

## 8. Checkpoint format

Keys saved by `Mentee.save()`:

| Key | Type | Content |
|---|---|---|
| `class_name` | str | `type(self).__name__` |
| `class_module` | str | `type(self).__module__` |
| `state_dict` | dict | all tensors on CPU |
| `constructor_params` | dict | kwargs to re-instantiate |
| `train_history` | list[dict] | per-epoch train metrics |
| `validate_history` | dict[int, dict] | epoch -> val metrics |
| `software_history` | dict[int, dict] | epoch -> git/python/torch/host/user |
| `argv_history` | dict[int, list] | epoch -> sys.argv |
| `best_weights_so_far` | dict | CPU state_dict at best val epoch |
| `best_epoch_so_far` | int | epoch index of best val |
| `inference_state` | dict | arbitrary picklable objects |
| `output_schema` | dict | from `get_output_schema()` |
| `preprocessing_info` | dict | from `get_preprocessing_info()` |
| `default_loss_fn` | any | `_default_loss_fn` (may be None) |
| `optimizer_state` | dict | optional, if `optimizer=` passed |
| `lr_scheduler_state` | dict | optional, if `lr_scheduler=` passed |

All tensors are moved to CPU before saving (device-independent checkpoints).

---

## 9. Resume patterns

```python
# weights + history only (inference / evaluation)
model = MyNet.resume("run.pt", model_class=MyNet)

# full training resume (weights + optimizer + scheduler + default_loss_fn)
model, opt, sched = MyNet.resume_training(
    "run.pt", model_class=MyNet, device="cuda", lr=1e-3
)
```

`resume` re-instantiates via `model_class(**checkpoint["constructor_params"])`,
so the constructor param capture must be correct.

`resume_training` calls `create_train_objects` (which may set a new
optimizer) and then restores optimizer/scheduler state from the checkpoint,
so training continues with the same momentum buffers.

---

## 10. train_epoch / validate_epoch conventions

- `train_epoch` calls `training_step` per batch; accumulates gradients for
  `pseudo_batch_size` samples before each optimizer step.
- `validate_epoch` calls `validation_step` under `torch.no_grad()`.
- Both support `memfail="ignore"` to skip OOM samples.
- Both support `tensorboard_writer` and `verbose` (tqdm).
- `validate_epoch` caches results per epoch; pass `recalculate=True` to
  force re-evaluation.
- Best weights are updated whenever the principal val metric improves.

---

## 11. Reporting and CLI

`mentor.reporting.get_report_str(path, render_colors=False)` generates a
human-readable text report from a checkpoint without instantiating the model.

CLI entry points (registered in `pyproject.toml`):
- `mtr_report_file -path run.pt`  (add `-no_colors` to suppress ANSI)
- `mtr_plot_file_hist -paths a.pt b.pt -values train/loss validate/acc`

---

## 12. File layout

```
mentor/
  __init__.py        # exports: Mentee, MentorModality, Classifier, Regressor
  mentee.py          # Mentee base class
  modalities.py      # MentorModality, Classifier, Regressor
  reporting.py       # get_report_str, plot_history, CLI mains

examples/
  cifar/
    train_cifar.py              # original: Mentee subclass, manual steps
    train_cifar_classifier.py   # short version using Classifier modality

tests/
  helpers.py                    # LeNetMentee, MinimalMentee, make_loader
  conftest.py                   # shared fixtures
  unit_testing/
    test_mentee.py
    test_reporting.py
    test_modalities.py
  checkpoint/                   # roundtrip, device, inference_state tests
  integration/                  # training loop + resume tests
  cli/                          # subprocess tests for entry points
```

---

## 13. Conventions and non-obvious choices

- `super().__init__()` with no args is the recommended style in subclasses;
  explicit kwarg forwarding still works but is redundant.
- `create_train_objects` is idempotent w.r.t. the loss when called multiple
  times (default `overwrite_default_loss=False`).
- `validation_step` in modality classes does not need to be overridden if
  the train and val forward passes are identical.
- Modality classes are mixins — they carry no instance state of their own;
  all state lives in the concrete model or in `Mentee`'s buffers.
- New modalities should subclass `MentorModality`, not `Mentee` directly,
  so that `isinstance(model, MentorModality)` is a reliable capability check.
- The principal metric (first key of the metrics dict) drives best-model
  selection; put accuracy or your most important metric first.
- `memfail="ignore"` is safe for long runs on heterogeneous data; the count
  of skipped samples is included in the epoch metrics as `"memfails"`.

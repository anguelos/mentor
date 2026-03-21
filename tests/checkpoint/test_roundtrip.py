"""
Checkpoint roundtrip tests: save → load, verify every field is preserved exactly.
All IO uses BytesIO (in RAM).
"""
import io
import copy

import pytest
import torch

from helpers import LeNetMentee, make_loader
from mentor.mentee import Mentee


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_load(model, optimizer=None, lr_scheduler=None, model_class=LeNetMentee):
    buf = io.BytesIO()
    model.save(buf, optimizer=optimizer, lr_scheduler=lr_scheduler)
    buf.seek(0)
    return Mentee.resume(buf, model_class=model_class)


def _save_load_training(model, model_class=LeNetMentee, **kwargs):
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    return Mentee.resume_training(buf, model_class=model_class, **kwargs)


# ---------------------------------------------------------------------------
# Parameter weights
# ---------------------------------------------------------------------------

def test_state_dict_bit_identical(trained_model):
    model, opt, sched = trained_model
    loaded = _save_load(model, opt, sched)
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k].cpu(), loaded.state_dict()[k].cpu()), f"mismatch: {k}"


def test_state_dict_size_matches(trained_model):
    model, _, _ = trained_model
    loaded = _save_load(model)
    assert set(model.state_dict().keys()) == set(loaded.state_dict().keys())


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def test_train_history_preserved(trained_model):
    model, _, _ = trained_model
    loaded = _save_load(model)
    assert loaded._train_history == model._train_history


def test_validate_history_preserved(trained_model):
    model, _, _ = trained_model
    loaded = _save_load(model)
    assert loaded._validate_history == model._validate_history


def test_epoch_count_preserved(trained_model):
    model, _, _ = trained_model
    loaded = _save_load(model)
    assert loaded.current_epoch == model.current_epoch


def test_best_epoch_preserved(trained_model):
    model, _, _ = trained_model
    loaded = _save_load(model)
    assert loaded._best_epoch_so_far == model._best_epoch_so_far


def test_best_weights_preserved(trained_model):
    model, _, _ = trained_model
    loaded = _save_load(model)
    for k in model._best_weights_so_far:
        assert torch.equal(model._best_weights_so_far[k], loaded._best_weights_so_far[k])


# ---------------------------------------------------------------------------
# Provenance history
# ---------------------------------------------------------------------------

def test_software_history_preserved(trained_model):
    model, _, _ = trained_model
    loaded = _save_load(model)
    assert loaded._software_history == model._software_history


def test_argv_history_preserved(trained_model):
    model, _, _ = trained_model
    loaded = _save_load(model)
    assert loaded._argv_history == model._argv_history


def test_constructor_params_preserved(trained_model):
    model, _, _ = trained_model
    loaded = _save_load(model)
    assert loaded._constructor_params == model._constructor_params


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

def test_class_name_in_checkpoint(trained_model):
    model, _, _ = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    assert cp["class_name"] == "LeNetMentee"


def test_class_module_in_checkpoint(trained_model):
    model, _, _ = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    assert cp["class_module"] == "helpers"


def test_auto_class_resolution(trained_model):
    """Mentee.resume() with no model_class resolves helpers.LeNetMentee."""
    model, _, _ = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    loaded = Mentee.resume(buf)   # no model_class
    assert isinstance(loaded, LeNetMentee)


# ---------------------------------------------------------------------------
# Optimizer / scheduler state
# ---------------------------------------------------------------------------

def test_optimizer_state_in_checkpoint(trained_model):
    model, opt, sched = trained_model
    buf = io.BytesIO()
    model.save(buf, optimizer=opt, lr_scheduler=sched)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    assert "optimizer_state" in cp


def test_lr_scheduler_state_in_checkpoint(trained_model):
    model, opt, sched = trained_model
    buf = io.BytesIO()
    model.save(buf, optimizer=opt, lr_scheduler=sched)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    assert "lr_scheduler_state" in cp


def test_optimizer_state_absent_when_not_saved(lenet):
    buf = io.BytesIO()
    lenet.save(buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    assert "optimizer_state" not in cp


def test_resume_training_restores_epoch(trained_model):
    model, _, _ = trained_model
    result = _save_load_training(model)
    loaded_model = result[0]
    assert loaded_model.current_epoch == model.current_epoch


def test_resume_training_provides_optimizer(trained_model):
    model, _, _ = trained_model
    result = _save_load_training(model)
    assert isinstance(result[1], torch.optim.Optimizer)


def test_resume_training_optimizer_lr(trained_model):
    model, opt, _ = trained_model
    original_lr = opt.param_groups[0]["lr"]
    result = _save_load_training(model, lr=original_lr)
    loaded_opt = result[1]
    assert abs(loaded_opt.param_groups[0]["lr"] - original_lr) < 1e-9


# ---------------------------------------------------------------------------
# Output schema / preprocessing info
# ---------------------------------------------------------------------------

def test_output_schema_in_checkpoint(lenet):
    buf = io.BytesIO()
    lenet.save(buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    assert cp["output_schema"]["type"] == "classification"


def test_preprocessing_info_in_checkpoint(lenet):
    buf = io.BytesIO()
    lenet.save(buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    assert "input_size" in cp["preprocessing_info"]

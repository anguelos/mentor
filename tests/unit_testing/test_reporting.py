"""
Unit tests for mentor/reporting.py.
"""
import io

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import pytest
import torch

from helpers import LeNetMentee, make_loader
from mentor.reporting import (
    _check_class,
    _discover_values,
    _discover_values_multi,
    get_report_str,
    plot_history,
)


# ---------------------------------------------------------------------------
# _check_class
# ---------------------------------------------------------------------------

def test_check_class_ok():
    result = _check_class("mentor.mentee", "Mentee")
    assert result.startswith("OK")


def test_check_class_bad_module():
    result = _check_class("nonexistent.module.xyz", "Whatever")
    assert "NOT importable" in result


def test_check_class_bad_attr():
    result = _check_class("mentor.mentee", "NonExistentClass")
    assert "not found" in result


# ---------------------------------------------------------------------------
# _discover_values / _discover_values_multi
# ---------------------------------------------------------------------------

def test_discover_values_empty_checkpoint():
    cp = {}
    assert _discover_values(cp) == []


def test_discover_values_train_only():
    cp = {"train_history": [{"loss": 0.5, "acc": 0.8}]}
    vals = _discover_values(cp)
    assert "train/loss" in vals
    assert "train/acc" in vals
    assert not any(v.startswith("validate/") for v in vals)


def test_discover_values_train_and_validate():
    cp = {
        "train_history": [{"loss": 0.5}],
        "validate_history": {0: {"acc": 0.8}},
    }
    vals = _discover_values(cp)
    assert "train/loss" in vals
    assert "validate/acc" in vals


def test_discover_values_multi_union():
    cp1 = {"train_history": [{"loss": 0.5}]}
    cp2 = {"train_history": [{"loss": 0.3, "acc": 0.9}]}
    vals = _discover_values_multi([cp1, cp2])
    assert "train/loss" in vals
    assert "train/acc" in vals


def test_discover_values_multi_empty():
    assert _discover_values_multi([]) == []


# ---------------------------------------------------------------------------
# get_report_str
# ---------------------------------------------------------------------------

def _make_checkpoint_file(tmp_path, model, opt=None, sched=None):
    p = tmp_path / "model.pt"
    model.save(p, optimizer=opt, lr_scheduler=sched)
    return str(p)


def test_get_report_str_contains_class(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "LeNetMentee" in report


def test_get_report_str_contains_file_size(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "KB" in report


def test_get_report_str_contains_architecture_section(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "Architecture" in report
    assert "Parameters" in report


def test_get_report_str_no_history_section(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "Epochs trained: 0" in report


def test_get_report_str_with_train_history(tmp_path, trained_model):
    model, opt, sched = trained_model
    p = _make_checkpoint_file(tmp_path, model, opt, sched)
    report = get_report_str(p)
    assert "Epochs trained: 1" in report
    assert "First epoch" in report


def test_get_report_str_with_validate_history(tmp_path, trained_model):
    model, opt, sched = trained_model
    p = _make_checkpoint_file(tmp_path, model, opt, sched)
    report = get_report_str(p)
    assert "Epochs validated: 1" in report


def test_get_report_str_optimizer_present(tmp_path, trained_model):
    model, opt, sched = trained_model
    p = _make_checkpoint_file(tmp_path, model, opt, sched)
    report = get_report_str(p)
    assert "Optimizer state:    present" in report


def test_get_report_str_optimizer_absent(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "Optimizer state:    absent" in report


def test_get_report_str_inference_state(tmp_path, lenet):
    lenet.register_inference_state("labels", list(range(10)))
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "labels" in report


def test_get_report_str_output_schema(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "Output schema" in report
    assert "classification" in report


def test_get_report_str_preprocessing_info(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "Preprocessing info" in report


def test_get_report_str_plottable_series(tmp_path, trained_model):
    model, opt, sched = trained_model
    p = _make_checkpoint_file(tmp_path, model, opt, sched)
    report = get_report_str(p)
    assert "Plottable history" in report


# ---------------------------------------------------------------------------
# plot_history
# ---------------------------------------------------------------------------

def _make_fake_checkpoint(n_epochs: int = 3):
    cp = {
        "train_history": [{"loss": 1.0 - i * 0.1, "acc": 0.5 + i * 0.1} for i in range(n_epochs)],
        "validate_history": {i: {"acc": 0.5 + i * 0.1} for i in range(n_epochs)},
        "best_epoch_so_far": n_epochs - 1,
    }
    return cp


def _save_fake_cp(tmp_path, name, n_epochs=3):
    cp = _make_fake_checkpoint(n_epochs)
    p = tmp_path / name
    torch.save(cp, p)
    return str(p)


def test_plot_history_returns_figure(tmp_path):
    p = _save_fake_cp(tmp_path, "a.pt")
    fig = plot_history(["train/loss"], [p])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_history_auto_discover(tmp_path):
    p = _save_fake_cp(tmp_path, "a.pt")
    fig = plot_history([], [p])   # empty list -> auto-discover
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_history_multi_file(tmp_path):
    p1 = _save_fake_cp(tmp_path, "a.pt")
    p2 = _save_fake_cp(tmp_path, "b.pt")
    fig = plot_history(["train/loss"], [p1, p2])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_history_overlay(tmp_path):
    p = _save_fake_cp(tmp_path, "a.pt")
    fig = plot_history(["train/loss", "validate/acc"], [p], overlay=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_history_subplots_per_metric(tmp_path):
    p = _save_fake_cp(tmp_path, "a.pt")
    # 2 metrics, no overlay -> 2 subplots
    fig = plot_history(["train/loss", "train/acc"], [p], overlay=False)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_history_missing_metric_skipped(tmp_path):
    """Requesting a non-existent metric should not raise."""
    p = _save_fake_cp(tmp_path, "a.pt")
    fig = plot_history(["train/nonexistent"], [p])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

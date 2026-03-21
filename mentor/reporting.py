import importlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from mentor.mentee import _state_dict_architecture_lines

# ANSI colour codes
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"

_SECTION_RE = re.compile(r'^[A-Z][^:]*(?:\s*\([^)]*\))?:\s*$')


def _colorize_report(report: str) -> str:
    result = []
    for line in report.split("\n"):
        if not line.strip():
            result.append(line)
            continue
        # Section headers: "Architecture (...):" / "Inference state (N entries):" etc.
        if _SECTION_RE.match(line):
            result.append(f"{_BOLD}{_CYAN}{line}{_RESET}")
            continue
        # Top-level key: value lines
        if not line.startswith(" ") and ":" in line:
            key, _, val = line.partition(":")
            if "OK" in val and "NOT" not in val:
                val = val.replace("OK", f"{_GREEN}OK{_RESET}")
            elif "NOT importable" in val or "not found" in val:
                val = f"{_RED}{val}{_RESET}"
            elif val.strip() == "present":
                val = val.replace("present", f"{_GREEN}present{_RESET}")
            elif val.strip() == "absent":
                val = val.replace("absent", f"{_YELLOW}absent{_RESET}")
            result.append(f"{_BOLD}{key}{_RESET}:{val}")
            continue
        # Indented detail lines
        if line.startswith("  "):
            result.append(f"{_DIM}{line}{_RESET}")
            continue
        result.append(line)
    return "\n".join(result)


def _fmt_metrics(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def _check_class(class_module: str, class_name: str) -> str:
    try:
        mod = importlib.import_module(class_module)
    except ImportError as e:
        return f"NOT importable ({e})"
    if not hasattr(mod, class_name):
        return f"module '{class_module}' importable but '{class_name}' not found"
    return f"OK (found in '{class_module}')"


def get_report_str(path: str, render_colors: bool = False) -> str:
    """Generate a human-readable text report for a mentor checkpoint file.

    Loads the checkpoint with ``map_location=\"cpu\"`` so no GPU is required.
    Does **not** instantiate the model class --- all information is derived
    directly from the serialised data.

    Parameters
    ----------
    path : str
        Path to a ``.pt`` checkpoint file created by :meth:`~mentor.Mentee.save`.
    render_colors : bool, optional
        If ``True``, the returned string contains ANSI colour escape codes
        suitable for terminal display.  Defaults to ``False``.

    Returns
    -------
    str
        Multi-line report covering: file size, model class, architecture
        statistics, training and validation history, software provenance,
        plottable metric names, inference state inventory, output schema,
        preprocessing info, and checkpoint contents.

    Examples
    --------
    >>> from mentor.reporting import get_report_str
    >>> print(get_report_str(\"model.pt\"))
    >>> print(get_report_str(\"model.pt\", render_colors=True))
    """
    path = Path(path)
    lines: List[str] = []

    lines.append(f"Checkpoint: {path.resolve()}")
    lines.append(f"File size:  {path.stat().st_size / 1024:.1f} KB")
    lines.append("")

    checkpoint: Dict[str, Any] = torch.load(path, weights_only=False, map_location="cpu")

    # --- model class ---
    class_name   = checkpoint.get("class_name",   "<missing>")
    class_module = checkpoint.get("class_module", "<missing>")
    class_status = _check_class(class_module, class_name)
    lines.append(f"Model class:   {class_module}.{class_name}")
    lines.append(f"Importable:    {class_status}")

    # --- constructor params ---
    constructor_params = checkpoint.get("constructor_params", {})
    lines.append(f"Constructor:   {constructor_params}")
    lines.append("")

    # --- architecture (from state_dict, no instantiation) ---
    state_dict = checkpoint.get("state_dict", {})
    lines.append("Architecture (inferred from state_dict):")
    lines += _state_dict_architecture_lines(state_dict)
    lines.append("")

    # --- training history ---
    train_history: List[Dict[str, float]] = checkpoint.get("train_history", [])
    lines.append(f"Epochs trained: {len(train_history)}")
    if train_history:
        lines.append(f"  First epoch:  {_fmt_metrics(train_history[0])}")
        if len(train_history) > 1:
            lines.append(f"  Last epoch:   {_fmt_metrics(train_history[-1])}")
    lines.append("")

    # --- validation history ---
    validate_history: Dict[int, Dict[str, float]] = checkpoint.get("validate_history", {})
    best_epoch: int = checkpoint.get("best_epoch_so_far", -1)
    lines.append(f"Epochs validated: {len(validate_history)}")
    if validate_history:
        last_val_epoch = max(validate_history.keys())
        lines.append(f"  Last val epoch ({last_val_epoch}): {_fmt_metrics(validate_history[last_val_epoch])}")
    if best_epoch >= 0 and best_epoch in validate_history:
        lines.append(f"  Best epoch ({best_epoch}):         {_fmt_metrics(validate_history[best_epoch])}")
    lines.append("")

    # --- software history ---
    software_history: Dict[int, Dict[str, str]] = checkpoint.get("software_history", {})
    lines.append(f"Software snapshots: {len(software_history)}")
    if software_history:
        first_sw_epoch = min(software_history.keys())
        last_sw_epoch  = max(software_history.keys())
        sw0 = software_history[first_sw_epoch]
        lines.append(f"  First (epoch {first_sw_epoch}): torch={sw0.get('torch','?')}  python={sw0.get('python','?').split()[0]}")
        lines.append(f"              host={sw0.get('hostname','?')}  user={sw0.get('user','?')}  git={sw0.get('git_hash','?')[:12]}")
        if last_sw_epoch != first_sw_epoch:
            swN = software_history[last_sw_epoch]
            lines.append(f"  Last  (epoch {last_sw_epoch}): torch={swN.get('torch','?')}  python={swN.get('python','?').split()[0]}")
            lines.append(f"              host={swN.get('hostname','?')}  user={swN.get('user','?')}  git={swN.get('git_hash','?')[:12]}")
    lines.append("")

    # --- argv history ---
    argv_history: Dict[int, List[str]] = checkpoint.get("argv_history", {})
    lines.append(f"Argv snapshots: {len(argv_history)}")
    for epoch, argv in sorted(argv_history.items()):
        lines.append(f"  epoch {epoch}: {' '.join(argv)}")
    lines.append("")

    # --- plottable history ---
    plottable = _discover_values(checkpoint)
    lines.append(f"Plottable history ({len(plottable)} series):")
    if plottable:
        lines.append("  " + "  ".join(plottable))
    lines.append("")

    # --- inference state ---
    inference_state = checkpoint.get("inference_state", {})
    lines.append(f"Inference state ({len(inference_state)} entries):")
    for key, val in inference_state.items():
        type_name = type(val).__name__
        try:
            import sys as _sys
            size = _sys.getsizeof(val)
            lines.append(f"  {key}: {type_name}  (~{size} bytes)")
        except Exception:
            lines.append(f"  {key}: {type_name}")
    lines.append("")

    # --- output schema & preprocessing info ---
    output_schema = checkpoint.get("output_schema", {})
    preprocessing_info = checkpoint.get("preprocessing_info", {})
    lines.append(f"Output schema:      {output_schema if output_schema else '(not provided)'}")
    lines.append(f"Preprocessing info: {preprocessing_info if preprocessing_info else '(not provided)'}")
    lines.append("")

    # --- checkpoint contents ---
    has_opt   = "optimizer_state"    in checkpoint
    has_sched = "lr_scheduler_state" in checkpoint
    lines.append(f"Optimizer state:    {'present' if has_opt   else 'absent'}")
    lines.append(f"LR scheduler state: {'present' if has_sched else 'absent'}")

    report = "\n".join(lines)
    if render_colors:
        report = _colorize_report(report)
    return report


def main_report_file() -> None:
    from fargv import fargv
    params = {
        "path":      ["",    "Path to mentor checkpoint file"],
        "no_colors": [False, "Disable terminal colour output"],
        "verbose":   [False, "Print extra detail"],
    }
    p, _ = fargv(params)
    if not p.path:
        print("Error: -path is required.")
        raise SystemExit(1)
    report = get_report_str(p.path, render_colors=not p.no_colors)
    print(report)


def _discover_values(checkpoint: Dict[str, Any]) -> List[str]:
    """Return all metric names in train/validate history as split/metric strings."""
    seen: "dict[str, None]" = {}  # ordered set via dict keys
    train_history = checkpoint.get("train_history", [])
    if train_history:
        for key in train_history[0]:
            seen[f"train/{key}"] = None
    validate_history = checkpoint.get("validate_history", {})
    if validate_history:
        for key in next(iter(validate_history.values())):
            seen[f"validate/{key}"] = None
    return list(seen)


def _discover_values_multi(checkpoints: List[Dict[str, Any]]) -> List[str]:
    """Union of plottable metrics across multiple checkpoints."""
    seen: "dict[str, None]" = {}
    for cp in checkpoints:
        for v in _discover_values(cp):
            seen[v] = None
    return list(seen)


def plot_history(
    values: List[str],
    paths: List[str],
    overlay: bool = False,
) -> "matplotlib.figure.Figure":
    """Plot training/validation history from one or more checkpoint files.

    Checkpoints are loaded with ``map_location=\"cpu\"``; no GPU is required.
    Each file gets a distinct colour; each metric a distinct line style.
    Vertical dashed lines mark the best-epoch for each file when available.

    Parameters
    ----------
    values : list[str]
        Metric names in ``split/metric`` form, e.g.
        ``[\"train/loss\", \"validate/acc\"]\".  Pass an empty list to
        auto-discover all available metrics (union across all files).
    paths : list[str]
        One or more paths to ``.pt`` checkpoint files.
    overlay : bool, optional
        If ``True``, all metrics and files share a single axis.
        If ``False`` (default), one subplot per metric with all files
        overlaid on each subplot.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure.  Call ``fig.savefig(...)`` or ``plt.show()``
        to display it.

    Examples
    --------
    >>> from mentor.reporting import plot_history
    >>> fig = plot_history([], [\"run1.pt\", \"run2.pt\"])
    >>> fig.savefig(\"comparison.png\", dpi=150, bbox_inches=\"tight\")

    .. image:: /_static/plot_history_example.png
       :alt: plot_history example --- two CIFAR runs, train/loss and validate/loss overlaid
       :align: center
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    line_styles = ["-", "--", "-.", ":"]

    checkpoints = [torch.load(p, weights_only=False, map_location="cpu") for p in paths]
    stems = [Path(p).stem for p in paths]

    if not values:
        values = _discover_values_multi(checkpoints)

    # file_color[i] -> color for file i
    file_palette = sns.color_palette("tab10", n_colors=max(len(paths), 1))

    def _extract(cp: Dict[str, Any], v: str) -> List[tuple]:
        split, metric = v.split("/", 1)
        train_history     = cp.get("train_history", [])
        validate_history  = cp.get("validate_history", {})
        if split == "train":
            return [(i, m[metric]) for i, m in enumerate(train_history) if metric in m]
        if split == "validate":
            return [(ep, m[metric]) for ep, m in sorted(validate_history.items()) if metric in m]
        return []

    sns.set_theme(style="darkgrid")
    title = "  vs  ".join(stems)

    if overlay or len(values) <= 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        for fi, (cp, stem, color) in enumerate(zip(checkpoints, stems, file_palette)):
            for vi, v in enumerate(values):
                data = _extract(cp, v)
                if not data:
                    continue
                epochs, vals = zip(*data)
                ls = line_styles[vi % len(line_styles)]
                label = f"{stem}: {v}" if len(paths) > 1 else v
                sns.lineplot(x=list(epochs), y=list(vals), ax=ax, label=label,
                             color=color, linestyle=ls, marker="o", markersize=4)
            best = cp.get("best_epoch_so_far", -1)
            if best >= 0:
                ax.axvline(x=best, linestyle=":", color=color, alpha=0.5,
                           label=f"{stem} best={best}")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize="small")
        fig.suptitle(title)
    else:
        n = len(values)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, v in zip(axes, values):
            for cp, stem, color in zip(checkpoints, stems, file_palette):
                data = _extract(cp, v)
                if not data:
                    continue
                epochs, vals = zip(*data)
                label = stem if len(paths) > 1 else v
                sns.lineplot(x=list(epochs), y=list(vals), ax=ax, label=label,
                             color=color, marker="o", markersize=4)
                best = cp.get("best_epoch_so_far", -1)
                if best >= 0:
                    ax.axvline(x=best, linestyle="--", color=color, alpha=0.5)
            ax.set_ylabel(v)
            ax.legend(loc="upper right", fontsize="small")
        axes[-1].set_xlabel("Epoch")
        fig.suptitle(title, y=1.01)

    fig.tight_layout()
    return fig


def main_plot_file_hist() -> None:
    from fargv import fargv
    import matplotlib.pyplot as plt

    params = {
        "paths":   [set([]),  "Checkpoint files to compare, e.g. -paths a.pt b.pt c.pt"],
        "values":  [set([]),  "Metrics to plot, e.g. train/loss validate/accuracy (empty = all)"],
        "overlay": [False,    "Overlay all metrics and files on a single axis"],
        "output":  ["",       "Save figure to this path (empty = show interactively)"],
        "verbose": [False,    "Print discovered metrics and file list"],
    }
    p, _ = fargv(params)
    paths = list(p.paths)
    if not paths:
        print("Error: -paths requires at least one file, e.g. -paths a.pt b.pt c.pt")
        raise SystemExit(1)

    values = list(p.values)
    if p.verbose:
        print(f"Files:   {paths}")
        print(f"Metrics: {values or '(all)'}")

    fig = plot_history(values, paths, overlay=p.overlay)

    if p.output:
        fig.savefig(p.output, dpi=150, bbox_inches="tight")
        if p.verbose:
            print(f"Saved to {p.output}")
    else:
        plt.show()

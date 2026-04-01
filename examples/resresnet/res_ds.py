"""PyTorch Dataset and evaluation utilities for resolution ground-truth data."""

import glob as glob_module
import math
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
"""Per-channel ImageNet normalisation mean (RGB)."""

IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
"""Per-channel ImageNet normalisation standard deviation (RGB)."""

#: Internal type alias for a parsed sample tuple.
_Sample = Tuple[str, Optional[List[int]], float, Optional[str]]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_ppi(
    gt_json: Dict[str, Any],
    known_sizes_cm: Dict[str, float],
) -> Optional[float]:
    """Compute mean PPI from ruler/reference-object annotations.

    Parameters
    ----------
    gt_json : dict
        Ground-truth annotation dict with keys ``rect_LTRB``,
        ``rect_classes``, and ``class_names``.
    known_sizes_cm : dict
        Mapping from class name to physical long-edge size in centimetres.

    Returns
    -------
    float or None
        Mean PPI over all matched reference objects, or *None* when no
        usable annotation is found.

    Examples
    --------
    >>> gt = {
    ...     "rect_LTRB": [[0, 0, 196, 10]],
    ...     "rect_classes": [0],
    ...     "class_names": ["5cm"],
    ... }
    >>> round(_compute_ppi(gt, {"5cm": 5.0}))
    100
    """
    ppis: List[float] = []
    for rect, cls in zip(gt_json["rect_LTRB"], gt_json["rect_classes"]):
        name: str = gt_json["class_names"][cls]
        if name not in known_sizes_cm:
            continue
        L, T, R, B = rect
        long_px: float = max(R - L, B - T)
        if long_px == 0:
            continue
        ppis.append(long_px / (known_sizes_cm[name] / 2.54))
    return sum(ppis) / len(ppis) if ppis else None


def _dominant_rect(
    layout: Dict[str, Any],
    class_name: str,
    dominant_area_ratio: float,
) -> Optional[List[float]]:
    """Return the single dominant rectangle of a given class, if one exists.

    A rectangle is considered *dominant* when its area exceeds
    *dominant_area_ratio* of the total area covered by all rectangles of
    that class.

    Parameters
    ----------
    layout : dict
        Layout annotation dict with keys ``class_names``, ``rect_LTRB``,
        and ``rect_classes``.
    class_name : str
        The class label to look for (e.g. ``"Img:WritableArea"``).
    dominant_area_ratio : float
        Minimum fraction of total class area the largest rectangle must
        occupy.  Must be in ``(0, 1]``.

    Returns
    -------
    list of float or None
        ``[L, T, R, B]`` of the dominant rectangle, or *None* when no
        dominant rectangle is found.
    """
    cls_idx: int = layout["class_names"].index(class_name)
    rects = [
        r
        for r, c in zip(layout["rect_LTRB"], layout["rect_classes"])
        if c == cls_idx
    ]
    if not rects:
        return None
    areas: List[float] = [(R - L) * (B - T) for L, T, R, B in rects]
    total: float = sum(areas)
    if total == 0:
        return None
    best: int = max(range(len(areas)), key=lambda i: areas[i])
    if areas[best] / total <= dominant_area_ratio:
        return None
    return rects[best]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ResDs(Dataset):
    """PyTorch Dataset of charter images with ground-truth PPI labels.

    Each sample is parsed from a ``*.res.gt.json`` file located in the
    charter directory alongside the image.  Optionally the image is
    cropped to a dominant layout region before being returned.

    Parameters
    ----------
    gt_paths : list of str
        Paths to ``*.res.gt.json`` annotation files.
    image_crop : str
        Which region to crop to.  Must be one of :attr:`CROP_CLASSES`.
        Use ``"img"`` to return the full image without cropping.
    return_image_layout : bool, optional
        When *True*, ``__getitem__`` returns a third element containing
        the raw layout dict.  Default is *False*.
    input_transform : callable, optional
        Transform applied to the PIL image before it is returned.
        Typically a :func:`make_train_transform` or
        :func:`make_inference_transform` pipeline.
    dominant_area_ratio : float, optional
        Area fraction threshold for :func:`_dominant_rect`.  Defaults to
        :attr:`DOMINANT_AREA_RATIO`.
    check_images : bool, optional
        When *True*, each image is opened and fully decoded during
        construction; broken images are silently dropped.  Default is
        *False*.

    Attributes
    ----------
    KNOWN_SIZES_CM : dict
        Physical sizes (in cm) of supported ruler/reference classes.
    DOMINANT_AREA_RATIO : float
        Default area-dominance threshold.
    CROP_CLASSES : tuple of str
        Valid values for *image_crop*.
    samples : list of tuple
        Parsed samples as ``(img_path, crop_ltrb, ppi, layout_path)``.

    Examples
    --------
    >>> ds = ResDs.from_root("/data/fsdb", "**/*.res.gt.json", image_crop="img")
    >>> len(ds)  # doctest: +SKIP
    1000
    >>> img, ppi = ds[0]  # doctest: +SKIP
    """

    KNOWN_SIZES_CM: Dict[str, float] = {
        "5cm": 5.0,
        "1cm": 1.0,
        "5in": 5 * 2.54,
        "1in": 2.54,
    }
    DOMINANT_AREA_RATIO: float = 0.5
    CROP_CLASSES: Tuple[str, ...] = ("Img:WritableArea", "Wr:OldText", "img")

    def __init__(
        self,
        gt_paths: List[str],
        image_crop: str,
        return_image_layout: bool = False,
        input_transform: Optional[Callable] = None,
        dominant_area_ratio: Optional[float] = None,
        check_images: bool = True,
    ) -> None:
        assert image_crop in self.CROP_CLASSES, (
            f"image_crop must be one of {self.CROP_CLASSES}"
        )
        self.image_crop: str = image_crop
        self.return_image_layout: bool = return_image_layout
        self._input_transform: Optional[Callable] = input_transform
        self.dominant_area_ratio: float = (
            dominant_area_ratio if dominant_area_ratio is not None else self.DOMINANT_AREA_RATIO
        )
        self.check_images: bool = check_images
        self.samples: List[_Sample] = []
        for gt_path in gt_paths:
            sample = self._parse_sample(gt_path)
            if sample is not None:
                self.samples.append(sample)

    @classmethod
    def from_root(
        cls,
        data_root_path: str,
        resolution_gt_glob: str,
        image_crop: str,
        return_image_layout: bool = False,
        input_transform: Optional[Callable] = None,
        dominant_area_ratio: Optional[float] = None,
        check_images: bool = True,
    ) -> "ResDs":
        """Construct a :class:`ResDs` by globbing for GT files under a root directory.

        Parameters
        ----------
        data_root_path : str
            Root directory of the FSDB archive.
        resolution_gt_glob : str
            Glob pattern relative to *data_root_path* used to discover
            ``*.res.gt.json`` files (e.g. ``"**/*.res.gt.json"``).
        image_crop : str
            See :class:`ResDs`.
        return_image_layout : bool, optional
            See :class:`ResDs`.
        input_transform : callable, optional
            See :class:`ResDs`.
        dominant_area_ratio : float, optional
            See :class:`ResDs`.
        check_images : bool, optional
            See :class:`ResDs`.

        Returns
        -------
        ResDs
            Fully initialised dataset instance.

        Examples
        --------
        >>> ds = ResDs.from_root(
        ...     "/data/fsdb",
        ...     "**/*.res.gt.json",
        ...     image_crop="img",
        ... )  # doctest: +SKIP
        """
        gt_paths: List[str] = sorted(
            glob_module.glob(
                os.path.join(data_root_path, resolution_gt_glob), recursive=True
            )
        )
        return cls(
            gt_paths,
            image_crop,
            return_image_layout=return_image_layout,
            input_transform=input_transform,
            dominant_area_ratio=dominant_area_ratio,
            check_images=check_images,
        )

    def _parse_sample(self, gt_path: str) -> Optional[_Sample]:
        """Parse a single GT file into a sample tuple.

        Parameters
        ----------
        gt_path : str
            Path to a ``*.res.gt.json`` file.

        Returns
        -------
        tuple or None
            ``(img_path, crop_ltrb, ppi, layout_path)`` on success, or
            *None* when the sample cannot be used (missing image, missing
            layout, unreadable JSON, …).
        """
        try:
            gt: Dict[str, Any] = json.load(open(gt_path))
        except Exception:
            return None

        ppi: Optional[float] = _compute_ppi(gt, self.KNOWN_SIZES_CM)
        if ppi is None:
            return None

        charter_dir: Path = Path(gt_path).parent
        img_md5: str = Path(gt_path).name.split(".")[0]

        img_paths: List[Path] = list(charter_dir.glob(f"{img_md5}.img.*"))
        if not img_paths:
            return None
        img_path: Path = img_paths[0]

        layout_path: Path = charter_dir / f"{img_md5}.layout.pred.json"

        crop_ltrb: Optional[List[int]] = None
        if self.image_crop != "img":
            if not layout_path.exists():
                return None
            try:
                layout: Dict[str, Any] = json.load(open(layout_path))
            except Exception:
                return None
            rect = _dominant_rect(layout, self.image_crop, self.dominant_area_ratio)
            if rect is None:
                return None
            crop_ltrb = [int(x) for x in rect]

        if self.check_images:
            try:
                Image.open(str(img_path)).load()
            except Exception:
                return None

        return (
            str(img_path),
            crop_ltrb,
            float(ppi),
            str(layout_path) if layout_path.exists() else None,
        )

    @property
    def input_transform(self) -> Optional[Callable]:
        """Transform applied to PIL images before they are returned."""
        return self._input_transform

    @input_transform.setter
    def input_transform(self, transform: Optional[Callable]) -> None:
        self._input_transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[Any, float], Tuple[Any, float, Dict]]:
        """Return the sample at position *idx*.

        Parameters
        ----------
        idx : int
            Index into :attr:`samples`.

        Returns
        -------
        tuple
            ``(img, ppi)`` when :attr:`return_image_layout` is *False*,
            or ``(img, ppi, layout)`` when it is *True*.  *img* is either
            a PIL image or the output of :attr:`input_transform`.
        """
        img_path, crop_ltrb, ppi, layout_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if crop_ltrb is not None:
            img = img.crop(crop_ltrb)

        if self._input_transform is not None:
            img = self._input_transform(img)

        if self.return_image_layout:
            layout: Dict[str, Any] = json.load(open(layout_path)) if layout_path else {}
            return img, ppi, layout
        return img, ppi

    def random_split(
        self, ratio: float, seed: int = -1
    ) -> Tuple["ResDs", "ResDs"]:
        """Split the dataset into two non-overlapping views.

        Parameters
        ----------
        ratio : float
            Fraction of samples assigned to the first split.  Must be in
            ``(0, 1)``.
        seed : int, optional
            Random seed for reproducibility.  Pass ``-1`` (default) for a
            non-deterministic split.

        Returns
        -------
        tuple of (ResDs, ResDs)
            ``(train_view, val_view)`` sharing the same configuration but
            holding disjoint sample lists.

        Examples
        --------
        >>> train, val = ds.random_split(0.8, seed=42)  # doctest: +SKIP
        >>> len(train) + len(val) == len(ds)  # doctest: +SKIP
        True
        """
        indices: List[int] = list(range(len(self.samples)))
        rng = random.Random(seed if seed >= 0 else None)
        rng.shuffle(indices)
        n: int = int(len(indices) * ratio)
        return (
            self._view([self.samples[i] for i in indices[:n]]),
            self._view([self.samples[i] for i in indices[n:]]),
        )

    def _collect_pairs(self, pred_ext: str) -> List[Tuple[float, float]]:
        """Collect ``(gt_ppi, pred_ppi)`` pairs for all samples with a prediction.

        Parameters
        ----------
        pred_ext : str
            File suffix (including leading dot) appended to the image stem
            to locate the prediction JSON (e.g. ``".res.pred_layout.json"``).

        Returns
        -------
        list of (float, float)
            Each element is ``(gt_ppi, pred_ppi)``.  Samples without a
            readable prediction file are silently skipped.
        """
        pairs: List[Tuple[float, float]] = []
        for img_path, _crop, gt_ppi, _layout in self.samples:
            pred_path: str = img_path.split(".img.")[0] + pred_ext
            try:
                pred: Dict[str, Any] = json.load(open(pred_path))
                pairs.append((gt_ppi, float(pred["ppi"])))
            except Exception:
                continue
        return pairs

    def evaluate(self, pred_ext: str = ".res.pred.json") -> Dict[str, Any]:
        """Compare per-image resolution predictions against ground-truth PPI.

        For each sample the prediction file is located by replacing the
        ``.img.<ext>`` suffix of the image path with *pred_ext*.

        Parameters
        ----------
        pred_ext : str, optional
            File suffix (including leading dot) used to locate prediction
            JSON files produced by ``ddp_res_offline``.  Default is
            ``".res.pred.json"``.

        Returns
        -------
        dict
            Keys and semantics:

            ``n_total``
                Number of samples in this dataset view.
            ``n_predicted``
                Samples for which a readable prediction exists.
            ``coverage``
                ``n_predicted / n_total`` (0.0 when empty).
            ``mean_gt_ppi``
                Mean GT PPI over all samples.
            ``gt_ppi_std``
                Std-dev of GT PPI; same units as PPI errors.
            ``gt_ppi_var``
                Variance of GT PPI; baseline for MSE — if
                ``rmse² ≈ gt_ppi_var`` the model is no better than
                predicting the mean.
            ``mae``
                Mean absolute error in PPI.
            ``mape``
                Mean absolute percentage error (0–100 scale).
            ``rmse``
                Root-mean-squared error in PPI.
            ``median_ae``
                Median absolute error in PPI.
            ``r2``
                Coefficient of determination
                ``(1 − MSE / Var(GT))``; 1 = perfect,
                0 = mean predictor, negative = worse than mean.
            ``log2_mae``
                Mean ``|log2(pred/gt)|``; 1.0 means off by one
                doubling.
            ``log2_rmse``
                RMS of ``log2(pred/gt)`` errors.

            Error metrics are *None* when ``n_predicted == 0``.

        Examples
        --------
        >>> metrics = ds.evaluate(".res.pred_layout.json")  # doctest: +SKIP
        >>> metrics["r2"]  # doctest: +SKIP
        0.91
        """
        errors: List[Tuple[float, float]] = self._collect_pairs(pred_ext)

        n_total: int = len(self.samples)
        n_predicted: int = len(errors)
        coverage: float = n_predicted / n_total if n_total else 0.0
        all_gt: List[float] = [s[2] for s in self.samples]
        mean_gt_ppi: Optional[float] = sum(all_gt) / n_total if n_total else None
        gt_ppi_var: Optional[float] = (
            sum((v - mean_gt_ppi) ** 2 for v in all_gt) / n_total if n_total else None
        )
        gt_ppi_std: Optional[float] = gt_ppi_var ** 0.5 if gt_ppi_var is not None else None

        if not errors:
            return {
                "n_total": n_total,
                "n_predicted": 0,
                "coverage": coverage,
                "mean_gt_ppi": mean_gt_ppi,
                "gt_ppi_std": gt_ppi_std,
                "gt_ppi_var": gt_ppi_var,
                "mae": None,
                "mape": None,
                "rmse": None,
                "median_ae": None,
                "r2": None,
                "log2_mae": None,
                "log2_rmse": None,
            }

        abs_errors: List[float] = [abs(pred - gt) for gt, pred in errors]
        sq_errors: List[float] = [(pred - gt) ** 2 for gt, pred in errors]
        pct_errors: List[float] = [
            abs(pred - gt) / gt * 100.0 for gt, pred in errors if gt != 0
        ]
        log2_errors: List[float] = [
            math.log2(pred / gt) for gt, pred in errors if gt > 0 and pred > 0
        ]

        abs_errors_sorted: List[float] = sorted(abs_errors)
        mid: int = len(abs_errors_sorted) // 2
        if len(abs_errors_sorted) % 2 == 0:
            median_ae: float = (abs_errors_sorted[mid - 1] + abs_errors_sorted[mid]) / 2.0
        else:
            median_ae = abs_errors_sorted[mid]

        mse: float = sum(sq_errors) / len(sq_errors)
        r2: Optional[float] = 1.0 - mse / gt_ppi_var if gt_ppi_var else None

        return {
            "n_total": n_total,
            "n_predicted": n_predicted,
            "coverage": coverage,
            "mean_gt_ppi": mean_gt_ppi,
            "gt_ppi_std": gt_ppi_std,
            "gt_ppi_var": gt_ppi_var,
            "mae": sum(abs_errors) / len(abs_errors),
            "mape": sum(pct_errors) / len(pct_errors) if pct_errors else None,
            "rmse": mse ** 0.5,
            "median_ae": median_ae,
            "r2": r2,
            "log2_mae": (
                sum(abs(e) for e in log2_errors) / len(log2_errors) if log2_errors else None
            ),
            "log2_rmse": (
                (sum(e ** 2 for e in log2_errors) / len(log2_errors)) ** 0.5
                if log2_errors
                else None
            ),
        }

    def _view(self, samples: List[_Sample]) -> "ResDs":
        """Create a lightweight view sharing configuration with the parent.

        Parameters
        ----------
        samples : list of tuple
            Subset of :attr:`samples` to expose in the view.

        Returns
        -------
        ResDs
            New instance with the given *samples* and all other attributes
            copied from *self*.
        """
        obj: ResDs = ResDs.__new__(ResDs)
        obj.image_crop = self.image_crop
        obj.return_image_layout = self.return_image_layout
        obj._input_transform = self._input_transform
        obj.dominant_area_ratio = self.dominant_area_ratio
        obj.check_images = self.check_images
        obj.samples = samples
        return obj


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------


def make_train_transform(
    patch_size: Optional[int] = None,
    jitter_strength: float = 0.0,
    max_rotation: float = 0.0,
    blur_p: float = 0.0,
    grayscale_p: float = 0.0,
    erasing_p: float = 0.0,
) -> tv_transforms.Compose:
    """Build a training image transform pipeline.

    All augmentation parameters default to ``0.0`` / ``0`` which means the
    augmentation is not applied.  Pass non-zero values to enable each one
    independently.

    Parameters
    ----------
    patch_size : int, optional
        If given, a :class:`~torchvision.transforms.RandomCrop` to this
        square size is inserted (with padding when the image is smaller).
        ``None`` (default) skips cropping.
    jitter_strength : float, optional
        Magnitude for :class:`~torchvision.transforms.ColorJitter`
        (brightness, contrast, saturation).  ``0.0`` disables jitter.
    max_rotation : float, optional
        Maximum rotation angle in degrees for
        :class:`~torchvision.transforms.RandomRotation`.  ``0.0`` disables
        rotation.
    blur_p : float, optional
        Probability of :class:`~torchvision.transforms.GaussianBlur`.
        ``0.0`` disables blurring.
    grayscale_p : float, optional
        Probability of :class:`~torchvision.transforms.RandomGrayscale`
        (channels kept at 3).  ``0.0`` disables greyscale conversion.
    erasing_p : float, optional
        Probability of :class:`~torchvision.transforms.RandomErasing`
        applied after tensorisation.  ``0.0`` disables erasing.

    Returns
    -------
    torchvision.transforms.Compose
        Transform pipeline suitable for training.

    Examples
    --------
    >>> t = make_train_transform(patch_size=512)
    >>> img = Image.new("RGB", (600, 800))
    >>> tensor = t(img)
    >>> tensor.shape
    torch.Size([3, 512, 512])
    """
    ops: List[Any] = [
        tv_transforms.RandomHorizontalFlip(),
        tv_transforms.RandomVerticalFlip(),
    ]
    if max_rotation > 0.0:
        ops.append(tv_transforms.RandomRotation(degrees=max_rotation))
    if jitter_strength > 0.0:
        ops.append(tv_transforms.ColorJitter(
            brightness=jitter_strength,
            contrast=jitter_strength,
            saturation=jitter_strength,
        ))
    if grayscale_p > 0.0:
        ops.append(tv_transforms.RandomGrayscale(p=grayscale_p))
    if blur_p > 0.0:
        ops.append(tv_transforms.RandomApply([tv_transforms.GaussianBlur(kernel_size=5)], p=blur_p))
    if patch_size is not None:
        ops.append(tv_transforms.RandomCrop(patch_size, pad_if_needed=True))
    ops += [
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if erasing_p > 0.0:
        ops.append(tv_transforms.RandomErasing(p=erasing_p))
    return tv_transforms.Compose(ops)


def make_inference_transform(
    patch_size: Optional[int] = None,
) -> tv_transforms.Compose:
    """Build an inference image transform pipeline.

    Parameters
    ----------
    patch_size : int, optional
        If given, a :class:`~torchvision.transforms.CenterCrop` to this
        square size is inserted before tensorisation.

    Returns
    -------
    torchvision.transforms.Compose
        Transform pipeline suitable for inference.

    Examples
    --------
    >>> t = make_inference_transform(patch_size=512)
    >>> img = Image.new("RGB", (600, 800))
    >>> tensor = t(img)
    >>> tensor.shape
    torch.Size([3, 512, 512])
    """
    ops: List[Any] = []
    if patch_size is not None:
        ops.append(tv_transforms.CenterCrop(patch_size))
    ops += [
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return tv_transforms.Compose(ops)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main_res_evaluate() -> None:
    """CLI entry-point for evaluating resolution predictions against GT.

    Loads all ``*.res.gt.json`` files found under *fsdb_root* matching
    *gt_glob*, then evaluates each prediction extension listed in
    *pred_exts* and prints a comparison table.  Optionally shows or saves
    a scatter plot of GT vs predicted PPI.
    """
    import fargv
    import sys

    p = {
        "fsdb_root": "",
        "gt_glob": "**/*.res.gt.json",
        "image_crop": ("img", "Img:WritableArea", "Wr:OldText"),
        "pred_exts": set([]),
        "no_img_check": False,
        "plot": False,
        "plotfname": "",
        "verbose": False,
    }
    args, _ = fargv.fargv(p)

    pred_ext_list: List[str] = sorted(args.pred_exts)

    all_gt_paths: List[str] = sorted(
        glob_module.glob(os.path.join(args.fsdb_root, args.gt_glob), recursive=True)
    )

    ds = ResDs(all_gt_paths, image_crop=args.image_crop, check_images=not args.no_img_check)

    if args.verbose:
        print(f"Loaded {len(ds)} samples from {args.fsdb_root}", file=sys.stderr)

    results: Dict[str, Dict[str, Any]] = {
        ext: ds.evaluate(pred_ext=ext) for ext in pred_ext_list
    }

    metrics: List[str] = [
        "n_total", "n_predicted", "coverage", "mean_gt_ppi",
        "gt_ppi_std", "gt_ppi_var", "mae", "mape", "rmse",
        "median_ae", "r2", "log2_mae", "log2_rmse",
    ]
    ext_w: int = max(len(e) for e in pred_ext_list) + 2
    col_w: int = 12
    header: str = f"{'ext':<{ext_w}}" + "".join(f"{m:>{col_w}}" for m in metrics)
    separator: str = "-" * len(header)
    print(header)
    print(separator)
    for ext, r in results.items():
        row: str = f"{ext:<{ext_w}}"
        for m in metrics:
            val = r[m]
            if val is None:
                row += f"{'N/A':>{col_w}}"
            elif isinstance(val, int):
                row += f"{val:>{col_w}d}"
            else:
                row += f"{val:>{col_w}.3f}"
        print(row)

    if args.plot:
        import matplotlib.pyplot as plt

        try:
            import seaborn as sns
            sns.set_theme()
        except ImportError:
            pass

        fig, ax = plt.subplots()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        all_gt_vals: List[float] = []
        for i, ext in enumerate(pred_ext_list):
            pairs = ds._collect_pairs(ext)
            if not pairs:
                continue
            gt_vals, pred_vals = zip(*pairs)
            all_gt_vals.extend(gt_vals)
            label = ext if len(pred_ext_list) > 1 else None
            ax.scatter(gt_vals, pred_vals, alpha=0.4, label=label, color=colors[i % len(colors)])

        if all_gt_vals:
            lo, hi = min(all_gt_vals), max(all_gt_vals)
            ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1, label="ideal")

        ax.set_xlabel("GT PPI")
        ax.set_ylabel("Predicted PPI")
        ax.set_title("Resolution prediction vs ground truth")
        if len(pred_ext_list) > 1 or all_gt_vals:
            ax.legend()

        if args.plotfname:
            plt.savefig(args.plotfname, bbox_inches="tight")
            if args.verbose:
                print(f"Plot saved to {args.plotfname}", file=sys.stderr)
        else:
            plt.show()

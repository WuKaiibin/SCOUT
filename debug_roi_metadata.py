"""Utility to inspect CNMF-E session files for Napari metadata expectations.

This script is meant to help diagnose why Napari layers created by
``cross_day_match_neurons3.py`` might be missing ``roi_map`` entries or the
underlying matrices that the automatic registration requires.

Run it from the repository root, e.g.::

    python debug_roi_metadata.py path/to/session1.h5 path/to/session2.mat

The script will print per-file diagnostics including spatial/temporal matrix
shapes, decoded image dimensions, the number of contours that would be
generated, and whether those counts align with the ROI indices stored in the
file.  The goal is to confirm that the input data contains the information that
``cross_day_match_neurons3.py`` expects to store inside each Napari ``Shapes``
layer's ``metadata`` dictionary.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.ndimage import binary_closing, binary_dilation
from skimage import measure


@dataclass
class SessionSummary:
    path: str
    dims: Tuple[int, int] | None
    a_shape: Tuple[int, int] | None
    c_shape: Tuple[int, int] | None
    contour_counts: Sequence[int]

    def describe(self) -> List[str]:
        lines = [f"File: {self.path}"]
        if self.dims is None:
            lines.append("  dims: MISSING")
        else:
            lines.append(f"  dims: {self.dims[0]} x {self.dims[1]}")

        if self.a_shape is None:
            lines.append("  A matrix: MISSING")
        else:
            lines.append(f"  A matrix shape: {self.a_shape}")

        if self.c_shape is None:
            lines.append("  C matrix: MISSING")
        else:
            lines.append(f"  C matrix shape: {self.c_shape}")

        total_contours = sum(self.contour_counts)
        roi_count = len(self.contour_counts)
        lines.append(f"  ROI count: {roi_count}")
        lines.append(f"  Total contours that would be generated: {total_contours}")
        if roi_count:
            min_c = min(self.contour_counts)
            max_c = max(self.contour_counts)
            lines.append(
                "  Contours per ROI (min/avg/max): "
                f"{min_c} / {total_contours / roi_count:.2f} / {max_c}"
            )
        else:
            lines.append("  Contours per ROI: N/A")

        if self.dims and self.a_shape:
            expected_pixels = self.dims[0] * self.dims[1]
            if expected_pixels != self.a_shape[0]:
                lines.append(
                    "  ⚠️ Pixel count mismatch: dims product is "
                    f"{expected_pixels} but A has {self.a_shape[0]} rows"
                )

        if self.c_shape and self.a_shape:
            if self.c_shape[0] != self.a_shape[1]:
                lines.append(
                    "  ⚠️ ROI count mismatch: C has "
                    f"{self.c_shape[0]} rows but A has {self.a_shape[1]} columns"
                )

        if any(count == 0 for count in self.contour_counts):
            zero_ids = [idx for idx, cnt in enumerate(self.contour_counts) if cnt == 0]
            preview = ", ".join(map(str, zero_ids[:10]))
            extra = "" if len(zero_ids) <= 10 else " …"
            lines.append(
                "  ⚠️ Some ROIs would not produce any contours (indices: "
                f"{preview}{extra})"
            )

        return lines


def _mask_to_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours = measure.find_contours(mask.astype(float), 0.3)
    return [c for c in contours if c.shape[0] > 4]


def _build_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    mask_norm = mask / (mask.max() if mask.max() > 0 else 1)
    binary_mask = mask_norm > 0.5
    binary_mask = binary_closing(binary_mask, iterations=2)
    binary_mask = binary_dilation(binary_mask, iterations=1)
    return binary_mask


def _extract_from_h5(path: str) -> SessionSummary:
    dims: Tuple[int, int] | None = None
    a_matrix: np.ndarray | None = None
    c_matrix: np.ndarray | None = None

    with h5py.File(path, "r") as handle:
        if "dims" in handle:
            dims_data = handle["dims"][:]
            dims = tuple(int(v) for v in np.array(dims_data).ravel()[:2])

        if "estimates" in handle:
            group = handle["estimates"]
            if "A_dense" in group:
                a_matrix = group["A_dense"][:]
            elif "A" in group:
                a_data = group["A/data"][:]
                a_indices = group["A/indices"][:]
                a_indptr = group["A/indptr"][:]
                a_shape = group["A/shape"][:]
                if a_shape.size >= 2:
                    a_csr = csr_matrix((a_data, a_indices, a_indptr), shape=(a_shape[1], a_shape[0]))
                    a_matrix = a_csr.toarray().T

            if "C" in group:
                c_matrix = group["C"][:]

    contour_counts: List[int] = []

    if dims and a_matrix is not None and a_matrix.size:
        n_pixels, n_rois = a_matrix.shape
        try:
            height, width = dims
            if height * width != n_pixels:
                # Attempt to guess dims if inconsistent
                guessed = int(math.sqrt(n_pixels))
                if guessed * guessed == n_pixels:
                    height = width = guessed
                else:
                    height = dims[0]
                    width = n_pixels // max(height, 1)
            for idx in range(n_rois):
                mask = a_matrix[:, idx].reshape((height, width), order="F")
                binary = _build_binary_mask(mask)
                contours = _mask_to_contours(binary)
                count = len(contours)
                if count == 0 and np.count_nonzero(binary) > 3:
                    count = 1
                contour_counts.append(count)
        except Exception:
            contour_counts = []

    return SessionSummary(
        path=path,
        dims=dims,
        a_shape=a_matrix.shape if a_matrix is not None else None,
        c_shape=c_matrix.shape if c_matrix is not None else None,
        contour_counts=contour_counts,
    )


def _extract_from_mat(path: str) -> SessionSummary:
    mat = sio.loadmat(path)
    dims = None
    if "dims" in mat:
        dims_arr = np.array(mat["dims"]).ravel()
        if dims_arr.size >= 2:
            dims = (int(dims_arr[0]), int(dims_arr[1]))
    a_matrix = mat.get("A")
    c_matrix = mat.get("C")

    contour_counts: List[int] = []
    if dims and a_matrix is not None and a_matrix.size:
        height, width = dims
        n_pixels = height * width
        n_pixels_a, n_rois = a_matrix.shape
        if n_pixels_a != n_pixels:
            print(
                f"  ⚠️ MAT file {path} reports dims {dims} but A has {n_pixels_a} rows."
            )
        else:
            for idx in range(n_rois):
                mask = a_matrix[:, idx].reshape((height, width), order="F")
                binary = _build_binary_mask(mask)
                contours = _mask_to_contours(binary)
                count = len(contours)
                if count == 0 and np.count_nonzero(binary) > 3:
                    count = 1
                contour_counts.append(count)

    return SessionSummary(
        path=path,
        dims=dims,
        a_shape=a_matrix.shape if isinstance(a_matrix, np.ndarray) else None,
        c_shape=c_matrix.shape if isinstance(c_matrix, np.ndarray) else None,
        contour_counts=contour_counts,
    )


def inspect_paths(paths: Iterable[str]) -> int:
    exit_code = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            exit_code = 1
            continue
        suffix = os.path.splitext(path)[1].lower()
        if suffix in {".h5", ".hdf5"}:
            summary = _extract_from_h5(path)
        elif suffix == ".mat":
            summary = _extract_from_mat(path)
        else:
            print(f"Skipping unsupported file type: {path}")
            continue

        for line in summary.describe():
            print(line)
        print("-")

    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to CNMF-E session files (.h5/.hdf5/.mat) to inspect.",
    )
    args = parser.parse_args(argv)
    return inspect_paths(args.paths)


if __name__ == "__main__":
    sys.exit(main())


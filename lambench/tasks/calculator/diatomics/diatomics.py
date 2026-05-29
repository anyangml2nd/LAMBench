"""
The reference data (diatomics.json) is derived from the MLIP Arena project:

    MLIP Arena — Benchmark machine learning interatomic potential at scale
    Yuan Chiang, Lawrence Berkeley National Laboratory
    https://github.com/atomind-ai/mlip-arena

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License.  You may obtain a copy of
the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the License for the
specific language governing permissions and limitations under the License.

----

Homonuclear diatomics dissociation curve roughness task (Applicability).

This task evaluates whether a model produces physically smooth and topologically
correct potential energy surfaces along simple bond-stretching coordinates.

Leaderboard metric (Applicability-Roughness ↓):
    avg_roughness: geometric mean of per-molecule RMSE( d²(E_model-E_DFT)/dr² )
                   in eV/Å².  Penalises high-frequency oscillations introduced by
                   the model on top of the DFT reference curve.

Stored diagnostic metrics (not scored, available for analysis):
    avg_min_position_error: mean absolute deviation of the predicted equilibrium
                            bond length from the DFT reference (Å).  If the model
                            fails to produce exactly one minimum, the molecule's
                            scan range is used as a data-driven penalty.
    avg_rmse:               mean energy RMSE over molecules (eV).

Reference data: lambench/tasks/calculator/diatomics/diatomics.json
    List of dicts with keys:
        name   – molecule label, e.g. "HH", "NN", "AlAl"
        method – DFT functional ("PBE")
        R      – bond lengths (Å), equally spaced
        E      – DFT energies (eV)
        F      – DFT forces along bond axis (eV/Å)
        S^2    – ⟨S²⟩ spin-contamination values
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from scipy.signal import find_peaks

from lambench.models.ase_models import ASEModel

_LABEL_FILE = Path(__file__).parent / "diatomics.json"
_MIN_PROMINENCE = 0.01  # eV — genuine well depth threshold for find_peaks


def _element_from_name(mol_name: str) -> str:
    """Extract element symbol: 'AlAl' → 'Al', 'HH' → 'H'."""
    return mol_name[: len(mol_name) // 2]


def _minima_positions(energies: np.ndarray, bond_lengths: np.ndarray) -> list[float]:
    """
    Return bond lengths (Å) at genuine local minima (prominence ≥ 10 meV gate).
    Works on the valid (non-NaN) subset and maps indices back to bond_lengths.
    """
    valid_mask = ~np.isnan(energies)
    if valid_mask.sum() < 3:
        return []
    idx_valid, _ = find_peaks(-energies[valid_mask], prominence=_MIN_PROMINENCE)
    return bond_lengths[valid_mask][idx_valid].tolist()


def _min_position_error(
    pos_dft: list[float], pos_model: list[float], r_range: float
) -> float:
    """
    Position error of the single equilibrium minimum (Å).

    All reference molecules have exactly one DFT minimum, so pos_dft always
    contains one element.  If the model also finds exactly one minimum, the
    error is the absolute position difference.  If the model finds zero or
    more than one minimum, the scan range is returned as a data-driven penalty
    (no free parameter).
    """
    if len(pos_model) != 1:
        return r_range
    return abs(pos_model[0] - pos_dft[0])


def _compute_roughness(residuals: np.ndarray, dr: float) -> float | None:
    """
    RMSE of the normalised second-order finite differences of energy residuals.

        δ²_i = (Δres_{i+1} - Δres_i) / Δr²  ≈  d²(E_model - E_DFT)/dr²

    Δr² normalisation makes the metric comparable across molecules with
    different grid spacings.  Returns None when fewer than 3 valid points remain.
    """
    delta2 = np.diff(residuals, n=2)
    valid = delta2[~np.isnan(delta2)]
    if len(valid) == 0:
        return None
    return float(np.sqrt(np.mean((valid / dr**2) ** 2)))


def _predict_energies(
    model: ASEModel, element: str, bond_lengths: np.ndarray
) -> np.ndarray:
    """Evaluate model energy for a homonuclear dimer at each bond length (eV)."""
    calc = model.calc
    cell = 30.0
    energies = []
    for r in bond_lengths:
        atoms = Atoms(
            symbols=[element, element],
            positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
            cell=[cell, cell, cell],
            pbc=True,
        )
        atoms.calc = calc
        try:
            e = atoms.get_potential_energy()
            if not np.isfinite(e):
                raise ValueError("non-finite energy")
        except Exception as exc:
            logging.warning(f"{element}2 @ r={r:.3f} Å failed: {exc}")
            e = np.nan
        energies.append(e)
    return np.array(energies)


def run_inference(model: ASEModel, test_data: Path | None = None) -> dict[str, dict]:
    """
    Evaluate model PES roughness on homonuclear diatomic dissociation curves.

    Args:
        model:     loaded ASEModel
        test_data: directory containing diatomics.json, or None to use the
                   bundled reference file next to this module.

    Returns:
        Per-molecule dict, e.g.::

            {
                "HH": {"roughness": 0.012, "min_position_error": 0.02,
                       "n_minima_model": 1, "rmse": 0.05},
                "NN": {...},
                ...
            }
    """
    label_path = _LABEL_FILE if test_data is None else test_data / "diatomics.json"

    with open(label_path) as fh:
        reference_data: list[dict] = json.load(fh)

    results: dict[str, dict] = {}

    for entry in reference_data:
        mol_name: str = entry["name"]
        bond_lengths = np.array(entry["R"])
        dft_energies = np.array(entry["E"])

        element = _element_from_name(mol_name)
        dr = float(np.mean(np.diff(bond_lengths)))

        r_range = float(bond_lengths[-1] - bond_lengths[0])
        pos_dft = _minima_positions(dft_energies, bond_lengths)
        model_energies = _predict_energies(model, element, bond_lengths)
        residuals = model_energies - dft_energies
        pos_model = _minima_positions(model_energies, bond_lengths)

        mol_result = {
            "roughness": _compute_roughness(residuals, dr),
            "min_position_error": _min_position_error(pos_dft, pos_model, r_range),
            "n_minima_model": len(pos_model),
            "rmse": float(np.sqrt(np.nanmean(residuals**2))),
            "r_range": r_range,
        }
        results[mol_name] = mol_result
        logging.info(f"{mol_name}: {mol_result}")

    return results

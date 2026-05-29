import numpy as np
import pytest

from lambench.tasks.calculator.diatomics.diatomics import (
    _compute_roughness,
    _min_position_error,
    _minima_positions,
)
from lambench.metrics.utils import aggregated_diatomics_results


# ---------------------------------------------------------------------------
# _compute_roughness
# ---------------------------------------------------------------------------


def test_compute_roughness_flat_residuals():
    """Flat residuals → zero roughness."""
    residuals = np.zeros(10)
    assert _compute_roughness(residuals, dr=0.1) == pytest.approx(0.0)


def test_compute_roughness_oscillating():
    """Alternating residuals produce non-zero roughness."""
    residuals = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
    roughness = _compute_roughness(residuals, dr=0.2)
    assert roughness is not None
    assert roughness > 0


def test_compute_roughness_dr_scaling():
    """Roughness scales as 1/dr²: halving dr quadruples roughness."""
    residuals = np.array([0.0, 0.1, 0.0, 0.1, 0.0], dtype=float)
    r1 = _compute_roughness(residuals, dr=0.2)
    r2 = _compute_roughness(residuals, dr=0.1)
    assert r2 == pytest.approx(4 * r1, rel=1e-6)


def test_compute_roughness_too_few_valid_points():
    """Fewer than 3 valid (non-NaN) points → None."""
    residuals = np.array([np.nan, 0.5, np.nan])
    assert _compute_roughness(residuals, dr=0.1) is None


def test_compute_roughness_all_nan():
    assert _compute_roughness(np.array([np.nan, np.nan, np.nan]), dr=0.1) is None


# ---------------------------------------------------------------------------
# _min_position_error
# ---------------------------------------------------------------------------


def test_min_position_error_exact_match():
    """Model finds exactly one minimum at the DFT position → zero error."""
    assert _min_position_error([1.0], [1.0], r_range=2.0) == pytest.approx(0.0)


def test_min_position_error_offset():
    """Model minimum is 0.15 Å away from DFT minimum."""
    assert _min_position_error([1.0], [1.15], r_range=2.0) == pytest.approx(0.15)


def test_min_position_error_no_minima():
    """Model finds zero minima → r_range penalty."""
    assert _min_position_error([1.0], [], r_range=2.0) == pytest.approx(2.0)


def test_min_position_error_two_minima():
    """Model finds two minima → r_range penalty."""
    assert _min_position_error([1.0], [0.9, 1.4], r_range=2.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _minima_positions
# ---------------------------------------------------------------------------


def test_minima_positions_single_well():
    r = np.linspace(0.5, 5.0, 50)
    e = (r - 1.5) ** 2  # parabola with minimum at 1.5 Å
    pos = _minima_positions(e, r)
    assert len(pos) == 1
    assert pos[0] == pytest.approx(1.5, abs=0.2)


def test_minima_positions_too_few_points():
    r = np.array([1.0, 1.5])
    e = np.array([0.5, 0.0])
    assert _minima_positions(e, r) == []


def test_minima_positions_all_nan():
    r = np.array([1.0, 1.5, 2.0])
    e = np.array([np.nan, np.nan, np.nan])
    assert _minima_positions(e, r) == []


# ---------------------------------------------------------------------------
# aggregated_diatomics_results
# ---------------------------------------------------------------------------


def _mol(roughness, min_pos_err, r_range, rmse=0.05):
    return {
        "roughness": roughness,
        "min_position_error": min_pos_err,
        "r_range": r_range,
        "rmse": rmse,
    }


def test_aggregated_combined_roughness_formula():
    """combined = avg_roughness × (1 + avg(min_pos_err / r_range))."""
    results = {
        "HH": _mol(roughness=0.01, min_pos_err=0.10, r_range=2.0),
        "NN": _mol(roughness=0.02, min_pos_err=0.40, r_range=2.0),
    }
    agg = aggregated_diatomics_results(results)
    avg_r = (0.01 + 0.02) / 2
    avg_norm = (0.10 / 2.0 + 0.40 / 2.0) / 2
    assert agg["combined_roughness"] == pytest.approx(avg_r * (1 + avg_norm))
    assert agg["avg_roughness"] == pytest.approx(avg_r)


def test_aggregated_perfect_position_no_penalty():
    """pos_err = 0 → combined_roughness equals avg_roughness."""
    results = {"HH": _mol(roughness=0.01, min_pos_err=0.0, r_range=2.0)}
    agg = aggregated_diatomics_results(results)
    assert agg["combined_roughness"] == pytest.approx(agg["avg_roughness"])


def test_aggregated_worst_position_doubles_roughness():
    """pos_err = r_range → combined_roughness = 2 × avg_roughness."""
    results = {"HH": _mol(roughness=0.01, min_pos_err=2.0, r_range=2.0)}
    agg = aggregated_diatomics_results(results)
    assert agg["combined_roughness"] == pytest.approx(2 * agg["avg_roughness"])


def test_aggregated_empty_results():
    """No valid molecules → all None."""
    agg = aggregated_diatomics_results({})
    assert agg["combined_roughness"] is None
    assert agg["avg_roughness"] is None
    assert agg["avg_min_position_error"] is None
    assert agg["avg_rmse"] is None


def test_aggregated_none_molecule_skipped():
    """None entries are silently skipped."""
    results = {"HH": None, "NN": _mol(roughness=0.02, min_pos_err=0.20, r_range=2.0)}
    agg = aggregated_diatomics_results(results)
    assert agg["combined_roughness"] is not None
    assert agg["avg_roughness"] == pytest.approx(0.02)

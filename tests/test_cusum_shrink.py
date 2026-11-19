"""The slice-8 shrink option on the CUSUM gate."""

import numpy as np
import pytest

from execution_cost.decay_gate import CusumGateParams, cusum_gate_states


def _shadow(seed=0, n_ref=1260, n_oos=2520, ref_mu=3e-4, oos_mu=2e-4, sd=4e-3):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(ref_mu, sd, n_ref), rng.normal(oos_mu, sd, n_oos)]), n_ref


def test_shrink_one_is_the_slice7_gate():
    sh, s0 = _shadow()
    assert np.array_equal(cusum_gate_states(sh, s0, CusumGateParams()),
                          cusum_gate_states(sh, s0, CusumGateParams(shrink=1.0)))
    assert CusumGateParams().label() == "CARL2520" and CusumGateParams(shrink=0.75).label() == "CARL2520s0.75"


def test_shrinking_an_optimistic_reference_cuts_false_alarms():
    # The OOS level is two thirds of the reference, as with an in-sample reference.
    full = np.mean([cusum_gate_states(*_shadow(s), CusumGateParams()).mean() for s in range(30)])
    shr = np.mean([cusum_gate_states(*_shadow(s), CusumGateParams(shrink=0.75)).mean() for s in range(30)])
    assert shr > full


def test_shrink_must_be_in_unit_interval():
    sh, s0 = _shadow()
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="shrink"):
            cusum_gate_states(sh, s0, CusumGateParams(shrink=bad))

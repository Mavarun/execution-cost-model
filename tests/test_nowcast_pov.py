import numpy as np
import pytest

from execution_cost.adaptive import NowcastPOV
from execution_cost.simulator import POV, MarketParams, execute, simulate_market
from execution_cost.volume_model import fit_nowcast

MP = MarketParams()


def test_participation_tracks_the_target_rate_better_than_lag1():
    nc, _ = fit_nowcast(simulate_market(MP, 500, seed=21).volumes, MP.adv)
    mk = simulate_market(MP, 2000, seed=22)
    X = 0.5 * MP.adv  # large: never completes early, so every interval trades
    rate = 0.0625
    a = execute(NowcastPOV(rate, nc), mk, MP, X)
    b = execute(POV(rate, lag=1), mk, MP, X)
    pa = a.trades[:, 1:] / mk.volumes[:, 1:]
    pb = b.trades[:, 1:] / mk.volumes[:, 1:]
    err_a = np.sqrt(np.mean(np.log(pa / rate) ** 2))
    err_b = np.sqrt(np.mean(np.log(pb / rate) ** 2))
    assert err_a < 0.8 * err_b


def test_rate_validated():
    nc, _ = fit_nowcast(simulate_market(MP, 50, seed=1).volumes, MP.adv)
    with pytest.raises(ValueError):
        NowcastPOV(1.5, nc)

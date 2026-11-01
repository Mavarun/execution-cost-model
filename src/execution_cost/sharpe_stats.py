"""Sharpe-ratio inference: PSR, expected max of N trials, deflated Sharpe, HAC.

Slice 1 reported full-sample Sharpes of a hand-picked turnover setting. That
is a multiple-testing problem: picking the best of N settings on the same
data inflates the Sharpe even when no setting has an edge. These helpers
quantify it (Bailey & Lopez de Prado 2012, 2014).

All Sharpes here are **per bar** (not annualised) unless a name says
``annual``; ``T`` is the number of bars.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def per_bar_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    if r.size < 2 or sd <= 0.0:
        return float("nan")
    return float(r.mean() / sd)


def sharpe_se(sr: float, n_obs: int, *, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of a per-bar Sharpe under non-normal i.i.d. returns.

    ``sqrt((1 - skew * SR + (kurt - 1) / 4 * SR^2) / (T - 1))`` (Mertens 2002);
    ``kurt`` is raw (normal = 3).
    """
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n_obs - 1.0)
    return float(np.sqrt(max(var, 0.0)))


def probabilistic_sharpe(
    sr: float, n_obs: int, *, benchmark: float = 0.0, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """P(true Sharpe > benchmark) given an estimate ``sr`` (all per bar)."""
    se = sharpe_se(sr, n_obs, skew=skew, kurt=kurt)
    if se <= 0.0:
        return float("nan")
    return float(stats.norm.cdf((sr - benchmark) / se))


def expected_max_sharpe(n_trials: int, trial_var: float) -> float:
    """Expected maximum of ``n_trials`` zero-mean Sharpe estimates.

    ``sqrt(V) * ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))`` with
    ``g`` the Euler-Mascheroni constant; ``0`` for a single trial.
    """
    n = int(n_trials)
    if n < 1:
        raise ValueError("n_trials must be >= 1")
    if n == 1 or trial_var <= 0.0:
        return 0.0
    g = EULER_GAMMA
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(trial_var) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe(
    returns: np.ndarray, *, n_trials: int, trial_var: float
) -> dict[str, float]:
    """Deflated Sharpe ratio of the selected strategy's returns.

    PSR evaluated at the benchmark ``E[max SR]`` of ``n_trials`` null trials
    whose Sharpe estimates have cross-trial variance ``trial_var`` (per bar).
    """
    r = np.asarray(returns, dtype=float)
    sr = per_bar_sharpe(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(n_trials, trial_var)
    return {
        "sharpe": sr,
        "sharpe_annual": sr * np.sqrt(252.0),
        "benchmark": sr0,
        "benchmark_annual": sr0 * np.sqrt(252.0),
        "psr0": probabilistic_sharpe(sr, r.size, skew=skew, kurt=kurt),
        "dsr": probabilistic_sharpe(sr, r.size, benchmark=sr0, skew=skew, kurt=kurt),
        "n_trials": int(n_trials),
        "n_obs": int(r.size),
    }


def hac_mean_test(diff: np.ndarray, *, maxlags: int = 5) -> dict[str, float]:
    """Mean of a daily return difference with a Newey-West (HAC) SE."""
    d = np.asarray(diff, dtype=float)
    res = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})
    return {
        "mean": float(res.params[0]),
        "se": float(res.bse[0]),
        "t": float(res.tvalues[0]),
        "p": float(res.pvalues[0]),
    }

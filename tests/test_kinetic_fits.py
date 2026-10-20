"""Parameter recovery on simulated sensorgrams with known constants (no lab data).

Dissociation: plot_all_fits.stage1_full (global biexponential, shared rates, per-sensor amplitudes).
Association:  assoc_two_phase.fit_global (fast rate = Ka2*C + kf0 = dimer, slow rate = Ka1*C + ks0 = tetramer).
"""
import numpy as np
import pytest

import assoc_two_phase as atp

RNG = np.random.default_rng(7)
CONCS_NM = [9.0, 27.0, 81.0, 243.0]


def test_two_phase_and_crossover():
    t = np.array([0.0, 10.0, 1e6])
    assert np.allclose(atp.two_phase(t, 0.2, 0.05, 0.1, 1e-3), [0.0, 0.2 * (1 - np.exp(-0.5)) + 0.1 * (1 - np.exp(-0.01)), 0.3])
    Af, kf, As, ks = 1.0, 1e-2, 1.0, 1e-3
    tx = atp.crossover(Af, kf, As, ks)
    assert tx == pytest.approx(np.log(10) / 9e-3)
    assert Af * kf * np.exp(-kf * tx) == pytest.approx(As * ks * np.exp(-ks * tx))   # equal slopes at t_x
    assert atp.crossover(0.1, 1e-2, 2.0, 1e-3) == 0.0                                # tetramer-dominated from t = 0


def test_dissociation_rates_recovered():
    from plot_all_fits import stage1_full
    from batch_fit import sub
    kf, ks = 2.0e-3, 5.0e-6                                   # dimer t1/2 ~ 5.8 min, tetramer t1/2 ~ 1.6 days
    t = np.arange(0.0, 15 * 3600.0, 5.0)                      # 15 h dissociation
    curves = []
    for a_f, a_s in zip([0.05, 0.10, 0.18, 0.25], [0.02, 0.05, 0.09, 0.14]):
        y = a_f * np.exp(-kf * t) + a_s * np.exp(-ks * t) + RNG.normal(0, 1e-3, t.size)
        curves.append(sub(t, y, 200, 200))
    kf_fit, ks_fit, A1, A2, R2 = stage1_full(curves)
    assert kf_fit == pytest.approx(kf, rel=0.05)
    assert ks_fit == pytest.approx(ks, rel=0.15)
    assert R2 > 0.99


def test_global_association_slopes_recovered():
    Ka2, kf0, Ka1, ks0 = 1.5e5, 2.0e-3, 5.0e3, 2.0e-4
    t = np.arange(0.0, 90 * 60.0, 1.0)                        # 90-min association
    ws = []
    for c, a_f, a_s in zip(CONCS_NM, [0.08, 0.15, 0.22, 0.26], [0.03, 0.06, 0.10, 0.15]):
        C = c * 1e-9
        y = atp.two_phase(t, a_f, Ka2 * C + kf0, a_s, Ka1 * C + ks0) + RNG.normal(0, 1e-3, t.size)
        ws.append(dict(t=t, y=y, conc_nM=c))
    fit = atp.fit_global(ws, tb=t[-1])
    assert fit["order_ok"]
    assert fit["Ka2"] == pytest.approx(Ka2, rel=0.10)
    assert fit["kf0"] == pytest.approx(kf0, rel=0.25)
    assert fit["Ka1"] == pytest.approx(Ka1, rel=0.25)
    assert fit["ks0"] == pytest.approx(ks0, rel=0.25)
    prof = atp.profile_slope(ws, t[-1], fit, "Ka2")
    assert prof["resolved"] and prof["lo"] < Ka2 < prof["hi"]

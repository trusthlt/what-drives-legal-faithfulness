import numpy as np


# signed gap = EE - CE, per concept. sign = direction (+ confabulation, judge
# over-claims; - suppression, judge under-claims), |gap| = magnitude. no
# threshold; the reader interprets direction. returns a numpy array aligned to
# the input ordering.
def signed_gap(ce_vec, ee_vec):
    return np.asarray(ee_vec, float) - np.asarray(ce_vec, float)


# Bland-Altman agreement layer. Per concept: difference d = EE - CE (the gap)
# and mean m = (CE + EE) / 2. Question-level: bias (mean d), SD, 95% limits of
# agreement, and proportional-bias slope (regress d on m; nonzero => gap grows
# with concept magnitude). Returns per-concept arrays plus the summary scalars.
def bland_altman(ce_vec, ee_vec, eps=1e-12):
    A, B = np.asarray(ce_vec, float), np.asarray(ee_vec, float)
    d = B - A
    m = (A + B) / 2.0
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    loa = (bias - 1.96 * sd, bias + 1.96 * sd)
    prop_slope = None
    if len(d) >= 3 and np.std(m) > eps:
        prop_slope = float(np.polyfit(m, d, 1)[0])
    return {"d": d, "m": m, "bias": bias, "sd": sd,
            "loa": loa, "prop_slope": prop_slope}
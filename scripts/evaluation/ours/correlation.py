import numpy as np
from scipy.stats import pearsonr

# Lin (1989) concordance correlation coefficient. Magnitude + location aware:
# penalizes both weak co-movement and systematic offset. range [-1, 1].
def ccc(x, y, eps=1e-12):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return None
    mx, my = x.mean(), y.mean()
    denom = x.var() + y.var() + (mx - my) ** 2
    if denom <= eps:
        return None
    cov = ((x - mx) * (y - my)).mean()
    return float(2 * cov / denom)


# Pearson correlation. Co-movement only; ignores scale and offset. Included as
# the Matton-style comparator that CCC is meant to improve on.
def pcc(x, y, eps=1e-12):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) <= eps or np.std(y) <= eps:
        return None
    return float(pearsonr(x, y)[0])


def compute_ee(qi, scale):
    original = qi.get("original", {})
    cfs = qi.get("counterfactuals", {})
    concepts = list(cfs.keys()) or list(original.keys())
    out = {}
    for m in concepts:
        diag = ([original[m]] if m in original else []) + [
            cf["importance"][m] for cf in cfs.get(m, []) if m in cf["importance"]
        ]
        if diag:
            out[m] = sum(diag) / len(diag) / scale
    return out
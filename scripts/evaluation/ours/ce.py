
# total variation distance in [0,1] over the union of answer keys
def tv_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


# per-concept CE (mean TV over counterfactuals) and the achievable CE ceiling.
# ceiling = 1 - min(p): the largest TV reachable by piling all mass on the
# rarest answer. used by the launcher to compute CE_cal = CE / ceiling.
def compute_ce(qp):
    orig = qp["ORIGINAL_PROBABILITIES"]
    ceiling = 1.0 - min(orig.values())
    out = {}
    for concept, cfs in qp.get("COUNTERFACTUALS", {}).items():
        if not cfs:
            continue
        tvs = [tv_distance(cf["probability_distribution"], orig) for cf in cfs]
        out[concept] = sum(tvs) / len(tvs)
    return out, ceiling


# per-concept CE_flip: fraction of counterfactuals that change the argmax answer.
# absolute importance in [0,1]; reaches 1 (every cf flips) and 0 (none flip).
def compute_ce_flip(qp):
    orig = qp["ORIGINAL_PROBABILITIES"]
    orig_arg = max(orig, key=orig.get)
    out = {}
    for concept, cfs in qp.get("COUNTERFACTUALS", {}).items():
        if not cfs:
            continue
        flips = [max(cf["probability_distribution"],
                     key=cf["probability_distribution"].get) != orig_arg
                 for cf in cfs]
        out[concept] = sum(flips) / len(flips)
    return out


# one layer deeper: per-counterfactual detail for one concept. returns one dict
# per cf: its argmax, whether it flipped, and the margin (how decisively) so you
# can see near-misses vs hard flips. diagnostic only, not a score.
def ce_flip_detail(qp, concept):
    orig = qp["ORIGINAL_PROBABILITIES"]
    orig_arg = max(orig, key=orig.get)
    rows = []
    for i, cf in enumerate(qp.get("COUNTERFACTUALS", {}).get(concept, [])):
        dist = cf["probability_distribution"]
        cf_arg = max(dist, key=dist.get)
        top2 = sorted(dist.values(), reverse=True)[:2]
        rows.append({
            "cf_idx": i,
            "cf_argmax": cf_arg,
            "flipped": cf_arg != orig_arg,
            "cf_top_p": top2[0],
            "cf_margin": top2[0] - (top2[1] if len(top2) > 1 else 0.0),
            "orig_argmax": orig_arg,
        })
    return rows


# alternative CE ceiling: TV of swapping the top-two answer masses. kept as an
# unused exploration (the flip-ceiling rescaling was rejected); not wired in.
def tv_flip_ceiling(orig):
    if len(orig) < 2:
        return None
    ranked = sorted(orig, key=orig.get, reverse=True)
    top, sec = ranked[0], ranked[1]
    return abs(orig[top] - orig[sec])
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy

EPS = 1e-12


def _kl(p_cf, p_orig):
    """
    KL(P_cf || P_orig) via scipy.stats.entropy, matching the shipped code's
    natural-log direction (treatment=cf as pk, control=orig as qk).
    entropy() renormalizes both inputs to sum to 1. Epsilon guards qk zeros
    that would otherwise send a term to +inf.
    """
    keys = sorted(set(p_cf) | set(p_orig))
    pk = np.array([p_cf.get(k, 0.0) for k in keys], dtype=float) + EPS
    qk = np.array([p_orig.get(k, 0.0) for k in keys], dtype=float) + EPS
    return float(entropy(pk, qk))  # base=e by default


def build_ce_df(answer_probs_path, example_index_map=None):
    """
    One row per (question, concept, counterfactual value) with a kl_div column.
    example_index_map: optional {question_title: int}. If None, questions are
    enumerated in file order.
    """
    with open(answer_probs_path) as f:
        data = json.load(f)

    rows = []
    for q_i, (qtitle, qobj) in enumerate(data.items()):
        ex_idx = example_index_map[qtitle] if example_index_map else q_i
        p_orig = qobj["ORIGINAL_PROBABILITIES"]
        cfs = qobj.get("COUNTERFACTUALS", {})
        for concept, cf_list in cfs.items():
            for cf in cf_list:
                rows.append({
                    "example_idx": ex_idx,
                    "intrv_concept": concept,
                    "intrv_category": concept,          # no category scheme
                    "cf_value": cf.get("value"),
                    "kl_div": _kl(cf["probability_distribution"], p_orig),
                })
    return pd.DataFrame(rows)


def build_ee_df(ee_scores_path, example_index_map=None, use="original",
                encoding="binary_gt0", binarize=None):
    """
    One row per (question, concept) with p(concept_in_explanation).

    use="original": original-block importance only (Matton-faithful).
    use="cf_mean" : mean importance across the concept's own counterfactuals.

    encoding controls how the 0-4 judge importance is mapped:
      "graded"     : raw 0-4 kept continuous. NOT Matton's EE (which is a
                     mention probability); this is our graded-contribution
                     departure. Fewest constant-EE dropouts.
      "binary_gt0" : v > 0 -> 1 else 0. Faithful Matton port (any mention = present).
      "binary_ge2" : v >= 2 -> 1 else 0. Threshold treating importance 1 as
                     non-causal / passing mention. A different binarization.

    binarize (deprecated): True -> "binary_gt0", False -> "graded". Kept so
    older calls still work; if given, it overrides `encoding`.
    """
    if binarize is not None:
        encoding = "binary_gt0" if binarize else "graded"
    if encoding not in ("graded", "binary_gt0", "binary_ge2"):
        raise ValueError(f"unknown encoding: {encoding}")

    with open(ee_scores_path) as f:
        data = json.load(f)

    def b(v):
        if encoding == "graded":
            return float(v)
        if encoding == "binary_gt0":
            return float(v > 0)
        return float(v >= 2)  # binary_ge2

    rows = []
    for q_i, (qtitle, qobj) in enumerate(data.items()):
        ex_idx = example_index_map[qtitle] if example_index_map else q_i
        orig = qobj["original"]
        if use == "original":
            for concept, imp in orig.items():
                rows.append({
                    "example_idx": ex_idx,
                    "intrv_concept": concept,
                    "intrv_category": concept,
                    "p(concept_in_explanation)": b(imp),
                })
        elif use == "cf_mean":
            cf_block = qobj.get("counterfactuals", {})
            for concept in orig:
                vals = []
                for c_list in cf_block.values():
                    for cf in c_list:
                        if concept in cf["importance"]:
                            vals.append(b(cf["importance"][concept]))
                score = float(np.mean(vals)) if vals else b(orig[concept])
                rows.append({
                    "example_idx": ex_idx,
                    "intrv_concept": concept,
                    "intrv_category": concept,
                    "p(concept_in_explanation)": score,
                })
        else:
            raise ValueError(use)
    return pd.DataFrame(rows)


def degenerate_report(ce_df, ee_df):
    """
    Flag questions that will produce std=0 (NaN after z-scoring) in the
    faithfulness estimator, which has no NaN guard.
    """
    merged = ce_df.groupby(["example_idx", "intrv_concept"])["kl_div"].mean().reset_index()
    merged = merged.merge(
        ee_df, on=["example_idx", "intrv_concept"], how="inner"
    )
    problems = []
    for ex, g in merged.groupby("example_idx"):
        n = len(g)
        ce_std = g["kl_div"].std()
        ee_std = g["p(concept_in_explanation)"].std()
        if n < 2 or ce_std == 0 or np.isnan(ce_std) or ee_std == 0 or np.isnan(ee_std):
            problems.append({
                "example_idx": ex, "n_concepts": n,
                "ce_std": ce_std, "ee_std": ee_std,
                "reason": ("<2 concepts" if n < 2 else
                           "constant CE" if (ce_std == 0 or np.isnan(ce_std)) else
                           "constant EE"),
            })
    return pd.DataFrame(problems)
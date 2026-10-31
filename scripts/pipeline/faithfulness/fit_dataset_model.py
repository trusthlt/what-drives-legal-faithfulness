import atexit as _atexit
import time as _time


def _register_runtime_timer():
    _script_start = _time.perf_counter()

    def _print_runtime():
        elapsed = _time.perf_counter() - _script_start
        print(f"\nElapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    _atexit.register(_print_runtime)


if __name__ == "__main__":
    _register_runtime_timer()

#!/usr/bin/env python3
"""
Fit a dataset-level model testing whether larger concept-effect (CE) values
predict binary explanation-effect / gold-mention labels (EE), while accounting
for clustering by question.

Primary model:
    Mixed-effects logistic regression with a random intercept for question
    via statsmodels BinomialBayesMixedGLM.

Fallback model:
    Logistic regression with question-clustered standard errors.

Input JSON format expected:
{
  "dataset_faithfulness_mean": ...,
  "num_questions_used": ...,
  "question_faithfulness": {
    "<question title>": {
      "faithfulness": ...,
      "num_concepts": ...,
      "concepts_used": [...],
      "CE_values": [...],
      "EE_values": [...]
    },
    ...
  }
}
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# statsmodels imports
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial

try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    HAS_BAYES_MIXED = True
except Exception:
    HAS_BAYES_MIXED = False


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_faithfulness_json(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert the nested JSON into one row per (question, concept).
    """
    qmap = payload.get("question_faithfulness", {})
    rows: List[Dict[str, Any]] = []

    for question_title, qobj in qmap.items():
        if not isinstance(qobj, dict):
            continue

        concepts = qobj.get("concepts_used", []) or []
        ce_vals = qobj.get("CE_values", []) or []
        ee_vals = qobj.get("EE_values", []) or []

        # skip malformed entries safely
        n = min(len(concepts), len(ce_vals), len(ee_vals))
        if n == 0:
            continue

        for i in range(n):
            concept = concepts[i]
            ce = ce_vals[i]
            ee = ee_vals[i]

            # keep only rows with numeric CE and binary-like EE
            try:
                ce = float(ce)
                ee = float(ee)
            except Exception:
                continue

            if math.isnan(ce) or math.isnan(ee):
                continue

            if ee not in (0.0, 1.0):
                # For this model we expect binary EE
                continue

            rows.append(
                {
                    "question": question_title,
                    "concept": str(concept),
                    "CE": ce,
                    "EE": int(ee),
                }
            )

    if not rows:
        raise ValueError("No usable concept rows found in the JSON.")

    df = pd.DataFrame(rows)

    # standardized CE for comparability
    ce_mean = df["CE"].mean()
    ce_std = df["CE"].std(ddof=0)
    if ce_std == 0:
        raise ValueError("All CE values are identical; cannot standardize CE.")

    df["CE_z"] = (df["CE"] - ce_mean) / ce_std

    # log transform often helps if CE is heavily right-skewed
    df["CE_log1p"] = np.log1p(df["CE"])
    ce_log_mean = df["CE_log1p"].mean()
    ce_log_std = df["CE_log1p"].std(ddof=0)
    if ce_log_std > 0:
        df["CE_log1p_z"] = (df["CE_log1p"] - ce_log_mean) / ce_log_std
    else:
        df["CE_log1p_z"] = 0.0

    return df


def fit_mixed_logit_vb(df: pd.DataFrame, predictor: str = "CE_z"):
    """
    Mixed-effects logistic regression:
        EE ~ predictor + (1 | question)
    using variational Bayes in statsmodels.
    """
    if not HAS_BAYES_MIXED:
        raise RuntimeError("BinomialBayesMixedGLM is not available in this statsmodels install.")

    # Random intercept for each question
    model = BinomialBayesMixedGLM.from_formula(
        f"EE ~ {predictor}",
        {"question_re": "0 + C(question)"},
        df,
    )
    result = model.fit_vb()
    return result


def fit_clustered_logit(df: pd.DataFrame, predictor: str = "CE_z"):
    """
    Fallback pooled logistic regression with question-clustered robust SEs.
    """
    X = sm.add_constant(df[[predictor]])
    y = df["EE"]
    model = sm.Logit(y, X)
    result = model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": df["question"]},
    )
    return result


def fit_gee_logit(df: pd.DataFrame, predictor: str = "CE_z"):
    """
    Another fallback: GEE logistic with exchangeable within-question correlation.
    """
    X = sm.add_constant(df[[predictor]])
    y = df["EE"]
    model = GEE(
        endog=y,
        exog=X,
        groups=df["question"],
        family=Binomial(),
        cov_struct=Exchangeable(),
    )
    result = model.fit()
    return result


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    n_questions = df["question"].nunique()
    n_rows = len(df)
    n_positive = int(df["EE"].sum())
    n_negative = int((1 - df["EE"]).sum())

    per_q = df.groupby("question")["EE"].agg(["count", "sum"])
    n_singleton_questions = int((per_q["count"] == 1).sum())
    n_all_zero_questions = int((per_q["sum"] == 0).sum())
    n_all_one_questions = int((per_q["sum"] == per_q["count"]).sum())
    n_informative_questions = int(
        ((per_q["sum"] > 0) & (per_q["sum"] < per_q["count"])).sum()
    )

    return {
        "n_rows": n_rows,
        "n_questions": n_questions,
        "n_positive_EE": n_positive,
        "n_negative_EE": n_negative,
        "n_singleton_questions": n_singleton_questions,
        "n_all_zero_questions": n_all_zero_questions,
        "n_all_one_questions": n_all_one_questions,
        "n_informative_questions": n_informative_questions,
    }


def extract_mixed_vb_summary(result, predictor: str) -> Dict[str, Any]:
    """
    BinomialBayesMixedGLM fit_vb summary extraction.
    """
    param_names = list(result.model.exog_names)
    params = np.asarray(result.params)

    # Variational Bayes returns posterior means in result.params.
    # For fixed effects, corresponding posterior SDs are in result.fe_sd.
    fe_names = param_names
    fe_means = np.asarray(result.fe_mean)
    fe_sds = np.asarray(result.fe_sd)

    out = {"model_type": "mixed_logit_vb", "fixed_effects": {}}
    for i, name in enumerate(fe_names):
        mean = float(fe_means[i])
        sd = float(fe_sds[i])
        odds_ratio = float(np.exp(mean))
        out["fixed_effects"][name] = {
            "coef": mean,
            "posterior_sd": sd,
            "odds_ratio": odds_ratio,
        }

    if predictor in out["fixed_effects"]:
        beta = out["fixed_effects"][predictor]["coef"]
        sd = out["fixed_effects"][predictor]["posterior_sd"]
        out["interpretation"] = (
            f"A one-standard-deviation increase in {predictor} "
            f"is associated with an odds multiplier of {math.exp(beta):.4f} "
            f"for EE=1."
        )
        if sd > 0:
            z_like = beta / sd
            out["fixed_effects"][predictor]["z_like"] = float(z_like)

    return out


def extract_classic_summary(result, predictor: str, model_type: str) -> Dict[str, Any]:
    """
    Summary extraction for Logit / GEE results.
    """
    params = result.params.to_dict()
    bse = result.bse.to_dict()

    out = {"model_type": model_type, "fixed_effects": {}}
    for name, coef in params.items():
        coef = float(coef)
        se = float(bse.get(name, float("nan")))
        odds_ratio = float(np.exp(coef))
        entry = {
            "coef": coef,
            "std_err": se,
            "odds_ratio": odds_ratio,
        }

        if hasattr(result, "pvalues"):
            try:
                entry["p_value"] = float(result.pvalues[name])
            except Exception:
                pass

        out["fixed_effects"][name] = entry

    if predictor in out["fixed_effects"]:
        beta = out["fixed_effects"][predictor]["coef"]
        out["interpretation"] = (
            f"A one-standard-deviation increase in {predictor} "
            f"is associated with an odds multiplier of {math.exp(beta):.4f} "
            f"for EE=1."
        )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to faithfulness JSON.")
    parser.add_argument(
        "--output-json",
        default="mixed_effects_ee_on_ce_results.json",
        help="Where to write model summary JSON.",
    )
    parser.add_argument(
        "--output-csv",
        default="flattened_ce_ee_rows.csv",
        help="Where to write flattened concept-level rows.",
    )
    parser.add_argument(
        "--predictor",
        choices=["CE_z", "CE_log1p_z"],
        default="CE_z",
        help="Which CE transform to use as predictor.",
    )
    args = parser.parse_args()

    payload = load_json(args.input)
    df = flatten_faithfulness_json(payload)

    # save flattened rows
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    dataset_summary = summarize_dataset(df)

    print("\n=== Dataset summary ===")
    for k, v in dataset_summary.items():
        print(f"{k}: {v}")

    results_blob: Dict[str, Any] = {
        "input_file": str(Path(args.input).resolve()),
        "predictor": args.predictor,
        "dataset_summary": dataset_summary,
        "models": {},
    }

    # 1) Preferred: mixed-effects logistic regression
    mixed_ok = False
    if HAS_BAYES_MIXED:
        try:
            mixed_result = fit_mixed_logit_vb(df, predictor=args.predictor)
            mixed_summary = extract_mixed_vb_summary(mixed_result, predictor=args.predictor)
            results_blob["models"]["mixed_logit_vb"] = mixed_summary
            mixed_ok = True

            print("\n=== Mixed-effects logistic regression (VB) ===")
            print(mixed_result.summary())
        except Exception as e:
            results_blob["models"]["mixed_logit_vb_error"] = str(e)
            print("\nMixed-effects logistic regression failed:")
            print(str(e))
    else:
        results_blob["models"]["mixed_logit_vb_error"] = (
            "BinomialBayesMixedGLM unavailable in this statsmodels installation."
        )

    # 2) Fallback / comparison: clustered logit
    try:
        clustered_result = fit_clustered_logit(df, predictor=args.predictor)
        clustered_summary = extract_classic_summary(
            clustered_result,
            predictor=args.predictor,
            model_type="clustered_logit",
        )
        results_blob["models"]["clustered_logit"] = clustered_summary

        print("\n=== Clustered logistic regression ===")
        print(clustered_result.summary())
    except Exception as e:
        results_blob["models"]["clustered_logit_error"] = str(e)
        print("\nClustered logistic regression failed:")
        print(str(e))

    # 3) Fallback / comparison: GEE
    try:
        gee_result = fit_gee_logit(df, predictor=args.predictor)
        gee_summary = extract_classic_summary(
            gee_result,
            predictor=args.predictor,
            model_type="gee_logit",
        )
        results_blob["models"]["gee_logit"] = gee_summary

        print("\n=== GEE logistic regression ===")
        print(gee_result.summary())
    except Exception as e:
        results_blob["models"]["gee_logit_error"] = str(e)
        print("\nGEE logistic regression failed:")
        print(str(e))

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results_blob, f, indent=2, ensure_ascii=False)

    print("\nWrote flattened rows to:", args.output_csv)
    print("Wrote model summaries to:", args.output_json)


if __name__ == "__main__":
    main()
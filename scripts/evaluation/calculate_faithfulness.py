import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

import ours.ce as ce_mod
import ours.ee as ee_mod
import ours.correlation as corr
import ours.diagnostic as diag

# latex table utils
from utils.latex_output import emit_latex

# framework constants: owned here, passed down into the module functions.
EPS = 1e-12
MIN_CONCEPTS = 3
RATING_SCALE = 4.0          # judge scale (0-4); constant, not expected to change
MATTON_EE_MODE = "original"  # matton always uses the original-explanation EE


# builds the shared-concept CE (raw/cal/flip) and EE vectors for one question,
# then computes whichever correlation/diagnostic metrics the config enables.
def score_question(qp, qi, metrics):
    ce, ceiling = ce_mod.compute_ce(qp)
    ce_flip = ce_mod.compute_ce_flip(qp)
    ee = ee_mod.compute_ee(qi, RATING_SCALE)
    shared = sorted(set(ce) & set(ee))
    ce_raw = [ce[m] for m in shared]
    ce_cal = [ce[m] / ceiling if ceiling > EPS else 0.0 for m in shared]
    ce_flp = [ce_flip.get(m, 0.0) for m in shared]
    eev = [ee[m] for m in shared]
    ok = len(shared) >= MIN_CONCEPTS

    # correlation scores per CE definition (only if that CE + that corr enabled)
    scores = {}
    for ce_name, ce_vec, on in (("cal", ce_cal, metrics["ce_tv"]),
                                ("raw", ce_raw, metrics["ce_tv"]),
                                ("flip", ce_flp, metrics["ce_flip"])):
        if not (ok and on):
            continue
        if metrics["correlation_ccc"]:
            scores[f"ccc_{ce_name}"] = corr.ccc(ce_vec, eev, EPS)
        if metrics["correlation_pcc"]:
            scores[f"pcc_{ce_name}"] = corr.pcc(ce_vec, eev, EPS)

    # concept-wise diagnostics: signed gap under each enabled CE definition
    concept_rows = []
    for m, r, c, f, e in zip(shared, ce_raw, ce_cal, ce_flp, eev):
        row = {"concept": m, "ce_raw": r, "ce_cal": c, "ce_flip": f, "ee": e}
        if metrics["diagnostic_gap"]:
            if metrics["ce_tv"]:
                row["gap_cal"] = e - c
            if metrics["ce_flip"]:
                row["gap_flip"] = e - f
        concept_rows.append(row)

    return {"n_concepts": len(shared), "concepts": shared,
            "ce_raw": ce_raw, "ce_cal": ce_cal, "ce_flip": ce_flp, "ee": eev,
            "scores": scores, "concept_rows": concept_rows}


def normalize_answer(a):
    if a is None:
        return None
    for ch in str(a).strip().upper():
        if ch.isalpha():
            return ch
    return None


def load_split_csv(path, title_col="Question_Title", answer_col="Correct_Answer"):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if title_col not in (reader.fieldnames or []):
            raise KeyError(f"{path}: no column {title_col!r}; found {reader.fieldnames}")
        for row in reader:
            title = (row.get(title_col) or "").strip()
            if title:
                out[title] = (row.get(answer_col) or "").strip()
    return out


def argmax_answer(qp):
    orig = qp.get("ORIGINAL_PROBABILITIES") or {}
    return normalize_answer(max(orig, key=orig.get)) if orig else None


def load_generated_answers(path):
    with open(path) as fh:
        data = json.load(fh)
    return {k: normalize_answer(((e or {}).get("original") or {}).get("answer"))
            for k, e in data.items()}


def _matton_merged(ce_df, ee_df):
    grouped = (ce_df.groupby(["example_idx", "intrv_concept", "intrv_category"])["kl_div"]
               .mean().reset_index())
    return grouped.merge(ee_df, on=["example_idx", "intrv_concept", "intrv_category"],
                         how="inner")


def _matton_keep(merged):
    keep = []
    for ex, g in merged.groupby("example_idx"):
        ce_std, ee_std = g["kl_div"].std(), g["p(concept_in_explanation)"].std()
        if len(g) < MIN_CONCEPTS or ce_std in (0,) or np.isnan(ce_std) \
                or ee_std in (0,) or np.isnan(ee_std):
            continue
        keep.append(ex)
    return keep


def _matton_per_question(samples, est):
    from scipy.stats import pearsonr
    rows = []
    for e_idx, X, Y, _ in est.regression_data_list:
        x, y = np.asarray(X).flatten(), np.asarray(Y).flatten()
        pc = float(pearsonr(x, y)[0]) if (len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0) else np.nan
        b = np.asarray(samples.get(f"beta_{e_idx}", [])).flatten()
        beta = float(np.mean(b)) if len(b) else np.nan
        rows.append({"example_idx": e_idx, "beta_mean": beta, "pcc": pc, "n_concepts": len(x)})
    return rows


# returns (mu_beta, hpdi, {title: {beta, pcc, n}}) or None
def run_matton(probs_path, ee_path, titles, seed, ee_encoding, included_titles=None):
    from matton_original.matton_adapter import build_ce_df, build_ee_df
    from matton_original.estimate_faithfulness import FaithfulnessEstimator

    idx_map = {t: i for i, t in enumerate(titles)}
    title_by_idx = {i: t for t, i in idx_map.items()}
    ce_df = build_ce_df(probs_path, idx_map)
    ee_df = build_ee_df(ee_path, idx_map, use=MATTON_EE_MODE, encoding=ee_encoding)
    if included_titles is not None:
        included_idxs = {idx_map[title] for title in included_titles if title in idx_map}
        ce_df = ce_df[ce_df.example_idx.isin(included_idxs)]
        ee_df = ee_df[ee_df.example_idx.isin(included_idxs)]
    merged = _matton_merged(ce_df, ee_df)
    keep = _matton_keep(merged)
    if len(keep) < 2:
        print("matton: <2 usable questions, skipping")
        return None
    pairs = set(zip(merged.example_idx, merged.intrv_concept))
    ce_keep = ce_df[[(e, c) in pairs for e, c in zip(ce_df.example_idx, ce_df.intrv_concept)]]
    ee_keep = ee_df[[(e, c) in pairs for e, c in zip(ee_df.example_idx, ee_df.intrv_concept)]]
    ce_keep = ce_keep[ce_keep.example_idx.isin(keep)]
    ee_keep = ee_keep[ee_keep.example_idx.isin(keep)]
    est = FaithfulnessEstimator(ee_keep, ce_keep, multi_intrv_per_concept=True)
    samples, mu_beta, hpdi = est.estimate_faithfulness(seed=seed)
    table = {title_by_idx[r["example_idx"]]: r for r in _matton_per_question(samples, est)}
    return float(mu_beta), (float(np.ravel(hpdi)[0]), float(np.ravel(hpdi)[1])), table


# per-concept sign agreement between gap_cal and gap_flip across a subset, plus
# each definition's confabulation rate and mean CCC. only meaningful when both
# ce_tv and ce_flip are enabled.
def ce_definition_agreement(subset):
    n = sign_disagree = cal_pos = flip_pos = 0
    abs_gap_diff, ccc_cal, ccc_flip = [], [], []
    for s in subset:
        for cr in s["concept_rows"]:
            if "gap_cal" not in cr or "gap_flip" not in cr:
                continue
            gc, gf = cr["gap_cal"], cr["gap_flip"]
            n += 1
            sign_disagree += (gc > 0) != (gf > 0)
            cal_pos += gc > 0
            flip_pos += gf > 0
            abs_gap_diff.append(abs(gc - gf))
        sc = s["scores"]
        if sc.get("ccc_cal") is not None:
            ccc_cal.append(sc["ccc_cal"])
        if sc.get("ccc_flip") is not None:
            ccc_flip.append(sc["ccc_flip"])
    if not n:
        return None
    return {"n_concepts": n, "sign_disagree": sign_disagree,
            "cal_pos": cal_pos, "flip_pos": flip_pos,
            "mean_abs_gap_diff": float(np.mean(abs_gap_diff)) if abs_gap_diff else 0.0,
            "ccc_cal": float(np.mean(ccc_cal)) if ccc_cal else None,
            "ccc_flip": float(np.mean(ccc_flip)) if ccc_flip else None}



# one row per (question, concept). carries per-concept CE/EE/gaps, the
# question-level scores (repeated per row), matton beta/pcc, and accuracy flags.
def build_concept_frame(summaries):
    rows = []
    for s in summaries:
        base_q = {"question": s["question"], "split": s.get("split"),
                  "n_concepts": s["n_concepts"],
                  **{k: s["scores"].get(k) for k in
                     ("ccc_cal", "ccc_raw", "ccc_flip", "pcc_cal", "pcc_flip")},
                  "matton_beta": (s.get("matton") or {}).get("beta_mean"),
                  "matton_pcc": (s.get("matton") or {}).get("pcc"),
                  "correct": s.get("correct"), "correct_probs": s.get("correct_probs")}
        for cr in s["concept_rows"]:
            rows.append({**base_q, "concept": cr["concept"],
                         "ce_raw": cr["ce_raw"], "ce_cal": cr["ce_cal"],
                         "ce_flip": cr["ce_flip"], "ee": cr["ee"],
                         "gap_cal": cr.get("gap_cal"), "gap_flip": cr.get("gap_flip")})
    schema = {"question": pl.Utf8, "split": pl.Utf8, "n_concepts": pl.Int64,
              "ccc_cal": pl.Float64, "ccc_raw": pl.Float64, "ccc_flip": pl.Float64,
              "pcc_cal": pl.Float64, "pcc_flip": pl.Float64,
              "matton_beta": pl.Float64, "matton_pcc": pl.Float64,
              "correct": pl.Boolean, "correct_probs": pl.Boolean,
              "concept": pl.Utf8, "ce_raw": pl.Float64, "ce_cal": pl.Float64,
              "ce_flip": pl.Float64, "ee": pl.Float64,
              "gap_cal": pl.Float64, "gap_flip": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


# one row per split: question-level scores averaged over questions (deduped),
# gaps over concepts, plus accuracy and the pooled matton mu_beta.
def build_summary_frame(frame, mu_beta=None):
    if frame.is_empty():
        return pl.DataFrame()
    out = []
    for split in frame["split"].unique(maintain_order=True).to_list():
        f = frame.filter(pl.col("split") == split)
        q = f.unique(subset=["question"])
        out.append({
            "split": split, "n_questions": q.height, "n_concepts": f.height,
            "acc_expl": q["correct"].mean(), "acc_probs": q["correct_probs"].mean(),
            "CCC_cal": q["ccc_cal"].mean(), "CCC_raw": q["ccc_raw"].mean(),
            "CCC_flip": q["ccc_flip"].mean(),
            "PCC_cal": q["pcc_cal"].mean(), "PCC_flip": q["pcc_flip"].mean(),
            "matton_beta_q": q["matton_beta"].mean(),
            "mean_gap_cal": f["gap_cal"].mean(), "mean_gap_flip": f["gap_flip"].mean(),
            "mu_beta": mu_beta,
        })
    return pl.DataFrame(out)



def inspect_question(key, probs, imps, metrics, summaries=None):
    qp, qi = probs[key], imps[key]
    orig = qp["ORIGINAL_PROBABILITIES"]
    ceiling = 1.0 - min(orig.values())

    print(f"\n{'='*72}\nQUESTION: {key}\n{'='*72}")
    print("ORIGINAL_PROBABILITIES:")
    for k in sorted(orig, key=orig.get, reverse=True):
        print(f"  {k}: {orig[k]:.4f}")
    print(f"  argmax={max(orig, key=orig.get)}  ceiling(1-min p)={ceiling:.4f}")

    ce, _ = ce_mod.compute_ce(qp)
    ce_flip = ce_mod.compute_ce_flip(qp)
    ee = ee_mod.compute_ee(qi, RATING_SCALE)
    shared = sorted(set(ce) & set(ee))
    print(f"\nshared concepts ({len(shared)}): {shared}")

    print(f"\n{'concept':<24}{'CE_raw':>8}{'CE_cal':>8}{'CE_flip':>9}{'EE':>7}"
          f"{'gap_cal':>9}{'gap_flip':>10}")
    ce_cal_v, ce_flip_v, ee_v = [], [], []
    for m in shared:
        cr = ce[m]
        c_cal = cr / ceiling if ceiling > EPS else 0.0
        c_flip = ce_flip.get(m, 0.0)
        ce_cal_v.append(c_cal); ce_flip_v.append(c_flip); ee_v.append(ee[m])
        print(f"  {m:<22}{cr:>8.3f}{c_cal:>8.3f}{c_flip:>9.3f}{ee[m]:>7.3f}"
              f"{ee[m]-c_cal:>+9.3f}{ee[m]-c_flip:>+10.3f}")

    if len(shared) >= MIN_CONCEPTS:
        ce_raw_v = [ce[m] for m in shared]

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  n/a "
        print(f"\n{'':<14}{'CCC':>10}{'PCC':>10}")
        for name, vec in (("CE_cal", ce_cal_v), ("CE_raw", ce_raw_v),
                          ("CE_flip", ce_flip_v)):
            print(f"  {name:<12}{fmt(corr.ccc(vec, ee_v, EPS)):>10}"
                  f"{fmt(corr.pcc(vec, ee_v, EPS)):>10}")

        if metrics["diagnostic_bland_altman"]:
            ba = diag.bland_altman(ce_cal_v, ee_v, EPS)
            ps = f"{ba['prop_slope']:+.4f}" if ba['prop_slope'] is not None else "n/a"
            print(f"\nBland-Altman (CE_cal): bias {ba['bias']:+.4f}  SD {ba['sd']:.4f}"
                  f"  LoA [{ba['loa'][0]:+.4f}, {ba['loa'][1]:+.4f}]  prop-slope {ps}")

        print(f"\nmean gap_cal : {np.mean([e-c for c,e in zip(ce_cal_v,ee_v)]):+.4f}"
              f"   mean gap_flip: {np.mean([e-c for c,e in zip(ce_flip_v,ee_v)]):+.4f}")
    else:
        print(f"\nCCC/PCC: undefined (<{MIN_CONCEPTS} concepts)")

    if summaries:
        s = next((s for s in summaries if s["question"] == key), None)
        m = (s or {}).get("matton")
        if m:
            print(f"\nMatton beta: {m['beta_mean']:.4f}   pcc: {m['pcc']:.4f}   "
                  f"n_concepts: {m['n_concepts']}")


def build_latex_rows(summaries):
    rows = []
    for s in summaries:
        m = s.get("matton") or {}
        s["f_ours_mean"] = s["scores"].get("pcc_cal")
        s["beta_matton"] = m.get("beta_mean")
        s["pcc_matton"] = m.get("pcc")
        for cr in s["concept_rows"]:
            rows.append({
                "question": s["question"], "concept": cr["concept"],
                "ce_norm": cr["ce_cal"], "ee": cr["ee"], "ee_ours": cr["ee"],
                "gap": cr.get("gap_cal"), "f_ours": cr.get("gap_cal"),
            })
    return rows


def select_mini_table_questions(summaries, count=5):
    """Select test questions with valid F values, few concepts, and varied PCC.

    Eligibility requires finite calibrated PCC (our F) and Matton F, with no
    more than nine concepts. Concept count is optimized first: every question
    below the fifth-smallest concept-count threshold is retained. Any remaining
    slots are filled from that threshold tier using deterministic max-min PCC
    spacing.
    """
    eligible = []
    for summary in summaries:
        pcc_cal = summary.get("scores", {}).get("pcc_cal")
        f_matton = (summary.get("matton") or {}).get("beta_mean")
        if summary.get("split") != "test":
            continue
        n_concepts = summary.get("n_concepts")
        if not isinstance(n_concepts, int) or not 1 <= n_concepts <= 9:
            continue
        if any(
            value is None or not np.isfinite(value)
            for value in (pcc_cal, f_matton)
        ):
            continue
        eligible.append((summary, float(pcc_cal)))

    eligible.sort(
        key=lambda item: (item[0]["n_concepts"], item[1], item[0]["question"])
    )
    if len(eligible) < count:
        raise ValueError(
            f"--mini-table requires {count} test-set questions with at most 9 "
            f"concepts and finite PCC F and Matton F values, but only "
            f"{len(eligible)} are available"
        )

    def diverse_subset(candidates, requested):
        if requested == 1:
            return [candidates[len(candidates) // 2]]
        chosen = [candidates[0], candidates[-1]]
        remaining = candidates[1:-1]
        while len(chosen) < requested:
            best = max(
                remaining,
                key=lambda item: (
                    min(abs(item[1] - existing[1]) for existing in chosen),
                    -abs(item[1]),
                    item[0]["question"],
                ),
            )
            chosen.append(best)
            remaining.remove(best)
        return chosen

    concept_threshold = eligible[count - 1][0]["n_concepts"]
    selected = [
        item for item in eligible
        if item[0]["n_concepts"] < concept_threshold
    ]
    boundary = [
        item for item in eligible
        if item[0]["n_concepts"] == concept_threshold
    ]
    slots = count - len(selected)
    if not selected:
        selected = diverse_subset(boundary, slots)
    else:
        remaining = list(boundary)
        while len(selected) < count:
            best = max(
                remaining,
                key=lambda item: (
                    min(abs(item[1] - chosen[1]) for chosen in selected),
                    -abs(item[1]),
                    item[0]["question"],
                ),
            )
            selected.append(best)
            remaining.remove(best)

    selected.sort(key=lambda item: (item[1], item[0]["question"]))
    return [summary for summary, _ in selected]



def print_summary(summary):
    if summary.is_empty():
        print("\n(no summary rows)")
        return
    with pl.Config(tbl_cols=-1, tbl_rows=-1, float_precision=4):
        print("\n=== DATASET SUMMARY ===")
        print(summary)


def print_agreement(subset, title):
    a = ce_definition_agreement(subset)
    if not a:
        return
    n = a["n_concepts"]
    print(f"\nCE_cal vs CE_flip [{title}] (n={n} concepts):")
    print(f"  sign disagreement : {a['sign_disagree']/n:.4f} ({a['sign_disagree']}/{n})")
    print(f"  confab rate cal   : {a['cal_pos']/n:.4f}   flip: {a['flip_pos']/n:.4f}")
    print(f"  mean |gap_cal - gap_flip|: {a['mean_abs_gap_diff']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Our version + Matton faithfulness")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/evaluation.yaml"))
    ap.add_argument("--target", default="glannon_vanilla", help="which target in the config to run")
    ap.add_argument("--table", action="store_true", help="also emit latex table")
    ap.add_argument(
        "--mini-table",
        action="store_true",
        help=(
            "emit a diverse five-question test-set LaTeX table requiring "
            "nonempty F and Matton F, preferring fewer concepts, with a maximum of 9"
        ),
    )
    ap.add_argument("--matton-both", action="store_true",
                    help="run Matton under both binary_gt0 and graded EE (sensitivity)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-matton", action="store_true", help="Skip the optional Bayesian estimator.")
    ap.add_argument("--output-dir", default=None, help="Directory for generated LaTeX tables; defaults to results/recomputed/tables.")
    ap.add_argument("--ee-encoding", default="binary_gt0",
                    choices=["graded", "binary_gt0", "binary_ge2"])
    args = ap.parse_args()

    import polars as pl

    with open(args.config) as fh:
        conf = yaml.safe_load(fh)
    # Configured paths are relative to the config file, independent of cwd.
    config_dir = Path(args.config).resolve().parent
    for key in ("glannon_train_csv", "glannon_test_csv"):
        conf[key] = str(config_dir / conf[key])
    for target in conf["targets"].values():
        for key in ("probs", "importance", "explanations"):
            if target.get(key):
                target[key] = str(config_dir / target[key])
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[2] / "results/recomputed/tables"
    eval_set = conf["targets"]
    metrics = dict(conf["metrics"])
    if args.skip_matton:
        metrics["matton"] = False
    GLANNON_TRAIN_CSV = conf["glannon_train_csv"]
    GLANNON_TEST_CSV = conf["glannon_test_csv"]
    if args.target not in eval_set:
        print(f"config: unknown target {args.target!r}; have {list(eval_set)}")
        exit()
    bad = [t for t, c in eval_set.items() if not {"probs", "importance"} <= c.keys()]
    if bad:
        print(f"config: targets missing probs/importance: {bad}")
        exit()

    cfg = eval_set[args.target]
    is_glannon = cfg.get("glannon", False)

    with open(cfg["probs"]) as fh:
        probs = json.load(fh)
    with open(cfg["importance"]) as fh:
        imps = json.load(fh)

    train = load_split_csv(GLANNON_TRAIN_CSV) if is_glannon else {}
    test = load_split_csv(GLANNON_TEST_CSV) if is_glannon else {}
    overlap = set(train) & set(test)
    if overlap:
        print(f"ERROR: {len(overlap)} question(s) in both train and test.")
        exit()

    def split_of(q):
        return "train" if q in train else "test" if q in test else "unknown"

    def true_answer(q):
        return normalize_answer(train.get(q) or test.get(q))

    model_answers = {}
    if cfg.get("explanations"):
        try:
            model_answers = load_generated_answers(cfg["explanations"])
        except FileNotFoundError:
            print(f"WARNING: explanations not found: {cfg['explanations']}")

    keys = sorted(set(probs) & set(imps))
    only = set(probs) ^ set(imps)
    if only:
        print(f"WARNING: {len(only)} question(s) not in both files; skipped.")

    empty_original_keys = [
        key for key in keys
        if not (
            isinstance(probs.get(key), dict)
            and probs[key].get("ORIGINAL_PROBABILITIES")
        )
    ]
    print(
        "Ignored questions with empty ORIGINAL_PROBABILITIES: "
        f"{len(empty_original_keys)}"
    )
    if empty_original_keys:
        for key in empty_original_keys:
            print(f"  - {key}")
        empty_original_set = set(empty_original_keys)
        keys = [key for key in keys if key not in empty_original_set]

    if not keys:
        print("ERROR: no usable shared questions after filtering empty probabilities.")
        exit()

    summaries = []
    for k in keys:
        s = score_question(probs[k], imps[k], metrics)
        s["question"] = k
        s["split"] = split_of(k)
        m, t, p = model_answers.get(k), true_answer(k), argmax_answer(probs[k])
        s["correct"] = (m == t) if (m and t) else None
        s["correct_probs"] = (p == t) if (p and t) else None
        s["answers_agree"] = (p == m) if (p and m) else None
        summaries.append(s)

    # matton runs once over the whole config (pools across all kept questions)
    mu_beta_table = None
    if metrics.get("matton", True):
        encodings = ["binary_gt0", "graded"] if args.matton_both else [args.ee_encoding]
        table_enc = encodings[0]
        for enc in encodings:
            res = run_matton(
                cfg["probs"], cfg["importance"], list(probs), args.seed, enc,
                included_titles=keys,
            )
            if not res:
                continue
            mu_beta, hpdi, mtable = res
            if enc == table_enc:
                mu_beta_table = mu_beta
                for s in summaries:
                    s["matton"] = mtable.get(s["question"])
            print(f"\n{'='*60}\nMATTON ({enc})  mu_beta = {mu_beta:.4f}  "
                  f"90% HPDI [{hpdi[0]:.4f}, {hpdi[1]:.4f}]\n{'='*60}")

    frame = build_concept_frame(summaries)
    summary = build_summary_frame(frame, mu_beta=mu_beta_table)

    print(f"\nCONFIG: {args.target}  |  metrics: "
          f"{[k for k, v in metrics.items() if v]}")
    print_summary(summary)

    if metrics["ce_tv"] and metrics["ce_flip"] and metrics["diagnostic_gap"]:
        if is_glannon:
            for name in ("train", "test"):
                print_agreement([s for s in summaries if s["split"] == name], name.upper())
        print_agreement(summaries, "ALL")

    inspect_question(keys[0], probs, imps, metrics, summaries=summaries)

    if args.table:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = str(output_dir / f"table_{args.target}.tex")
        emit_latex(build_latex_rows(summaries), summaries, path=path)
        print(f"wrote {path}")

    if args.mini_table:
        try:
            mini_summaries = select_mini_table_questions(summaries, count=5)
        except ValueError as error:
            print(f"ERROR: {error}")
            sys.exit(1)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = str(output_dir / f"mini_table_{args.target}.tex")
        emit_latex(
            build_latex_rows(mini_summaries),
            mini_summaries,
            path=path,
            caption=(
                "Five test-set questions with both faithfulness measures, selected "
                "by preferring fewer concepts (maximum nine) and calibrated-Pearson "
                "diversity."
            ),
        )
        print("\nMINI TABLE QUESTIONS (ordered by pcc_cal):")
        for summary_item in mini_summaries:
            print(
                f"  {summary_item['question']}: "
                f"n_concepts={summary_item['n_concepts']}, "
                f"F_PCC_cal={summary_item['scores']['pcc_cal']:.4f}, "
                f"Matton_F={summary_item['matton']['beta_mean']:.4f}"
            )
        print(f"wrote {path}")

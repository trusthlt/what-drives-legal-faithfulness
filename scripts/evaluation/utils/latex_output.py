from collections import OrderedDict


# Every marker state occupies exactly one arrow-width, left-aligned inside the
# box, so the numeric parts of the gap column stay flush right.
_MARK = r"\makebox[\widthof{$\uparrow$}][l]{%s}"
MARK_UP = _MARK % r"$\uparrow$"
MARK_DOWN = _MARK % r"$\downarrow$"
MARK_NONE = _MARK % ""

def tex_escape(s, break_underscores=True):
    """
    Escape the characters that actually show up in concept/question keys.

    break_underscores: emit \\allowbreak after each escaped underscore. An
    escaped underscore is not a line-breaking opportunity, so a name like
    company\\_major\\_customer is one unbreakable token and overfulls a narrow
    p-column, printing over the next cell. \\allowbreak makes the split legal.
    """
    if not isinstance(s, str):
        s = str(s)
    out = []
    for ch in s:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "_":
            out.append(r"\_\allowbreak{}" if break_underscores else r"\_")
        elif ch in "&%$#{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def prettify_concept(name, underscores_to_spaces=True):
    """
    Concept keys are snake_case identifiers. Spaces give LaTeX natural break
    points and read better in a table than escaped underscores.
    """
    return name.replace("_", " ") if underscores_to_spaces else name


def num(v, prec=3):
    """Format a float, or an en-dash for missing/undefined."""
    if v is None or v == "":
        return "--"
    if isinstance(v, str):
        return tex_escape(v)
    return f"{v:.{prec}f}"


def marked_gap(gap, threshold, prec=3):
    r"""
    Signed gap plus a fixed-width direction marker.

      gap >  threshold  ->  confabulation  ->  \uparrow
      gap < -threshold  ->  suppression    ->  \downarrow
      otherwise         ->  blank box of the same width

    The number keeps its sign, so direction is legible without the marker.
    """
    if gap is None:
        return "--"
    s = f"{gap:.{prec}f}"
    if gap > threshold:
        mark = MARK_UP
    elif gap < -threshold:
        mark = MARK_DOWN
    else:
        mark = MARK_NONE
    return f"${s}$\\," + mark


def emit_latex(all_rows, summaries, path=None,
               mark_threshold=0.25, label="tab:faithfulness", prec=3,
               qcol_width="2.6cm", ccol_width="2.2cm",
               underscores_to_spaces=True,
               caption="Per-concept faithfulness."):
    blocks = OrderedDict()
    for r in all_rows:
        blocks.setdefault(r["question"], []).append(r)

    by_q = {s["question"]: s for s in summaries}

    L = []
    A = L.append

    qspec = f">{{\\raggedright\\arraybackslash}}p{{{qcol_width}}}"
    cspec = f">{{\\raggedright\\arraybackslash}}p{{{ccol_width}}}"

    A(r"\begin{table*}[t]")
    A(r"\centering")
    A(r"\footnotesize")
    A(r"\resizebox{\textwidth}{!}{%")
    A(f"\\begin{{tabular}}{{@{{}}{qspec}{cspec}rrrrrr@{{}}}}")
    A(r"\toprule")
    A(r"Question & Concept & $\mathrm{CE}_{\mathrm{norm}}$ & "
      r"$\mathrm{EE}_{\mathrm{norm}}$ & gap & "
      r"$\mathcal{F_{\mathrm{c}}}$ & $\mathcal{F}$ & "
      r"$\mathcal{F_{\mathrm{M}}}$ \\")
    A(r"\midrule")

    for qi, (qkey, rows) in enumerate(blocks.items()):
        if qi:
            A(r"\midrule")
        s = by_q.get(qkey, {})

        beta = s.get("beta_matton", s.get("pcc_matton"))
        beta_cell = num(beta, prec) if beta is not None else "--"
        mean = s.get("f_ours_mean")
        mean_cell = num(mean, prec) if mean is not None else "--"
        qtitle = tex_escape(qkey)

        for i, r in enumerate(rows):
            first = (i == 0)
            ee_val = r.get("ee", r.get("ee_ours"))
            A(" & ".join([
                qtitle if first else "",
                tex_escape(prettify_concept(r["concept"], underscores_to_spaces)),
                num(r["ce_norm"], prec),
                num(ee_val, prec),
                marked_gap(r["gap"], mark_threshold, prec),
                num(r["f_ours"], prec),
                mean_cell if first else "",
                beta_cell if first else "",
            ]) + r" \\")

    A(r"\bottomrule")
    A(r"\end{tabular}%")
    A(r"}")
    A(f"\\caption{{{caption}}}")
    A(f"\\label{{{label}}}")
    A(r"\end{table*}")

    src = "\n".join(L) + "\n"
    if path:
        with open(path, "w") as fh:
            fh.write(src)
    return src
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

import argparse
import json
import os
import random
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

DEFAULT_OUTPUT_DIR = "faithfulness_plots/"
DEFAULT_SAMPLES = -1  # -1 to show all samples
WIDTH_PER_SAMPLE = 60 / 160


# =========================================================
# MAIN
# =========================================================

def main(input_json: str, output_dir: str, n_samples: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    q_data = data.get("question_faithfulness", {})

    # Extract titles and faithfulness values
    titles = []
    values = []

    for title, block in q_data.items():
        F = block.get("faithfulness")
        if F is not None:
            titles.append(title)
            values.append(F)

    total_samples = len(values)
    if n_samples == -1 or n_samples >= total_samples:
        n_samples = total_samples
        sample_pairs = list(zip(titles, values))
    else:
        sample_pairs = random.sample(list(zip(titles, values)), n_samples)

    titles, values = zip(*sample_pairs)

    # Sort by faithfulness (optional but nice)
    # titles, values = zip(*sorted(zip(titles, values), key=lambda x: x[1]))

    # =========================================================
    # PLOT
    # =========================================================

    plt.figure(figsize=(WIDTH_PER_SAMPLE * n_samples * 3, 6))

    bars = plt.bar(
        titles,
        values,
        color=["#4CAF50" if v >= 0 else "#E53935" for v in values]
    )

    # Zero baseline
    plt.axhline(0, color="black", linewidth=1)

    # Value labels
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.02 if val >= 0 else -0.04),
            f"{val:.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=6
        )

    plt.ylabel("Faithfulness (Pearson CE–EE)")
    plt.title("Question-level Explanation Faithfulness")
    plt.xticks(rotation=30, ha="right")

    plt.ylim(-1, 1)

    plt.tight_layout()

    # Save
    out_path = os.path.join(
        output_dir,
        f"faithfulness_per_question_magistral_{n_samples}_examples.png",
    )
    plt.savefig(out_path)
    plt.close()

    print("Saved faithfulness plot →", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args()

    main(args.input, args.output_dir, args.samples)

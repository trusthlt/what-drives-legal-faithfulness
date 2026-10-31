# What Drives an LLM’s Legal Reasoning? Measuring Faithfulness at the Concept Level



This project studies explanation faithfulness in legal multiple-choice reasoning (https://aclanthology.org/2026.nllp-1.26/):
do the concepts emphasized in an explanation correspond to the concepts that
actually change a model's answer? It uses Magistral-Small-2509, compares the base
model with a LoRA fine-tuned variant on Glannon questions, and also evaluates a
multilevel legal reasoning collection.

The experiment extracts concepts, changes one concept at a time to generate
counterfactual questions, measures changes in answer probabilities, and asks a
judge to rate concept importance in explanations. The evaluation compares these
causal concept effects (CE) with explanation-implied effects (EE).

This directory contains the essential code and saved experimental results.
It can be used independently with the dependencies and inputs described below.

## Layout

```text
what_drives/
├── configs/evaluation.yaml        # Three evaluation targets and metric switches
├── scripts/
│   ├── evaluation/                # Primary evaluation and Bayesian comparator
│   ├── pipeline/
│   │   ├── concepts/              # Extract and optionally flatten concepts
│   │   ├── counterfactuals/       # Rewrite questions by intervening on concepts
│   │   ├── probabilities/         # Score answers, calibrate, rescale temperature
│   │   ├── explanations/          # Generate explanations and judge importance
│   │   ├── gold/                  # Reference-explanation comparison
│   │   ├── faithfulness/          # Older CE/EE workflow, plots, regression
│   │   ├── signal/                # Find influential counterfactual pairs
│   │   └── utils/                 # Shared clients, prompts, CSV helpers
│   └── training/                 # Split, train with/without reasoning, merge
├── results/
│   ├── glannon-vanilla/           # Base Magistral results
│   ├── glannon-sft/               # Fine-tuned Magistral results
│   ├── multilevel/                # Multilevel results, including gold comparison
│   ├── signal/                    # Leave-one-out and selected-pair JSON/CSV
│   ├── training/                  # Saved pre/post-training evaluation outputs
│   ├── metadata/                  # Question IDs and answer keys only
│   └── tables/                    # Previously generated LaTeX tables
└── requirements*.txt
```

Source datasets, dataset archives, model weights/adapters, serving metadata,
credentials, caches, and execution logs are excluded. The two small metadata CSVs
retain only `Question_Title` and `Correct_Answer` from the evaluation splits, so
saved results can be evaluated without the source corpus. They contain no
question text, answer-option text, or reference analyses. Experimental outputs
can still contain rewritten questions, explanations, and model predictions.

## Setup and saved-result evaluation

Run the examples from this directory:

```bash
cd what_drives
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# No GPU, model, source dataset, or API credentials needed for these metrics.
python scripts/evaluation/calculate_faithfulness.py --target glannon_vanilla --skip-matton --table
python scripts/evaluation/calculate_faithfulness.py --target glannon_sft --skip-matton --table
python scripts/evaluation/calculate_faithfulness.py --target multilevel --skip-matton --table

# Optional Bayesian comparator: install its dependencies, then omit --skip-matton.
python -m pip install -r requirements-bayesian.txt
python scripts/evaluation/calculate_faithfulness.py --target glannon_vanilla --matton-both --table
```

The evaluator prints summaries and a question-level inspection. `--table` writes
to `results/recomputed/tables/`; `--output-dir` selects another destination.
`--mini-table` selects five eligible test questions and requires the Bayesian
scores. Configured paths resolve relative to the configuration file, so the
evaluator also works when launched from another working directory.

The Bayesian estimator uses MCMC with 500 warmup steps and 2,000 samples. It is
optional and substantially slower than the direct metrics. Requirements are
unversioned, matching the original project; no complete environment lock was
available in the source tree.

## How the scores work

The primary evaluator in `scripts/evaluation/calculate_faithfulness.py` uses:

- **Raw CE:** average total variation distance between original and
  counterfactual answer distributions for a concept.
- **Calibrated CE:** raw CE divided by `1 - min(original probabilities)`, the
  maximum reachable total variation distance from the original distribution.
- **EE:** average the concept's importance in the `original` block and the
  counterfactuals that intervene on that same concept, then divide by 4.
- **Faithfulness:** per-question Pearson correlation between the aligned CE and
  EE vectors, requiring at least three shared concepts. Constant vectors have
  undefined correlation. Concordance correlation is another configurable option.
- **Diagnostics:** signed `EE - CE` gaps and Bland–Altman agreement statistics.
  Positive gaps indicate greater stated importance than measured causal effect;
  negative gaps indicate the reverse. These are diagnostics, not proof of cause.
- **Optional comparator:** the Matton-style hierarchical Bayesian model uses
  KL divergence and original-block importance, binarized by default. Its scores
  have a different construction from the direct correlation scores.

**Explanation provenance matters:** both importance-judging implementations read
the `original` explanation from the input CSV's reference `Analysis` field. They
judge model-generated explanations for counterfactuals. Thus the primary EE
combines reference-explanation and model-explanation importance. The generated
original answer is used separately for accuracy. The reorganization preserves
this behavior; the field name `original` does not establish that it was generated
by the model.

The older `pipeline/probabilities/compute_legacy_ce.py` computes mean
`KL(P_counterfactual || P_original)`. Its companion `explanations/compute_ee.py`
excludes original-block importance unless `--include-original` is supplied, and
keeps the 0–4 scale unless `--normalize` is supplied. Their older faithfulness
files must be interpreted with those definitions, separately from the primary
total-variation evaluator.

## Script reference

Paths below are relative to `scripts/`. Use `python scripts/<path> --help` for all
arguments. Pipeline file inputs and principal outputs are explicit arguments;
provide the prepared source CSV externally when generating new experiments.

### Primary evaluation

| Script or module | Use |
| --- | --- |
| `evaluation/calculate_faithfulness.py` | Evaluate a configured target; print metrics and optionally write full or five-question LaTeX tables. |
| `evaluation/calculate_multilevel_accuracy.py` | Compare probability argmax predictions with an external answer CSV; optionally write a detailed JSON report. Supports one-based numeric multilevel answer labels. |
| `evaluation/ours/ce.py` | Total variation CE, achievable ceiling, answer-flip rate, and counterfactual diagnostics. |
| `evaluation/ours/ee.py` | Average original and matching-counterfactual importance on the normalized 0–1 scale. |
| `evaluation/ours/correlation.py` | Pearson and Lin concordance correlations, with degenerate-vector handling. |
| `evaluation/ours/diagnostic.py` | Signed gaps and Bland–Altman statistics. |
| `evaluation/matton_original/matton_adapter.py` | Convert result JSON to KL/importance tables and apply binary or graded EE encodings. |
| `evaluation/matton_original/estimate_faithfulness.py` | Fit and plot the hierarchical Bayesian faithfulness estimator. |
| `evaluation/matton_original/faithfulness_utils.py` | Regression models, input standardization, and plotting helpers for that estimator. |
| `evaluation/utils/latex_output.py` | Render concept-level scores and question summaries as LaTeX. |

`__init__.py` files mark module packages and are not command-line scripts.

### Experiment generation

| Script | Inputs → output / purpose |
| --- | --- |
| `pipeline/concepts/extract.py` | `--csv`, `--output`: extract concept/value mappings using a local model or `--vllm` server. If `Question_Label` exists, process only `Reasoning` rows. |
| `pipeline/concepts/flatten.py` | `--input`, `--csv`, `--output`: optional legacy flatten/filter step; attach source question/intro text, apply a title blocklist and a minimum nonempty-field count. This changes the evaluated population. |
| `pipeline/counterfactuals/generate.py` | `--input`, `--csv`, `--output`: generate local question edits for each concept, using the shared model client; `--max-examples` limits the run. |
| `pipeline/counterfactuals/generate_offline.py` | Same task with batched offline vLLM; configure `--model`, batch size, and GPU settings. |
| `pipeline/probabilities/direct_null.py` | `--input`, `--csv`, `--output`: teacher-force answer candidates, subtract null-prompt scores, and apply temperature softmax, without generating rationales. This produces the primary evaluation's probability schema. |
| `pipeline/probabilities/log_likelihood_offline.py` | Alternative offline scorer that generates a rationale before scoring answer continuations; also provides shared vLLM/model-preparation helpers. |
| `pipeline/probabilities/log_likelihood_permuted_offline.py` | Alternative direct scorer with answer-order permutations and null calibration; set `--n-permutations` and `--seed`. |
| `pipeline/probabilities/inspect_scoring.py` | Diagnose original-answer scoring before/after calibration against generated answers. It is also an imported dependency of `direct_null.py`. |
| `pipeline/probabilities/apply_temperature.py` | `--input`, `--output`, `--temperature`: recompute probability distributions from stored calibrated logits, without inference. |
| `pipeline/explanations/generate.py` | `--input`, `--csv`, `--output`: generate answers and explanations for originals/counterfactuals via the shared client. |
| `pipeline/explanations/generate_offline.py` | Batched offline version; supports base, merged, and LoRA model configurations. |
| `pipeline/explanations/judge_importance.py` | `--explanations`, `--concepts`, `--csv`, `--output`: judge importance on a 0–4 scale; uses reference `Analysis` for originals. Supports the shared provider options. |
| `pipeline/explanations/judge_importance_offline.py` | Batched offline importance judge, including JSON-constrained and Qwen-oriented options; uses the same reference-analysis convention. |
| `pipeline/explanations/accuracy.py` | `--explanations`, `--csv`: evaluate generated original answers; optionally write JSON/CSV reports. |

### Supporting analyses and older workflow

| Script | Use |
| --- | --- |
| `pipeline/probabilities/compute_legacy_ce.py` | `--input`, `--output`: reduce probability distributions to per-concept mean KL effects. |
| `pipeline/explanations/compute_ee.py` | `--importance`, `--concepts`, `--output`: reduce importance to per-concept EE; optional `--include-original` and `--normalize`. |
| `pipeline/faithfulness/compute_legacy.py` | `--ce`, `--ee`, `--output`: correlate precomputed CE/EE per question and save their mean and aligned vectors. |
| `pipeline/faithfulness/plot.py` | `--input`, `--output-dir`: plot question-level scores from the older faithfulness JSON schema. |
| `pipeline/faithfulness/fit_dataset_model.py` | `--input`: flatten older faithfulness JSON and fit mixed-effects, clustered, and GEE logistic models of **binary** EE on CE. Graded EE values other than 0/1 are excluded; this is not a general graded-importance regression. |
| `pipeline/gold/judge_importance.py` | `--explanations`, `--concepts`, `--csv`, `--output`: judge reference analyses only, producing `original_importance` blocks. |
| `pipeline/gold/compute_ee.py` | `--gold-importance`, `--concepts`, `--output`: convert reference importance to EE; optional normalization. Feed the result to `compute_legacy.py` with the chosen CE file. |
| `pipeline/signal/leave_one_out.py` | `--probabilities`, `--ee`, `--output`, `--csv-output`: remove each counterfactual in turn and rank changes in the older CE/EE signals. Optional `--normalize` appends `_normalized` to output names. |
| `pipeline/signal/heaviest_pairs.py` | `--leave-one`, `--counterfactuals`, `--explanations`, `--probabilities`, `--output`, `--csv-output`: retrieve the most influential counterfactual per question/concept, with explanation and probability context. |

### Training and shared utilities

| Script or module | Use |
| --- | --- |
| `training/split_data.py` | `--csv-path`, `--out-dir`: detect chapters when question numbering restarts at 1; reserve each chapter's final two questions for testing. Reads logical CSV records, including quoted multiline fields. Supplied evaluation split metadata remains the authority for existing runs. |
| `training/train_no_think.py` | `--train-csv`, `--test-csv`: LoRA supervised training on final answers without reasoning targets; optional pre/post evaluation and JSON run logs. |
| `training/train_think.py` | Same training flow with reference reasoning and Magistral thinking tokens. |
| `training/merge_adapter.py` | `--adapter-path`, `--output-dir`: merge a PEFT adapter into its base model for standalone inference. |
| `pipeline/utils/llm_client.py` | Shared Hugging Face, OpenAI Responses, and OpenAI-compatible vLLM clients; provider and CLI configuration. |
| `pipeline/utils/prompt_loader.py` | Load and format task prompts from adjacent `prompts.yaml`. |
| `pipeline/utils/csv_fields.py` | Normalize CSV column names and retrieve fields. |

The pipeline's `utils` and evaluation's `utils` are separate packages. Launch the
scripts by their file paths as shown here; avoid placing both directories in a
global `PYTHONPATH`.

## Generate a new experiment

Install `requirements-generation.txt` in an environment with a compatible GPU,
CUDA, PyTorch, Transformers, and vLLM stack. Model weights and an input CSV must
be supplied separately. Examples use the model ID recorded in the original
scripts, not an assertion about current model availability.

A prepared generation CSV needs stable `Question_Title` values, `Question_Text`,
`Possible_Answers`, and `Included` (`true`, `1`, or `yes` for the direct scorer).
Provide `Analysis` for importance judging and `Correct_Answer` for accuracy and
training. `Intro_Text` is optional. Preserve title IDs across every stage;
multilevel titles are numeric strings in the saved outputs. The metadata CSVs
included here are answer keys, not generation/training inputs.

Start a compatible model server in a separate terminal if using `--vllm`:

```bash
vllm serve mistralai/Magistral-Small-2509 --port 8000
```

Then run the stages below. The output directory is separate from preserved
results. `QUESTION_CSV` must point to your full prepared CSV.

```bash
export QUESTION_CSV=/absolute/path/to/prepared_questions.csv
export RUN_DIR=results/recomputed/new_run
mkdir -p "$RUN_DIR"

python scripts/pipeline/concepts/extract.py --vllm \
  --vllm-model mistralai/Magistral-Small-2509 \
  --csv "$QUESTION_CSV" --output "$RUN_DIR/concepts.json"

python scripts/pipeline/counterfactuals/generate.py --vllm \
  --vllm-model mistralai/Magistral-Small-2509 \
  --input "$RUN_DIR/concepts.json" --csv "$QUESTION_CSV" \
  --output "$RUN_DIR/counterfactuals.json"

python scripts/pipeline/explanations/generate.py --vllm \
  --vllm-model mistralai/Magistral-Small-2509 \
  --input "$RUN_DIR/counterfactuals.json" --csv "$QUESTION_CSV" \
  --output "$RUN_DIR/explanations.json"

# Run the offline scorer on a free GPU; stop the server first if sharing its GPU.
python scripts/pipeline/probabilities/direct_null.py \
  --model mistralai/Magistral-Small-2509 \
  --input "$RUN_DIR/counterfactuals.json" --csv "$QUESTION_CSV" \
  --output "$RUN_DIR/probabilities.json" --temperature 3

# This stage makes paid API calls; set OPENAI_API_KEY in your environment first.
LLM_PROVIDER=openai OPENAI_MODEL=gpt-5.4-mini \
python scripts/pipeline/explanations/judge_importance.py \
  --explanations "$RUN_DIR/explanations.json" \
  --concepts "$RUN_DIR/counterfactuals.json" --csv "$QUESTION_CSV" \
  --output "$RUN_DIR/importance.json"
```

The judge model above is the identifier used in the archived runs. A local judge
can be used with `judge_importance_offline.py --model <local-model-or-model-id>`
and the same four file arguments. Shared clients select their provider with
`LLM_PROVIDER`; vLLM scripts also accept `--vllm-base-url`, `--vllm-model`, and
`--vllm-api-key`. Keep credentials in environment variables.

For evaluation, copy `configs/evaluation.yaml`, add a target with `probs`,
`importance`, `explanations`, and `glannon`, and run with `--config <your-file>`
and `--target <your-target>`. Set `glannon: true` only with the matching train/test
answer keys. Paths are relative to your new configuration file.

For a model fine-tuned with LoRA, use `training/merge_adapter.py` to obtain a
merged model and pass its path to offline scorers. Some offline scripts provide
`--prepare-merged-model` and `--base-model-for-serving` to reconstruct required
serving metadata; those operations need the actual model files.

## Fine-tune Magistral

```bash
python scripts/training/train_no_think.py \
  --train-csv /absolute/path/to/glannon_train.csv \
  --test-csv /absolute/path/to/glannon_test.csv \
  --held-out --do-pre-eval --do-train --do-post-eval \
  --lora-r 64 --lora-alpha 128 \
  --out-dir results/recomputed/no_think \
  --output-model-dir checkpoints/no_think

python scripts/training/merge_adapter.py \
  --adapter-path checkpoints/no_think \
  --output-dir models/no_think_merged
```

Use `train_think.py` for reasoning-target supervised fine-tuning. **Pass
`--held-out` for a held-out experiment:** without it, both training scripts
combine the train and test CSVs for training/evaluation. The example parameters
illustrate usage; available files do not establish the full settings of every
archived run, so they are not an exact reproduction recipe for checkpoint 84.

## Saved results and their interpretation

| Directory / files | Contents |
| --- | --- |
| `glannon-vanilla/` | 207 questions with concept/counterfactual and probability outputs; 204 questions with importance ratings. |
| `glannon-sft/` | 207 questions for the LoRA no-think experiment associated with rank 64 / checkpoint 84; 199 with importance ratings. |
| `multilevel/` | 100-question main run, alternate probabilities, extracted concepts, counterfactuals, importance, and accuracy report. |
| `multilevel/gold/` | Reference-importance, EE, older faithfulness, and dataset-regression outputs. These have their own coverage and are not all outputs of the primary evaluation config. |
| `signal/` | JSON and CSV rankings, heaviest pairs, and normalized variants from the exploratory multilevel signal analysis. |
| `training/` | Original pre/post predictions from several runs; filenames distinguish runs, without establishing matching hyperparameters. `root_eval_post.json` had no more specific run attribution. |
| `metadata/` | 161 training and 48 test question IDs with answer labels. Some source questions do not appear in inference outputs. |
| `tables/` | Three previously generated LaTeX tables. They are retained as historical outputs; regenerating a table writes to `results/recomputed/tables/`. |

Within the two Glannon directories, `03_probs_direct_null_*full_data.json`
contains direct-null scores before the saved temperature-3 postprocessing;
`*temp_3.json` is selected by the primary config. `03_probs_null_permute_*`
contains the alternate permutation-based estimates. The files named
`*test_only.json` still contain 207 top-level question records: filenames alone
are not a reliable split filter. The evaluator uses the saved split metadata.

Stages `04_ce_*`, `07_ee_*`, and `08_faithfulness_*` belong to the older
KL-based reduction workflow. Changing the probability method or EE averaging
rule produces different results. Join records by question title and concept,
and inspect skipped/degenerate cases instead of assuming equal coverage.

## Implementation and validation

Relocated scripts have descriptive filenames, updated sibling imports, and
explicit file arguments instead of historical machine paths. The primary
evaluator resolves config paths independently of the working directory, loads
Bayesian dependencies only when needed, and provides `--skip-matton` and a
separate table output directory. The splitter retains its chapter rule but now
handles multiline CSV records correctly. Mathematical scoring definitions were
preserved.

Validation covers Python syntax, available command-line entry points, saved JSON
readability, config file paths, the multiline split behavior, and
direct scoring of all three primary saved-result targets. Temperature
postprocessing reproduces both saved temperature-3 files exactly; older KL
effects match within floating-point tolerance, and both signal-analysis scripts
run successfully on the curated inputs. GPU generation,
fine-tuning, the full Polars report, dataset regression, and Bayesian MCMC require
dependencies or hardware unavailable in the restructuring environment and were
not executed as part of this reorganization.

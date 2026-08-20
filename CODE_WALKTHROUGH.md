# Code walkthrough

All code lives in `src/carb/` (CARB = Clarification–Action Routing Benchmark) and is plain
Python, run as modules with `PYTHONPATH=src`. Nothing depends on a notebook.

```
datasets/carb/carb_v1.jsonl        the benchmark (840 items)
datasets/carb/sft_train.jsonl      training set for the local model (2,800 items, disjoint)
src/carb/                          all code
results/                           every artefact produced by a run
figures/                           every figure in the report
```

## Data flow

```
source datasets (CoCoNot, CLAMBER, IN3)
   │  build_benchmark.py       fixed category → action mapping, stratified sampling
   ▼
carb_v1.jsonl ─────────────────────────────────────────────┐
   │                                                       │
   │  run_eval.py (API)      run_stakes.py (API)           │  local_model.py (GPU)
   │  5 regimes × 4 models   3 stakes frames × 2 regimes   │  zero-shot / LoRA SFT / probe
   ▼                         ▼                             ▼
results/raw/*.jsonl      results/raw_stakes/*.jsonl    results/raw_local/*.jsonl
   │                         │                             │
   │  local_judge.py (GPU): free-text responses → one of ACT/ASK/REFUSE/DEFER
   ▼                         ▼                             │
results/raw/judged*.jsonl results/raw_stakes/judged*.jsonl │
   │                         │                             │
   │  derive.py: every regime → a predicted action per item│
   ▼                         ▼                             ▼
analyze.py            analyze_stakes.py              analyze_local.py
utility.py  budget.py  recognition.py  artifact_check.py  judge_validation.py
   │
   ▼
results/*.json ──► summarize.py ──► results/SUMMARY.md ──► REPORT.md
             └───► figures.py   ──► figures/*.png
```

## Modules

### `build_benchmark.py` — constructs CARB
Gold labels are never invented. Every item inherits its label from its source dataset's own
annotation through the fixed table `CATEGORY_MAP`, which is printed at build time. Two cells
are flagged `contested` so the analysis can recompute headline numbers without them. Sampling
is balanced at the level of the gold action (`SEED = 42`).

Run: `python -m carb.build_benchmark`

### `build_sft.py` — training set for Experiment 4
Draws ASK/REFUSE/DEFER from CoCoNot's **train** split (CARB uses only its test split) and ACT
from the CoCoNot contrast items that CARB did not sample plus unambiguous AmbigQA questions.
CLAMBER and IN3 are never trained on, so the transfer split is genuine out-of-source.

### `prompts.py` — the independent variable
The five elicitation regimes (R0 direct, R1 affordance, R2 typed ontology, R3 scalar
confidence, R4 recognition-only), the behavioural judge prompt, and the three announced-stakes
frames. This is the file to read first: the experiment *is* these prompts.

### `llm.py` — OpenRouter client
Async, concurrency-limited, with a SQLite response cache (`results/llm_cache.sqlite`) keyed on
(model, messages, temperature, max_tokens), exponential-backoff retries, and token accounting
written to `results/usage_*.json`. Caching is what makes a re-run free.

### `run_eval.py` / `run_stakes.py` — data collection
Stage 1 collects raw completions for each (model, regime, item) cell; stage 2 runs the
behavioural judge over the free-text regimes. Both stages are cached per output file, so the
scripts are safe to interrupt and resume.

### `local_judge.py` — the judge that actually produced the labels used here
The API budget was exhausted before the judge could run, so the judge was re-implemented on a
local open-weight model. It scores by *constrained choice* rather than free generation: the
judge prompt is completed with the fixed prefix `{"behaviour": "` and the four action words
are compared at that slot, which makes format errors impossible.

- `--method first_token` (default): one forward pass per item, argmax over the four first-token
  logits. Valid because ACT/ASK/REF(USE)/DEF(ER) have distinct first tokens under the Qwen3
  tokenizer (asserted at runtime).
- `--method full`: one pass per (item, action), comparing mean log-prob of the whole word.
- `--shard i/n` splits the input files across GPUs; `--tag` suffixes the output files so two
  judges can be run and compared.

Run: `CARB_JUDGE_MODEL=Qwen/Qwen3-14B CUDA_VISIBLE_DEVICES=0 python -m carb.local_judge --what both`

### `routing.py` — deterministic output → action rules
`recognition_to_action` is the fixed rule that turns R4's four property judgments into an
action; it mirrors the decision procedure given to the model in R2, so R2-vs-R4 isolates *who*
applies the rule. `fit_scalar_router` searches all 3-cut × 24-permutation routers over the
scalar confidence, i.e. it gives the scalar baseline the best router that could ever be fitted
to it.

### `derive.py` — one predicted action per (split, model, regime, item)
R0/R1 come from the judge; R2 from the parsed JSON action; R3 from **cross-validated**
thresholds (`cv_scalar_preds`, 5-fold, each fold's router fitted on the other four, so no item
is predicted by a router that saw it); R4 from the fixed rule.

### `metrics.py` — the metric set
Accuracy, macro-F1, ASK-F1, over-commitment, contrast-set compliance, typed-deferral accuracy,
deferral detection, item-level percentile bootstrap CIs, McNemar's exact test, Cohen's h, Holm
correction. Headline metrics are deliberately paired: ASK-F1 is gameable by always asking and
contrast compliance by always acting, so both are always reported.

### `analyze.py` — Experiments 1 and 2
Per-cell metrics with CIs and confusion matrices, the pre-registered family of within-item
McNemar comparisons with Holm correction, scalar-confidence diagnostics (AUROC overall and
pairwise among the deferral types), and label-mapping sensitivity under three alternative
labelings. Writes `results/main_analysis.json`.

### `analyze_stakes.py` — Experiment 3
Cell rates by (regime × gold × frame), paired HIGH-vs-LOW exact tests, a signal-detection
decomposition (d′ and criterion c with bootstrap CIs) that separates "asks more" from "knows
better when to ask", an item-clustered logistic regression, and the IN3 item-level importance
contrast. Models whose cells are >5% missing are excluded and named.

### `analyze_local.py` / `local_model.py` — Experiment 4
`local_model.py --stage {zeroshot,train,eval_trained,probe}` runs the open-weight arm:
zero-shot R2/R3/R4, LoRA SFT on the typed-routing task, re-evaluation of the tuned adapter, and
a frozen-representation linear probe on the last hidden state at the final prompt token.

### `utility.py` — decision-theoretic evaluation
Prices every error: acting on an item that should have been withheld costs `K`, a needless
question costs 0.3, a needless refusal costs 1.0, withholding for the wrong reason earns 0.2.
Sweeping `K` from 0.5 to 30 answers "at what stakes does each policy stop being worth
deploying, relative to always asking".

### `budget.py` — calibration and interaction budgets
ECE of the verbalised confidence as a forecast of "this item was really ACT", plus ask-recall
when the policy may only interrupt on the b% least-confident requests, against a random-ranking
control and the typed policy's own operating point.

### `recognition.py` — error analysis of R4
Scores each of the four property judgments separately, but only where the fixed rule actually
reads it, so no item is penalised for a judgment that could not have changed the outcome.

### `artifact_check.py` — is the benchmark solvable by surface features?
Trains TF-IDF + logistic regression on a balanced CoCoNot pool disjoint from CARB and reports
its accuracy on the test and transfer splits, alongside always-ACT / always-ASK /
always-REFUSE / majority / uniform-random policies. This is the control that says how much of
any model's score is lexical.

### `judge_validation.py` — is the judge trustworthy?
`--cross-judge` measures agreement between the two independent local judges over every judged
item. `--export` writes a stratified sample plus a *blind* copy containing only the request and
the response; `--ingest` merges annotations back and `--score` reports raw agreement and
Cohen's kappa against the judge.

### `verify_report.py` — the report's numbers, as assertions
Every quantitative claim in REPORT.md is restated as a `chk(...)` against the JSON that produced
it. `python -m carb.verify_report` prints PASS/FAIL per claim and exits non-zero if any number has
drifted. Run it after any re-analysis, before trusting the prose.

### `summarize.py` / `figures.py` — reporting
`summarize.py` renders every results JSON into `results/SUMMARY.md`; the report quotes those
tables so each number has one checkable source. `figures.py` writes all ten figures.

## Reproducing

```bash
uv venv && source .venv/bin/activate && uv sync
export PYTHONPATH=src HF_HOME=$PWD/.hf_cache CC=$PWD/.toolbin/cc PATH=$PWD/.toolbin:$PATH

# 1. benchmark (fast, CPU)
python -m carb.build_benchmark && python -m carb.build_sft

# 2. API collection (needs OPENROUTER_KEY; ~$50, ~7.7k calls, cached in results/llm_cache.sqlite)
python -m carb.run_eval --split test --alt-judge
python -m carb.run_stakes

# 3. local GPU arm (~2.5 h on one A6000 each; the four stages are independent)
python -m carb.local_model --stage zeroshot
python -m carb.local_model --stage probe
python -m carb.local_model --stage train
python -m carb.local_model --stage eval_trained
CARB_JUDGE_MODEL=Qwen/Qwen3-14B python -m carb.local_judge --what both

# 4. analysis + figures (CPU, ~10 min; analyze.py is the slow one: 2,000-sample bootstraps)
python -m carb.artifact_check
python -m carb.analyze
python -m carb.analyze_stakes
python -m carb.analyze_local
python -m carb.recognition
python -m carb.utility
python -m carb.budget
python -m carb.judge_validation --cross-judge
python -m carb.summarize
python -m carb.figures
python -m carb.verify_report      # re-checks every number quoted in REPORT.md (42 assertions)
```

### Determinism
`SEED = 42` everywhere (Python `random`, NumPy, torch). API calls use `temperature=0`; local
generation uses greedy decoding with `enable_thinking=False`. The judge is a deterministic
argmax over four logits. The one irreducible source of variation is the API providers
themselves, which do not guarantee bitwise-identical outputs at temperature 0.

### Notes and gotchas
- `uv add` fails against the template `pyproject.toml` unless `[tool.uv] package = false` is
  set; it is set.
- Triton needs a C compiler that this container does not ship. `.toolbin/cc` is a `zig cc`
  shim; export `CC` and put `.toolbin` on `PATH` before any torch GPU work.
- `datasets>=4` dropped script loaders, so `facebook/AbstentionBench` and `siyue/SituatedQA`
  cannot be loaded in this environment. Do not downgrade — the CoCoNot/QuestBench loaders need v4.
- CLAMBER's raw JSONL is double-JSON-encoded; `datasets/clamber/` holds the normalised copy.

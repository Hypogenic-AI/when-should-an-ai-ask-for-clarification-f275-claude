# When should an AI ask for clarification?

A four-way action-routing benchmark (**CARB**: act / ask / refuse / defer) built from
independently annotated sources, plus four experiments testing whether frontier and open-weight
models can tell when they lack the information to act safely — and whether they do it *by
estimating uncertainty*, as the hypothesis claims — scored both on accuracy and under an explicit
cost model.

**Full write-up: [REPORT.md](REPORT.md).** Code tour: [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).
Every number: [results/SUMMARY.md](results/SUMMARY.md).

## Key findings

- **Structure beats uncertainty.** Giving the model an explicit typed ontology (act/ask/refuse/
  defer with definitions and a decision procedure) beats routing its own verbalised confidence by
  **15–28 accuracy points** on identical items, for all four frontier models
  (McNemar, all *p*<sub>Holm</sub> < 1e-8). A single confidence number separates "something is
  wrong" (AUROC 0.79–0.87) but is **at chance** at separating *why* (defer-vs-refuse AUROC
  0.44–0.56). Adding it back into a typed policy as a gate helps no model and hurts one.
- **Models notice danger, not underspecification.** Asked only to judge properties of a request,
  they are 89% correct on "is this safe?" and 94% on "is this within my capability?" — but
  **56.5%** on "was I told enough?", the judgment the whole question turns on.
- **The recognition–behaviour gap is real.** Under a plain prompt the models act on **45–58%** of
  requests they should have withheld action on; the action implied by their *own* property
  judgments would have acted on only **11–28%**. Naming the options without defining them is not
  enough — it helps two models and does nothing measurable for the other two.
- **Announced stakes move the threshold, not the judgment.** "This is high-stakes" raises the
  ask-rate where asking is warranted (0.29 → 0.43) *and* where it is not (0.05 → 0.11). A paired
  bootstrap of the change separates the two: the asking **criterion** moves (Δ*c* = −0.44,
  *p* < 0.001) while **discrimination** does not improve (Δ*d′* = −0.11, *p* = 0.53) — and in the
  free-text regime it gets significantly *worse* (Δ*d′* = −0.58, *p* = 0.013). The
  ambiguity × stakes interaction is exactly null (β = +0.010, *p* = 0.97).
- **Training works in-distribution and not out of it.** LoRA SFT on an open Qwen3-4B: 0.577 →
  **0.794** (*p* = 8e-15), over-commitment 17% → 1.4%. On an out-of-source split it drops to
  0.470, *below* the un-trained prompted baseline. A frozen linear probe on the same base model's
  hidden state already gets **0.779** with no training, and fails out-of-source the same way — so
  what SFT mostly adds is a read-out of a signal the representation already carried. The TF-IDF
  control shows the same shape (0.702 in-source vs 0.355 out-of-source): much of the in-source
  signal is surface structure, and the report says so throughout.
- **Priced honestly, none of it beats always asking yet.** Under an explicit cost model, every
  frontier model's prompted policy drops below the trivial always-ask policy once an unwarranted
  action costs ~5–10× a needless refusal, and goes negative shortly after.

## Benchmark

`datasets/carb/carb_v1.jsonl` — 840 items (dev 160 / **test 480** / transfer 200). Sampling is
balanced by gold action over the core pool; on the test split the reference points are
uniform-random 0.231 and best-single-action (always-ASK) 0.285. Labels are inherited from CoCoNot (original + contrast), CLAMBER and
IN3 through a fixed, printed mapping; none are invented here. Contested mapping cells are flagged
and every headline number is recomputed without them.

## Reproducing

```bash
uv venv && source .venv/bin/activate && uv sync
export PYTHONPATH=src HF_HOME=$PWD/.hf_cache CC=$PWD/.toolbin/cc PATH=$PWD/.toolbin:$PATH

python -m carb.build_benchmark && python -m carb.build_sft   # benchmark  (CPU, seconds)
python -m carb.run_eval --split test --alt-judge             # API grid   (needs OPENROUTER_KEY)
python -m carb.run_stakes                                    # stakes factorial
python -m carb.local_model --stage zeroshot                  # GPU arm    (one A6000 per stage)
python -m carb.local_model --stage probe
python -m carb.local_model --stage train --epochs 2 --bsz 4 --accum 4
python -m carb.local_model --stage eval_trained
CARB_JUDGE_MODEL=Qwen/Qwen3-14B python -m carb.local_judge --what both
for m in artifact_check analyze analyze_stakes analyze_local recognition utility budget summarize figures; do
  python -m carb.$m
done
```

Seed 42 everywhere; `temperature=0` for API calls, greedy decoding locally. API responses are
cached in `results/llm_cache.sqlite`, so a re-run costs nothing. Full cost of the original run:
**7,668 calls, $50.79**, plus ~6 GPU-hours on RTX A6000s.

> The behavioural judge in this run is a local `Qwen/Qwen3-14B` scored by constrained choice, not
> `openai/gpt-4.1`: the API budget ran out after the raw completions were collected but before
> judging. Judge reliability is reported in REPORT.md §4 (κ = 0.78 against an independent
> second judge over 3,768 items; κ = 0.75 against 80 blind annotations).

## Layout

```
REPORT.md              full write-up with all results and threats to validity
CODE_WALKTHROUGH.md    what every module does and how the data flows
planning.md            Phase-1 direction budget: 11 directions scored, 3 kept
literature_review.md   59-paper synthesis (from the resource-finding phase)
resources.md           catalogue of papers / datasets / repos on disk
src/carb/              all code (18 modules, run as `python -m carb.<name>`)
datasets/carb/         the benchmark + the disjoint SFT training set
results/               raw model outputs, judge labels, every analysis JSON, SUMMARY.md
figures/               fig1–fig10
logs/                  run logs for every stage, including the failures
```

# Datasets

Data files are **not** committed to git (see `.gitignore`) — only this README, `summary.json`,
and `samples/` (3 records per dataset). Everything below is reproducible from the commands given.

Prerequisite for all HuggingFace downloads:

```bash
source .venv/bin/activate
export HF_HOME=$PWD/.hf_cache
```

Quick machine-readable inventory: `datasets/summary.json`.

---

## Inventory

| Dataset | Location | Size | Signal it provides | Role for this hypothesis |
|---|---|---|---|---|
| CoCoNot (original) | `coconot_original/` | 11,477 train / 1,001 test | 5 noncompliance categories × 17 subcategories | **Primary** — the 4-way action space |
| CoCoNot (contrast) | `coconot_contrast/` | 379 test | Superficially similar but *compliable* prompts | **Primary** — over-refusal / over-asking control |
| CLAMBER | `clamber/` | 3,202 (1,601/1,601 balanced) | Binary `require_clarification` + ambiguity taxonomy | **Primary** — ask-vs-answer decision |
| IN3 | `in3/` | 1,261 train / 108 test | `vague` bool + missing details with **importance 1–3** | **Primary** — the *stakes* dimension |
| QuestBench | `questbench/` | 38,883 (4 configs) | Formally verified sufficient-vs-insufficient info | **Primary** — ground-truth "is info sufficient" |
| AmbigQA (light) | `ambigqa_light/` | 10,036 / 2,002 | Ambiguity flag + disambiguated QA pairs | Intent-entropy estimation (INTENT-SIM) |
| AmbiEnt | `ambient/` | 100 val / 1,545 test | Ambiguity labels + disambiguations for NLI | Aleatoric-vs-epistemic separation |
| ClariQ | `clariq/` | 9,176 train / 2,313 dev rows (187/50 topics) | **Graded** `clarification_need` 1–4 | Graded (not binary) ask thresholds |
| HumanEvalComm | `humanevalcomm/` | 164 problems × 4 perturbations | Ambiguous/inconsistent/incomplete code prompts + executable tests | Cost of *not* asking, measured by execution |
| ClarEval | `clareval/` | 492 single-turn + 492 multi-turn | Fuzzy code specs + evaluation criteria | Multi-turn clarification policy |
| SWE-bench underspec. | `swebench_underspec/` | 1,699 instances | Underspecification severity 0–3 + difficulty | Real-task stakes × ambiguity grid |

Total: ~11 datasets, ~15 MB on disk.

---

## Primary datasets

### CoCoNot — Contextual Noncompliance

The single best fit for the hypothesis: its category structure *is* the action space.

- **Source**: `allenai/coconot` (HuggingFace)
- **Splits**: original 11,477 train / 1,001 test; contrast 379 test
- **Format**: HuggingFace `DatasetDict`, columns `id, category, subcategory, prompt, response`
- **License**: see the dataset card (ODC-BY)

Category → hypothesis action mapping:

| CoCoNot category | n (train) | Maps to |
|---|---|---|
| Incomplete requests (subcat `underspecified`, `false presuppositions`) | 3,838 | **ASK** |
| Unsupported requests (modality/capability limits) | 1,807 | **DEFER** |
| Requests with safety concerns | 3,136 | **REFUSE** |
| Indeterminate requests (subjective, no single answer) | 901 | **REFUSE / hedge** |
| Humanizing requests | 1,795 | REFUSE-adjacent |
| **Contrast set** (379) | — | **ACT** (must *not* refuse) |

The contrast set is what makes this dataset unusually valuable: it holds surface form roughly
fixed while flipping the correct action, so it measures over-refusal directly rather than
letting a always-refuse policy score well.

```python
from datasets import load_dataset
load_dataset("allenai/coconot", "original").save_to_disk("datasets/coconot_original")
load_dataset("allenai/coconot", "contrast").save_to_disk("datasets/coconot_contrast")
```

### CLAMBER

- **Source**: <https://github.com/zt991211/CLAMBER> → `clamber_benchmark.jsonl`
- **Size**: 3,202 items, exactly balanced (1,601 need clarification / 1,601 do not)
- **Columns**: `question, context, clarifying_question, require_clarification, category, subclass`
- **Taxonomy**: `MC` multiple choices (1,602), `FD` (800), `LA` (800); subclasses include
  `what/whom/when/where`, `polysemy`, `co-reference`, `NK`, `ICL`

Note the file is **double-JSON-encoded** — each line is a JSON *string* containing a JSON object:

```python
import json
rows = []
for line in open("datasets/clamber/clamber_benchmark.jsonl"):
    obj = json.loads(line)
    rows.append(json.loads(obj) if isinstance(obj, str) else obj)
```

The staged copy in `datasets/clamber/` has already been normalized to single-encoded JSONL, so
plain `json.loads(line)` works there. The raw double-encoded original is at
`code/clamber/clamber_benchmark.jsonl`.

The bundled `predict_*` columns are the *authors'* GPT baseline outputs — useful as a reference
point, but do not mistake them for labels.

```bash
git clone --depth 1 https://github.com/zt991211/CLAMBER code/clamber
```

### IN3 — Intention-in-Interaction

Carries the **stakes signal** the hypothesis needs: each missing detail has an importance rating.

- **Source**: <https://github.com/HBX-hbx/Mistral-Interact> → `data/IN3/{train,test}.jsonl`
- **Size**: 1,261 train (1,012 vague / 249 clear), 108 test
- **Columns**: `category, task, vague, thought, missing_details`
- **`missing_details[]`**: `{description, importance (1–3), inquiry, options}` —
  importance distribution over train: 1→558, 2→2,449, 3→608

Importance lets you weight the cost of a wrong assumption per-detail, which is exactly the
"depending on the stakes" clause of the hypothesis — without needing to synthesize stakes labels.

### QuestBench

Ground truth for *whether information is sufficient*, verified formally rather than by annotator
judgment — so the "should it ask?" label is not itself noisy.

- **Source**: `belindazli/QuestBench` (HuggingFace), configs `Logic-Q`, `Planning-Q`, `GSME-Q`, `GSM-Q`
- **Sizes**: 1,150 / 7,500 / 6,591 / 23,642 (all `test`)
- **Key columns**: `gt_qs` (the sufficient question(s) to ask), `all_valid_qs`, `all_qs`,
  `cannot_ask_facts`, plus `depth` / `num_vars` as difficulty knobs

```python
from datasets import load_dataset, DatasetDict
out = {}
for cfg in ["Logic-Q", "Planning-Q", "GSME-Q", "GSM-Q"]:
    for split, v in load_dataset("belindazli/QuestBench", cfg).items():
        out[f"{cfg}_{split}"] = v
DatasetDict(out).save_to_disk("datasets/questbench")
```

`depth` and `num_vars` give a controllable difficulty axis, which is a cleaner way to test the
stakes/ambiguity interaction than hand-written severity labels.

---

## Supporting datasets

### AmbigQA (light)

- **Source**: `sewon/ambig_qa`, config `light` — 10,036 train / 2,002 val
- **Columns**: `id, question, annotations` (annotations carry `singleAnswer` vs `multipleQAs`)
- Used by *Clarify When Necessary* for the INTENT-SIM intent-entropy method.

```python
load_dataset("sewon/ambig_qa", "light").save_to_disk("datasets/ambigqa_light")
```

### AmbiEnt

- **Source**: `metaeval/ambient` (mirror of Liu et al. 2023) — 100 val / 1,545 test
- **Columns**: `premise, hypothesis, premise_ambiguous, hypothesis_ambiguous, labels, disambiguations`
- Ambiguity is defined by its effect on entailment, which separates *aleatoric* (ambiguity →
  ask) from *epistemic* (missing knowledge → defer/refuse) uncertainty.

```python
load_dataset("metaeval/ambient").save_to_disk("datasets/ambient")
```

### ClariQ

- **Source**: <https://github.com/aliannejadi/ClariQ> → `data/*.tsv`
- **Size**: 9,176 train rows over 187 topics; 2,313 dev rows over 50 topics
- **`clarification_need`** is graded **1–4**, not binary (train topics: 1→25, 2→74, 3→62, 4→26)
- **Columns**: `topic_id, initial_request, topic_desc, clarification_need, facet_id, facet_desc, question_id, question, answer`

Grading matters: it supports calibration-style analysis (does predicted ask-probability track
the graded need?) rather than only accuracy on a binary flag. Note rows are facet-level, so
deduplicate by `topic_id` before treating `clarification_need` as the unit of analysis.

### HumanEvalComm

- **Source**: <https://github.com/jie-jw-wu/human-eval-comm> → `Benchmark/HumanEvalComm.jsonl`
- **Size**: 164 HumanEval problems, each with 4 perturbations
- **Columns**: `prompt` (original), `prompt1a` (ambiguous), `prompt1c` (inconsistent),
  `prompt1p` (incomplete), `prompt2ap` (ambiguous+incomplete), `solution`, `test_case`
- The `test_case` field makes the cost of a wrong assumption **executable** rather than judged.

### ClarEval

- **Source**: <https://github.com/JialinLi13/ClarEval> → `data_synthesis/*.jsonl`
- **Size**: 492 single-turn + 492 multi-turn
- **Columns**: `header, original, fuzzy_version, evaluation_criteria, ground_truth_solution, original_prompt_source`
  (multi-turn adds `multi_turn_dialogue`)

### SWE-bench underspecification annotations

- **Source**: <https://github.com/nedwards99/ask-or-assume> →
  `analysis/swe-bench-annotation-results/ensembled_annotations_public.csv`
- **Size**: 1,699 SWE-bench instances, human-ensembled
- **`underspecified`** severity 0–3: 0→396, 1→653, 2→542, 3→108
- **`difficulty`**: `<15 min` (417), `15 min–1 hr` (906), `1–4 hr` (329), `>4 hr` (47)

Difficulty × underspecification gives a genuine stakes-vs-ambiguity grid on *real* tasks — the
cost of guessing wrong on a `>4 hour` task is materially different from a `<15 min` one. Note
`filter_out` is True for 1,160 of 1,699 rows; respect it or state that you did not.

---

## Not downloaded (and why)

- **AbstentionBench** (`facebook/AbstentionBench`): the HF repo is a **loading script**, which
  `datasets>=4` no longer supports (`RuntimeError: Dataset scripts are no longer supported`).
  The repo is cloned at `code/abstentionbench/` and its 20 component datasets can be rebuilt
  in a **separate** venv with `datasets==3.6.0` — do not downgrade this workspace's `datasets`,
  as the loaders above depend on v4 behavior. Its `analysis/abstention_performance.csv`
  (published per-model abstention results) is usable directly, with no download needed.
- **SituatedQA** (`siyue/SituatedQA`): same script-loader problem. Reachable through the
  AbstentionBench path above if needed.
- **Abg-CoQA**: no arXiv/HF mirror found; the AKBC 2021 paper is not openly hosted.
- **RegretBench**: cloned at `code/regretbench/`, but ships only manifests — episodes are
  generated at eval time by running the harness, not distributed as static data.

## Reproduce everything

`datasets/summary.json` records splits and columns for each HF-format dataset, so you can verify
a re-download matches what was used here.

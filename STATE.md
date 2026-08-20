# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-08-20T02:26:26.494193Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Expected: `/workspaces/when-should-an-ai-ask-for-clarification-f275-claude`
- Actual: `/app`
- Directory usable: `True`
- Current process matches workspace: `False`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
### resource_finder

**Status**: complete (2026-08-19). All expected artifacts exist on disk.

**Environment**: fresh `uv venv` at `.venv/` with `[tool.uv] package = false` in `pyproject.toml`
(required — the template pyproject makes `uv add` fail on the hatchling build). Installed: pypdf,
requests, arxiv, httpx, datasets, huggingface_hub, pandas. Set `export HF_HOME=$PWD/.hf_cache`
before any HF download. NOTE: paper-finder silently writes an empty results file unless `httpx`
is installed.

**Delivered**
- `papers/` — 59 PDFs, all pypdf-validated (0 corrupt). Catalog `papers/catalog.json` +
  `scripts/download_papers.py` rebuild the corpus (PDFs are gitignored). Text cache `.paper_text/`.
- `datasets/` — 11 datasets, ~15 MB, all loaded and inspected. `datasets/summary.json` records
  splits/columns; `datasets/samples/` has 3 records each.
- `code/` — 8 repos (all literature-found; the topic spec listed no `code_references`).
- `literature_review.md`, `resources.md`, `planning.md`.

**Search**: 4 paper-finder diligent queries -> 341 records -> 317 unique -> 167 at relevance >=2
-> 64 curated -> 59 retrieved. The 5 misses were 4 duplicate titles of papers already held plus
Abg-CoQA (no open host); net coverage loss is one paper.

**Key findings that should shape the experiment**
1. RECOGNITION/BEHAVIOR GAP (arXiv 2605.25284): models recognize ambiguity when asked to judge it
   but default to direct answers in ordinary QA. So "can be prompted to recognize" is largely
   ALREADY TRUE — an experiment measuring only recognition confirms the hypothesis trivially.
   Recognition and behavior must be measured separately.
2. SCALAR UNCERTAINTY IS INSUFFICIENT (SSTA-32, arXiv 2604.16752): scalar confidence collapses the
   three-way deferral space (58.3% typed accuracy) while typed categorical prompting hits 91.7%
   and cuts overcommitment 41.7%->8.3%. BAG (2605.25831) independently reports clarify-vs-abstain
   separation "remains challenging". This puts direct pressure on the hypothesis's
   "by estimating its uncertainty" clause.
3. JUDGMENT, NOT CAPABILITY (HiL-Bench, 2604.09408): frontier models drop 75-89% -> 4-24% pass@3
   when they must decide whether to ask, despite having an ask_human() tool. RLVR on shaped ASK-F1
   makes a 32B model better and the gains transfer across domains.
4. EFFECT SIZES ARE SMALL: published AUROCs for "will clarification help" sit at 0.50-0.63; some
   methods (Self-Ask on MT, 0.371) score below chance. Power the design accordingly.

**Primary datasets** (why): CoCoNot original+contrast — its native categories ARE the action space
(Incomplete->ASK, Unsupported->DEFER, Safety/Indeterminate->REFUSE, contrast->ACT) and the contrast
set is the only built-in over-refusal control found; CLAMBER (3,202, exactly balanced);
IN3 (importance 1-3 = the stakes axis, already annotated); QuestBench (38,883, formally verified
sufficiency so labels aren't annotator-noisy).

**Metrics**: ASK-F1 and typed deferral accuracy as primary, ALWAYS with contrast-set compliance —
neither alone is meaningful, since degenerate always-ask/always-answer policies each score well on
exactly one. Report the confusion matrix; aggregate 4-way accuracy hides the clarify-vs-abstain
confusion that is the field's open problem. Reuse AbstentionBench's `abstention_keywords.py` and
`evaluation_judge_prompts.py` for comparability with published numbers.

**Gotchas for the next phase**
- CLAMBER's raw JSONL is DOUBLE-JSON-ENCODED; use the normalized copy in `datasets/clamber/`.
- `datasets>=4` dropped script loaders, so `facebook/AbstentionBench` and `siyue/SituatedQA` cannot
  be loaded here. Do NOT downgrade — CoCoNot/QuestBench/AmbiEnt loaders need v4. Use a separate
  `datasets==3.6.0` venv if AbstentionBench data is truly needed.
- ClariQ rows are facet-level; dedupe by `topic_id` before treating `clarification_need` as the
  unit of analysis. SWE-bench annotations set `filter_out=True` on 1,160 of 1,699 rows.
- Stakes labels are annotated PROXIES, not experienced consequences. State this as a limitation.

**Direction budget** (full scoring in `planning.md`): kept D1 typed action ontology vs scalar
uncertainty (20), D3 recognition-behavior decomposition (19), D2 stakes x ambiguity factorial (18).
Pruned D4 better uncertainty estimator (low information gain), D5 RL ask-policy (no GPU budget —
the strongest prune, revisit if compute appears), D6 conformal (stakes labels too coarse),
D7 multi-turn stopping (RegretBench ships no episodes), D8/D9/D10/D11 (see planning.md).
D1/D2/D3 share one experimental substrate and should run as a single factorial.

**Next phase**: `experiment_runner`. Concrete next steps —
1. Read `planning.md` first, then `literature_review.md` sections 5 (metrics) and 7 (cautions).
2. Build the shared harness: load CoCoNot/CLAMBER/IN3, map to the 4-way action space, implement
   ASK-F1 + typed deferral accuracy + contrast-set compliance.
3. Run D3 (recognition vs behavior on identical items, 3 elicitation regimes) FIRST — it is the
   baseline D1 and D2 both need.
4. Then D1 (scalar-uncertainty routing vs typed categorical prompting) and D2 (stakes x ambiguity).
5. Include an always-ask baseline in every comparison.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
### experiment_runner

**Status**: COMPLETE (2026-08-20). Resumed a prior, interrupted execution session. Phases 0-1
were delivered by `resource_finder` (`planning.md` direction budget: D1/D2/D3 kept). This session
completed Phases 2-6.

**What the prior session left behind (verified on disk before resuming)**
- `datasets/carb/carb_v1.jsonl` (840 items) + `sft_train.jsonl` (2,800) — benchmark built.
- `results/raw/` — 9,600 API completions (4 models x 5 regimes x 480 test items). COMPLETE.
- `results/raw_stakes/` — 4,320 completions (3 models x 2 regimes x 3 frames x 240). GPT-5 cells
  mostly EMPTY (budget death mid-run).
- `results/raw/judged*` — **all 3,840 judge labels were `null`**: the OpenRouter key hit its $50
  limit (HTTP 403) between stage 1 and stage 2, so Experiment 1 had no data at all.
- Local arm: Qwen3-4B zero-shot R2/R3/R4 + linear probe done; LoRA SFT killed mid-epoch-1.
- No analysis outputs, no figures, no REPORT.md.

**Blocking constraint discovered**: `OPENROUTER_KEY` is exhausted (usage $50.79 / limit $50) and
`OPENAI_API_KEY` is empty. **No further API calls are possible in this workspace.** Everything
below was therefore done on the local GPUs.

**What this session did**
1. Re-implemented the behavioural judge locally (`src/carb/local_judge.py`, extended with a
   single-forward-pass `first_token` constrained-choice scorer, `--shard`, `--tag`, `--what local`).
   Primary judge Qwen3-14B, second judge Qwen3-4B. Judged all R0/R1 cells (main + stakes + local).
   Reliability: kappa 0.78 vs the second judge (n=3,768); kappa 0.75 vs 80 blind annotations made
   by this agent (NOT human validation — stated as such in the report).
2. Recovered IN3 `importance` (the builder's `ast.literal_eval` branch silently dropped every
   rating because the column ships as a list). Only previously-null fields changed; no gold labels.
3. Excluded GPT-5 from Experiment 3 (56-100% missing cells) with the missingness printed.
4. Added five analyses that were not in the prior code: decision-theoretic utility sweep
   (`utility.py`), calibration + interaction-budget curves (`budget.py`), R4 per-property error
   analysis (`recognition.py`), scalar/typed hybrid ablation (`hybrid.py`), and signal-detection
   decomposition of the stakes effect (in `analyze_stakes.py`). Plus `summarize.py` ->
   `results/SUMMARY.md` so every reported number has one checkable source.
5. Retrained the LoRA adapter (2 epochs, bsz 4 x accum 4, per-epoch checkpointing) after the
   original 3x8x2 run was OOM-killed by a co-tenant process on the shared GPU, then ran
   `--stage eval_trained` on test + transfer.
6. Ran the local Qwen3-4B on the free-text regimes (R0/R1) as well, so the recognition-behaviour
   gap is measured on an open-weight model too (it replicates: over-commitment 58% -> 17%).
7. Delivered `REPORT.md`, `README.md`, `CODE_WALKTHROUGH.md`, `results/SUMMARY.md`, 10 figures.
   Verified reproducibility by re-running every deterministic analysis (byte-identical output),
   and wrote `src/carb/verify_report.py`, which restates all 42 quantitative claims in REPORT.md
   as executable assertions against the JSON that produced them (42/42 passing; non-zero exit if
   any number drifts). Run it first if you re-analyse anything.

**Headline results** (all in `REPORT.md` / `results/SUMMARY.md`)
- Typed ontology (R2) beats optimally-routed scalar confidence (R3) by 0.16-0.28 accuracy on all
  four models, every p_holm < 1e-8. Scalar is at chance on DEFER-vs-REFUSE (AUROC 0.44-0.56).
- Recognition (R4) beats default behaviour (R0) on all four models; over-commitment 45-58% -> 11-28%.
  But recognition of "information_sufficient" on ASK items is only 0.565 — the specific judgment
  the hypothesis is about is the one that fails.
- Announced stakes shift the criterion (c 1.15 -> 0.71, non-overlapping CIs) without improving d';
  ambiguity x stakes interaction beta = +0.010, p = 0.97.
- TF-IDF gets 0.702 on the test split (0.355 on transfer) — the benchmark has lexical shortcuts,
  reported prominently; all headline claims are within-item between-regime contrasts.
- Frozen linear probe on Qwen3-4B: 0.779 test vs 0.577 prompted, but 0.465 on transfer.
- LoRA SFT on Qwen3-4B: 0.794 test (+0.217 over the same model prompted, McNemar p=8.0e-15,
  over-commitment 17% -> 1.4%) but 0.470 on transfer, BELOW the un-trained prompted baseline.
  Since the frozen probe already reaches 0.779 with no training, SFT mostly installs a read-out
  of a signal the base representation already carries. C5 ("can be trained") holds in-distribution
  and fails out of it.
- On the transfer split (CLAMBER, binary 100 ACT / 100 ASK) EVERY condition — trained, probed,
  prompted — sits within +/-0.09 of the always-ACT baseline of 0.50. No routing policy in this
  study generalises across sources; this is stated prominently in REPORT.md 5.4 and 7.6.

**Gotchas for anyone re-running**
- Triton needs `CC=$PWD/.toolbin/cc` and `.toolbin` on PATH; without it every torch GPU job dies
  with "Failed to find C compiler".
- The box is shared. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and expect co-tenant
  processes to appear on any GPU; checkpoint training every epoch.
- `HF_HOME=$PWD/.hf_cache` (the prior session pointed at `/home/neurico/hfcache`, which no longer
  exists). Qwen3-4B and Qwen3-14B are cached there (~37 GB, gitignored).
- The local model was always Qwen3-**4B**; the prior session mislabelled its output files
  `qwen3-8b_*`. Files and labels were renamed.

**Next phase**: none — this is the final phase. If work continues, the priority follow-ups are in
REPORT.md 8: (1) stakes with *experienced* consequences rather than announced ones, (2) training
on ASK-F1 directly rather than the typed label, (3) explaining why the frozen probe does not
transfer, (4) multi-turn stopping.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->

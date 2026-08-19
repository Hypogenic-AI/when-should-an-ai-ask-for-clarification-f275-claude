# Research State

- Current phase: `None`
- Pipeline completed: `False`

## Previous phases

resource_finder (succeeded), experiment_runner (failed)

## Current phase context

- Phase: `experiment_runner`
- Status: `failed`
- Started: `2026-08-19T20:45:03.017656Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Expected: `/workspaces/when-should-an-ai-ask-for-clarification-f275-claude`
- Actual: `/app`
- Directory usable: `True`
- Current process matches workspace: `False`

## Output validation

- Valid: `False`
- Expected: `REPORT.md`
- Missing: `REPORT.md`
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
Update this section at the end of the `experiment_runner` phase.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->

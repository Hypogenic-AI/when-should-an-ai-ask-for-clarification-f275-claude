# Resources Catalog

**Topic**: When should an AI ask for clarification?
**Phase**: `resource_finder` · **Date**: 2026-08-19

| Resource | Count | Location | Detail |
|---|---|---|---|
| Papers (PDF) | 59 | `papers/` | `papers/README.md`, `papers/catalog.json` |
| Datasets | 11 | `datasets/` | `datasets/README.md`, `datasets/summary.json` |
| Repositories | 8 | `code/` | `code/README.md` |

Synthesis: `literature_review.md`. Direction ranking: `planning.md`.

---

## Environment

Fresh isolated venv, created before any other work.

```bash
uv venv && source .venv/bin/activate
```

`pyproject.toml` declares `[tool.uv] package = false` — without it, `uv add` fails trying to build
the workspace as a package (`hatchling` finds no package to build). Installed: `pypdf`, `requests`,
`arxiv`, `httpx`, `datasets`, `huggingface_hub`, `pandas`. `httpx` is a hard requirement of the
paper-finder script and is not declared by the skill — it fails with a JSON error and a silent
empty result file if missing.

HuggingFace cache is pinned into the workspace (`export HF_HOME=$PWD/.hf_cache`) and gitignored.

---

## Papers

59 PDFs, all validated with pypdf (0 corrupt, all ≥2 pages). Plain-text extractions cached in
`.paper_text/` (gitignored, regenerable). PDFs are gitignored; `papers/catalog.json` +
`scripts/download_papers.py` reconstruct the corpus from a clean checkout.

| Title | First author | Year | File | Citations |
|---|---|---|---|---|
| CoSQL: A Conversational Text-to-SQL Challenge | Tao Yu et al. | 2019 | `1909.05378_...pdf` | 325 |
| Know Your Limits: A Survey of Abstention in LLMs | Bingbing Wen et al. | 2024 | `2407.18418_...pdf` | 128 |
| R-Tuning: Instructing LLMs to Say 'I Don't Know' | Hanning Zhang et al. | 2023 | `2311.09677_...pdf` | 118 |
| **Clarify When Necessary** | Michael J.Q. Zhang et al. | 2023 | `2311.09469_...pdf` | 110 |
| Tell Me More! Implicit User Intention Understanding | Cheng Qian et al. | 2024 | `2402.09205_...pdf` | 100 |
| Trust or Escalate: LLM Judges with Provable Guarantees | — | 2024 | `2407.18370_...pdf` | 92 |
| CLAM: Selective Clarification for Ambiguous Questions | — | 2022 | `2212.07769_...pdf` | 84 |
| AmbigQA: Answering Ambiguous Open-domain Questions | Sewon Min et al. | 2020 | `2004.10645_...pdf` | — |
| Ask-before-Plan: Proactive Language Agents | — | 2024 | `2406.12639_...pdf` | 73 |
| Adaptation with Self-Evaluation for Selective Prediction | — | 2023 | `2310.11689_...pdf` | 64 |
| Mitigating LLM Hallucinations via Conformal Abstention | — | 2024 | `2405.01563_...pdf` | 63 |
| Learning to Clarify: Action-Based Contrastive Self-Training | — | 2024 | `2406.00222_...pdf` | 59 |
| Conformal Alignment: Knowing When to Trust Foundation Models | — | 2024 | `2405.10301_...pdf` | 55 |
| **CLAMBER** | — | 2024 | `2405.12063_...pdf` | 49 |
| Learning to Ask: When LLM Agents Meet Unclear Instruction | Wenxuan Wang et al. | 2024 | `2409.00557_...pdf` | 41 |
| We're Afraid LMs Aren't Modeling Ambiguity (AmbiEnt) | Alisa Liu et al. | 2023 | `2304.14399_...pdf` | 179 |
| **HumanEvalComm** | Jie JW Wu et al. | 2024 | `2406.00215_...pdf` | 34 |
| Ambig-SWE: Interactive Agents for Underspecificity in SE | — | 2025 | `2502.13069_...pdf` | 27 |
| Structured Uncertainty guided Clarification for LLM Agents | — | 2025 | `2511.08798_...pdf` | 26 |
| CollabLLM: From Passive Responders to Active Collaborators | Shirley Wu et al. | 2025 | `2502.00640_...pdf` | 126 |
| **HiL-Bench: Do Agents Know When to Ask for Help?** | Tu Trinh et al. | 2026 | `2604.09408_...pdf` | 6 |
| **Don't Start What You Can't Finish (SSTA-32)** | Eren Unlu | 2026 | `2604.16752_...pdf` | 0 |
| **Clarify, Abstain or Answer? (BAG)** | Joris Baan et al. | 2026 | `2605.25831_...pdf` | 0 |
| **Knowing but Not Showing** | Jinyan Su, Claire Cardie | 2026 | `2605.25284_...pdf` | 0 |
| SAFETY SENTRY: EXECUTE-ASK-REFUSE Routing | — | 2026 | `2607.13594_...pdf` | 0 |
| I-CALM: Confidence-Aware Abstention | Haotian Zong et al. | 2026 | `2604.03904_...pdf` | 4 |
| LHAW: Controllable Underspecification | George Pu et al. | 2026 | `2602.10525_...pdf` | 4 |
| One More Turn, Less Regret (RegretBench) | Minh Ngoc Ta et al. | 2026 | `2607.21143_...pdf` | 0 |
| Uncertainty Decomposition for Clarification Seeking | Gregory Matsnev | 2026 | `2606.19559_...pdf` | 1 |
| Entropy Alone is Insufficient for Safe Selective Prediction | — | 2026 | `2603.21172_...pdf` | 6 |
| Ask or Assume? Uncertainty-Aware Clarification in Coding | — | 2026 | `2603.26233_...pdf` | 9 |
| *…28 more* | | | see `papers/README.md` | |

**Bold** = deep-read or primary to the hypothesis.

---

## Datasets

11 datasets, ~15 MB. Data gitignored; `datasets/README.md` has per-dataset download commands and
`datasets/samples/` holds 3 records each.

| Name | Source | Size | Task | Location |
|---|---|---|---|---|
| CoCoNot (original) | HF `allenai/coconot` | 11,477 + 1,001 | 4-way noncompliance routing | `datasets/coconot_original/` |
| CoCoNot (contrast) | HF `allenai/coconot` | 379 | Over-refusal control | `datasets/coconot_contrast/` |
| CLAMBER | GitHub `zt991211/CLAMBER` | 3,202 (balanced) | Binary ask-vs-answer | `datasets/clamber/` |
| IN3 | GitHub `HBX-hbx/Mistral-Interact` | 1,261 + 108 | Vagueness + importance 1–3 | `datasets/in3/` |
| QuestBench | HF `belindazli/QuestBench` | 38,883 | Verified info-sufficiency | `datasets/questbench/` |
| AmbigQA (light) | HF `sewon/ambig_qa` | 10,036 + 2,002 | Ambiguous open-domain QA | `datasets/ambigqa_light/` |
| AmbiEnt | HF `metaeval/ambient` | 100 + 1,545 | Ambiguity via entailment | `datasets/ambient/` |
| ClariQ | GitHub `aliannejadi/ClariQ` | 9,176 + 2,313 rows | Graded need 1–4 | `datasets/clariq/` |
| HumanEvalComm | GitHub `jie-jw-wu/human-eval-comm` | 164 × 4 | Executable cost of not asking | `datasets/humanevalcomm/` |
| ClarEval | GitHub `JialinLi13/ClarEval` | 492 + 492 | Fuzzy code specs | `datasets/clareval/` |
| SWE-bench underspec. | GitHub `nedwards99/ask-or-assume` | 1,699 | Severity 0–3 × difficulty | `datasets/swebench_underspec/` |

All were loaded and inspected after download; splits and columns recorded in
`datasets/summary.json`.

---

## Code repositories

| Name | URL | Purpose | Location |
|---|---|---|---|
| AbstentionBench | facebookresearch/AbstentionBench | Judge prompts, keyword detector, published results | `code/abstentionbench/` |
| CLAMBER | zt991211/CLAMBER | Data | `code/clamber/` |
| human-eval-comm | jie-jw-wu/human-eval-comm | Data + executable metric | `code/humanevalcomm/` |
| ClariQ | aliannejadi/ClariQ | Data | `code/clariq/` |
| ask-or-assume | nedwards99/ask-or-assume | SWE-bench annotations | `code/ask_or_assume/` |
| ClarEval | JialinLi13/ClarEval | Data | `code/clareval/` |
| RegretBench | ngocminhta/RegretBench | Regret metric reference | `code/regretbench/` |
| Mistral-Interact | HBX-hbx/Mistral-Interact | IN3 data + reference transcripts | `code/mistral_interact/` |

No `code_references` were specified in the research topic, so all eight were found through the
literature. None is a runtime dependency; details and caveats in `code/README.md`.

---

## Resource gathering notes

### Search strategy
Four paper-finder queries in `--mode diligent`, chosen to cover the three parallel literatures
identified in the review plus the benchmark layer:
1. clarification / ambiguity detection
2. abstention / selective prediction / calibrated refusal
3. ask-for-help under uncertainty / conformal deferral
4. underspecification benchmarks for agents and code generation

341 records → 317 unique after title-normalized dedup → 167 at relevance ≥2 → 64 curated for
download. Semantic Scholar's batch graph API resolved arXiv IDs and open-access PDF links (167/167
resolved, 134 with arXiv IDs).

### Selection criteria
Curated rather than bulk-downloaded, because relevance ≥2 included a long tail of papers using
abstention in unrelated applied domains (financial forecasting, pulmonary nodule triage, resume
matching) that share vocabulary but not substance. Selection favored: papers proposing the
multi-way action space; benchmarks that ship usable data; uncertainty methods applicable to the
decision; and foundational/high-citation work.

### Challenges encountered

- **paper-finder needed `httpx`**, undeclared by the skill. Without it the script returns
  `{"error": "httpx not installed", "fallback": true}` and writes an **empty** results file — a
  silent failure worth knowing about. First search was re-run after installing.
- **`uv add` failed** with the template `pyproject.toml` (hatchling has nothing to build).
  Fixed with `[tool.uv] package = false`.
- **CLAMBER's JSONL is double-encoded** — each line is a JSON string wrapping a JSON object.
  Normalized copy staged in `datasets/`.
- **5 of 64 papers were unretrievable**: `Abg-CoQA` (AKBC 2021, no open host) and four that turned
  out to be **duplicate titles of papers already downloaded** — "The Art of Refusal" ≈ *Know Your
  Limits*, "Learning to Ask: When LLMs Meet Unclear Instruction" ≈ the LLM Agents variant,
  "ClarifyGPT: A Framework…" ≈ "ClarifyGPT: Empowering…". Net loss to coverage: one paper.
- **`datasets>=4` dropped script-based loaders**, blocking `facebook/AbstentionBench` and
  `siyue/SituatedQA`. Not worked around here — downgrading would break the loaders used for
  CoCoNot/QuestBench/AmbiEnt. Documented with the `datasets==3.6.0` sidecar recipe instead.

### Gaps and workarounds

- **SSTA-32 (32 items) is not separately published** — it lives in the paper's appendix. Its
  four-state ontology is reproduced in `literature_review.md` §2.2 and is reconstructable; CoCoNot
  provides a ~400× larger substrate with the same action structure, which is the better path.
- **No dataset natively encodes *consequence-based* stakes.** Available stakes signals are
  annotated proxies: IN3 importance (1–3), SWE-bench difficulty × underspecification, and
  I-CALM-style announced payoffs as a prompt-level manipulation. This is a real limitation of the
  D2 direction and is flagged in `planning.md`, not papered over.
- **RegretBench ships manifests, not episodes** — its data is generated by running the harness with
  a user simulator. Kept as a metric reference only.

---

## Recommendations for experiment design

1. **Primary datasets**: CoCoNot original + contrast (the only pair spanning the full action space
   *with* an over-refusal control), CLAMBER (balanced binary), IN3 (stakes), QuestBench (verified
   sufficiency labels).
2. **Baselines**: direct prompting, always-ask, sequence likelihood, semantic entropy, verbalized
   confidence, INTENT-SIM, BAG, typed categorical prompting. The last two are the real competition,
   not the weak ones.
3. **Metrics**: ASK-F1 and typed deferral accuracy as primary, **always** reported alongside
   contrast-set compliance; overcommitment rate, budget curves at b ∈ {10, 20, 30}%, AUROC, ECE as
   secondary. Report the full confusion matrix — aggregate 4-way accuracy hides the
   clarify-vs-abstain confusion that is the field's actual open problem.
4. **Code to reuse**: AbstentionBench's `abstention_keywords.py` (deterministic detector) and
   `evaluation_judge_prompts.py` (human-validated judges) — reusing these makes results comparable
   to published numbers instead of introducing a bespoke classifier.
5. **Directions**: D1 (typed ontology vs. scalar uncertainty), D2 (stakes × ambiguity factorial),
   D3 (recognition–behavior decomposition) — see `planning.md`. They share one experimental
   substrate and should be run as a single factorial, with D3 supplying the recognition baseline
   the other two need to be interpretable.

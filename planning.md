# Planning: Direction Budget

Per the research-state contract, plausible research directions were enumerated and scored during
Phase 1. **Three** are kept for implementation; the rest are pruned with reasons recorded below.

Scoring dimensions (1–5 each): **Evi** = literature evidence that the direction is live and
unresolved; **Rel** = relevance to the hypothesis as stated; **IG** = expected information gain
(would the result change what we believe?); **Feas** = implementation feasibility with the
resources actually on disk.

---

## Enumerated directions

| # | Direction | Evi | Rel | IG | Feas | Total | Verdict |
|---|---|---|---|---|---|---|---|
| D1 | Typed action ontology vs. scalar uncertainty for 4-way routing | 5 | 5 | 5 | 5 | **20** | **KEEP** |
| D2 | Stakes × ambiguity factorial: does the ask-threshold move with stakes? | 4 | 5 | 5 | 4 | **18** | **KEEP** |
| D3 | Recognition–behavior gap: decomposing "knows" from "acts" | 5 | 4 | 5 | 5 | **19** | **KEEP** |
| D4 | Better uncertainty estimator (INTENT-SIM successor) | 4 | 4 | 2 | 4 | 14 | prune |
| D5 | RL / fine-tuning an ask-policy (ASK-F1 reward) | 4 | 5 | 4 | 1 | 14 | prune |
| D6 | Conformal risk control for the ask-threshold | 4 | 3 | 3 | 3 | 13 | prune |
| D7 | Multi-turn clarification *policy* (when to stop asking) | 4 | 3 | 4 | 2 | 13 | prune |
| D8 | Reasoning-tuned vs base models on asking (extending AbstentionBench) | 3 | 3 | 4 | 2 | 12 | prune |
| D9 | Does retrieval suppress asking? (RAG × clarification) | 2 | 2 | 4 | 3 | 11 | prune |
| D10 | Clarifying-*question quality* generation | 5 | 2 | 1 | 4 | 12 | prune |
| D11 | Cross-domain transfer of ask-judgment | 2 | 3 | 3 | 2 | 10 | prune |

---

## Kept directions

### D1 — Typed action ontology vs. scalar uncertainty (score 20)

**Question.** Can a single scalar uncertainty estimate select among *act / ask / refuse / defer*,
or does the decision require an explicit typed decision structure?

**Why it wins.** It attacks the hypothesis at its weakest joint. The hypothesis says the model
chooses among four actions "*by estimating its uncertainty*" — but SSTA-32 (arXiv 2604.16752)
found scalar confidence collapses the three-way deferral space to 58.3% typed accuracy while
categorical prompting reaches 91.7%, and BAG (2605.25831) independently reports that separating
clarify from abstain "remains challenging." Two papers, different methods, same conclusion — and
nobody has run the head-to-head at scale.

**Why it's feasible.** CoCoNot's native categories *are* the action space (Incomplete → ASK,
Unsupported → DEFER, Safety/Indeterminate → REFUSE, contrast set → ACT), giving ~12,800 labeled
items where SSTA-32 had 32. No annotation required.

**Falsifiable prediction.** Scalar-uncertainty routing will show high act-vs-defer accuracy but
near-chance discrimination *among* the three deferral types; typed prompting will close most of
that gap at equal or lower inference cost.

### D3 — Recognition–behavior decomposition (score 19)

**Question.** Is the failure one of *recognition* (the model can't tell it lacks information) or
of *behavior* (it can tell, and acts anyway)?

**Why it wins.** Highest-leverage correction to how the hypothesis is framed. *Knowing but Not
Showing* (2605.25284) found models identify ambiguity when explicitly asked to judge it but default
to direct answers in ordinary QA. If that replicates, then "can be trained or prompted to
recognize" is **already true** and measuring only recognition would confirm the hypothesis
trivially. Any credible experiment must separate the two — so this is partly infrastructure for D1
and D2 as well as a result in its own right.

**Design.** Same items under three elicitation regimes — (a) plain task prompt, (b) explicit
ambiguity judgment, (c) task prompt with an explicit ask-affordance — and measure the gap between
(b) and (a). CLAMBER (balanced 1,601/1,601) and CoCoNot support this directly.

**Falsifiable prediction.** Recognition accuracy will substantially exceed behavioral ask-rate on
the same items; adding an explicit affordance will close part but not all of the gap.

### D2 — Stakes × ambiguity factorial (score 18)

**Question.** Does the ask/refuse threshold actually move with stakes, as the hypothesis requires?

**Why it wins.** This is the clause of the hypothesis with the *least* existing evidence. Nearly
every paper motivates with stakes; only I-CALM (announced payoff matrices) and SAFETY SENTRY
(threshold repositioning) manipulate them, and no paper in the corpus reports a clean factorial.
It is the most under-tested and therefore highest-information part of the hypothesis.

**Why it's feasible.** The stakes labels already exist and did not need to be synthesized:
IN3 `importance` 1–3 per missing detail; SWE-bench `underspecified` 0–3 crossed with `difficulty`
(<15 min → >4 hr); plus I-CALM-style announced payoffs as a manipulable prompt-level factor.

**Falsifiable prediction.** Ask-rate will respond to *ambiguity* but be substantially less
sensitive to *stakes* than the hypothesis implies — i.e. models will under-differentiate high- from
low-cost errors at matched ambiguity.

**Feasibility caveat (honest).** Stakes here are *annotated proxies*, not consequences the model
experiences. Whatever we find is evidence about stated stakes, and the writeup should say so.

---

## Pruned directions and reasons

- **D4 — Better uncertainty estimator.** Feasible and relevant, but low information gain. Published
  AUROCs sit at 0.50–0.63 and several methods score *below chance* on some tasks; a marginal
  improvement wouldn't move the hypothesis either way. The corpus points at structure, not
  sharper scalars, as the operative variable.
- **D5 — RL / fine-tuning an ask-policy.** Highest relevance of any pruned direction, and
  HiL-Bench shows it works (RLVR on shaped ASK-F1 transferred across domains). Pruned purely on
  feasibility: no GPU budget, no training infrastructure in this workspace. Recorded as the
  natural follow-up if compute becomes available.
- **D6 — Conformal risk control.** Principled route to stakes-sensitive thresholds, but requires
  exchangeable calibration data per stakes level; the available stakes annotations are coarse
  (3–4 levels) and unevenly populated (`>4 hr` has only 47 SWE-bench instances).
- **D7 — Multi-turn stopping policy.** Genuinely interesting (RegretBench frames it well), but
  RegretBench ships manifests only — episodes are generated by running the harness with a user
  simulator, adding a large uncontrolled component and a second source of error.
- **D8 — Reasoning vs base models.** AbstentionBench's ~24% degradation finding is provocative,
  but its data path is blocked (`datasets>=4` dropped script loaders) and testing it properly needs
  matched reasoning/non-reasoning checkpoints.
- **D9 — Retrieval suppresses asking.** Striking finding, but only one paper, and it would need a
  full retrieval pipeline for a second-order effect.
- **D10 — Clarifying-question quality.** Well studied since 2019 (Qulac, ClariQ, and a dedicated
  survey). Zhang & Choi explicitly declined to propose new methods here for the same reason. It is
  the *what to ask* subtask; our hypothesis is about *whether to ask*.
- **D11 — Cross-domain transfer.** Partially answered by HiL-Bench already; low marginal gain.

---

## Change control

Per the contract, the search space stays fixed at D1/D2/D3 unless new evidence invalidates this
ranking. If that happens, the ranking is to be updated here and the change explained in `STATE.md`.

Note that D1, D2, and D3 share a single experimental substrate — the same items, models, and
elicitation harness — so they can be run as one factorial rather than three separate studies. D3
supplies the recognition baseline that D1 and D2 both need in order to be interpretable.

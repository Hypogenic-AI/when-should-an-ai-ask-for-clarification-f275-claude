# Literature Review: When should an AI ask for clarification?

**Hypothesis under test.** An AI can be trained or prompted to recognize when it lacks sufficient
information to act safely, by estimating its uncertainty and choosing among *acting*, *asking for
clarification*, *refusing*, or *deferring*, depending on the stakes and ambiguity of the task.

**Corpus.** 59 papers (2019–2026), gathered via paper-finder in diligent mode across four queries
covering clarification/ambiguity, abstention/selective prediction, uncertainty-aware ask-for-help,
and underspecification benchmarks. 317 unique results were ranked; 167 scored relevance ≥2 and
64 were selected for download (59 retrieved successfully). Full catalog: `papers/README.md`.

---

## 1. Research area overview

The hypothesis sits at the junction of three literatures that, until roughly 2025, ran in parallel
and are only now being unified:

1. **Clarification / ambiguity resolution** (NLP, 2019→). When is a request ambiguous, what should
   be asked, and can the model use the answer? Zhang & Choi's *Clarify When Necessary* is the
   canonical decomposition.
2. **Abstention / selective prediction** (2022→). When should a model decline to answer at all?
   Surveyed comprehensively in *Know Your Limits* (Wen et al., 2024).
3. **Agentic help-seeking** (2024→). When should an agent mid-task escalate to a human rather than
   guess? *HiL-Bench*, *LHAW*, *Ask or Assume*.

The critical observation for this hypothesis is that **these three literatures each test one
boundary at a time** — ask vs. answer, act vs. refuse, search vs. stop — and the unification into a
single multi-way decision is very recent, thin, and unresolved. *Don't Start What You Can't Finish*
(Unlu, 2026) states this explicitly: "these literatures usually test one boundary at a time…
for practical agent deployment, **binary abstention is insufficient**." That is the gap the
hypothesis targets.

---

## 2. Key papers

### 2.1 The decomposition that everything else builds on

#### Clarify When Necessary: Resolving Ambiguity Through Interaction with LMs
- **Authors**: Michael J.Q. Zhang, Eunsol Choi (UT Austin) · **Year**: 2023 · arXiv 2311.09469 · 110 citations
- **Contribution**: Task-agnostic three-subtask framework — (1) *when* to clarify, (2) *what* to
  ask, (3) how to *use* the answer — plus INTENT-SIM, an uncertainty estimator.
- **Method (INTENT-SIM)**: greedily generate a clarifying question `q` from input `x`; sample
  S=10 user answers at T=0.5; cluster semantically equivalent answers with a DeBERTa-large MNLI
  model (bidirectional entailment); take the entropy of the resulting cluster distribution.
- **Crucial conceptual move**: the task requires *disentangling aleatoric from epistemic
  uncertainty*. You should ask when aleatoric uncertainty is high (intent is genuinely ambiguous)
  **and** epistemic uncertainty is low (you'll know the answer once told). Plain output entropy
  conflates the two, which is precisely why it underperforms.
- **Datasets**: AmbigQA (QA), AmbiEnt (NLI), DiscourseMT (MT).
- **Metrics**: **AUROC** for identifying which items improve with clarification, and **performance
  under a fixed interaction budget** `b ∈ {10, 20, 30}%` — ask on the top-b% most uncertain items.
- **Results**: modest but real. QA/GPT-3 AUROC 0.628 (INTENT-SIM) vs 0.590 (likelihood); at
  b=30% it captures 49% of the total achievable gain vs 30% for random. On MT/GPT-3, however,
  INTENT-SIM's AUROC is 0.512 and Self-Ask scores 0.371 — **worse than chance**.
- **Relevance**: supplies the experimental protocol (budget curves + AUROC) and the aleatoric/
  epistemic framing that the hypothesis's "estimating its uncertainty" clause needs. Also a
  caution: absolute AUROCs cluster in the 0.50–0.63 range, so this problem is genuinely hard and
  effect sizes are small.

#### Know Your Limits: A Survey of Abstention in LLMs
- **Authors**: Wen et al. · **Year**: 2024 · TACL · arXiv 2407.18418 · 128 citations
- Organizes abstention along three axes — **query**, **model**, **human values** — and argues
  abstention should be studied as a *meta-capability* that transcends specific tasks. That framing
  matters here: the hypothesis claims one policy should generalize across stakes and domains.

### 2.2 Papers that already propose the multi-way action space

These are the closest prior work to the hypothesis and must be treated as the real baselines.

#### Don't Start What You Can't Finish: A Counterfactual Audit of Support-State Triage
- **Author**: Eren Unlu · **Year**: 2026 · arXiv 2604.16752
- **Ontology (four support states)**, defined by *minimal repair distance* from solvability over a
  task tuple `(u, e, z, κ)` = (user request, in-context evidence, available support/tools,
  environmental contract):
  | State | Condition | Gold action |
  |---|---|---|
  | Complete | `solvable(u,e,z,κ)` | **ANSWER** |
  | Clarifiable | ∃ one clarification `c` s.t. `solvable(u⊕c, …)` | **CLARIFY** |
  | Support-Blocked | ∃ one support grant `s` s.t. `solvable(u,e,z⊕s,κ)` | **REQUEST SUPPORT** |
  | Unsupported-Now | neither single repair suffices | **ABSTAIN** |
- **Dataset**: SSTA-32 — 32 matched items across 8 task families, built by **minimal counterfactual
  edits** that flip the same base request across all four states. 3-pass adversarial validation.
- **Results**: default execution overcommits on 41.7% of non-complete tasks. Scalar confidence
  prompting avoids overcommitment but **collapses the three-way deferral space** (58.3% typed
  deferral accuracy). Surfacing the categorical ontology in the prompt (Action-Only or the typed
  Preflight Support Check) reaches **91.7%** typed accuracy and cuts overcommitment to 8.3%.
  Ablations: removing the support-sufficiency dimension selectively degrades REQUEST SUPPORT;
  removing evidence-sufficiency triggers systematic overcommitment.
- **Relevance**: this is nearly the hypothesis, already tested. **Its own stated limitation is the
  opening**: n=32, single frontier model, and Dual-Persona Auto-Auditing runs "within a single
  context window," which the author concedes yields **upper-bound capability estimates**. A
  larger-n, multi-model, properly-isolated replication is a well-motivated contribution.
- **Headline result to build on**: *scalar confidence is insufficient* — a single uncertainty
  number cannot select among three distinct deferral types. This directly qualifies the
  hypothesis's "estimating its uncertainty" clause.

#### Clarify, Abstain or Answer? Strategising in Conversation with Belief-Augmented Generation
- **Authors**: Baan, Aziz, Plank, Fernández (UvA/LMU) · **Year**: 2026 · arXiv 2605.25831
- **Method (BAG)**: sample K responses to form a textual *belief state*, put it back in the prompt,
  and let the model reason over its own samples to pick answer / clarify / abstain. Training-free,
  works on closed models.
- **Results across six LLMs**: models *by default rarely clarify or abstain*. BAG improves QA
  accuracy and yields strategy choices more faithful to the belief state than prompt-only
  baselines. **But: "disentangling when to clarify from when to abstain remains challenging."**
- **Relevance**: the strongest single-paper precedent, and it names the unsolved sub-problem —
  clarify-vs-abstain separation — as its own negative result. Any experiment here should measure
  that separation explicitly rather than reporting aggregate 3-way accuracy, which hides it.

#### SAFETY SENTRY: Context-Aware Human Intervention via EXECUTE-ASK-REFUSE Routing
- **Year**: 2026 · arXiv 2607.13594
- Reframes binary guard models as **per-instance three-way routing** over {EXECUTE, ASK, REFUSE},
  arguing the binary view conflates *is the action harmful* with *is it appropriate given context*.
  A lightweight guard model with a **single decoding-time threshold** repositions one checkpoint
  across deployments of differing risk tolerance without retraining.
- **Relevance**: the cleanest existing mechanism for the "depending on the stakes" clause — a
  tunable threshold rather than a retrained policy per risk level.

### 2.3 The recognition–behavior gap (the most actionable finding in the corpus)

#### Knowing but Not Showing: LLMs Recognize Ambiguity but Rarely Ask Clarifying Questions
- **Authors**: Jinyan Su, Claire Cardie (Cornell) · **Year**: 2026 · arXiv 2605.25284
- **Design**: three settings over ambiguous / unambiguous / disambiguated questions — standard QA,
  explicit ambiguity judgment, and behavioral analysis where a judge classifies responses as
  direct answer / refusal / clarifying question.
- **Finding**: a clear gap. Models identify ambiguity when *explicitly asked to judge it*, yet in
  the QA setting they overwhelmingly default to direct answers. **Retrieved context widens the
  gap** — it improves answerability while making models *less* likely to ask.
- **Relevance**: this splits the hypothesis into two claims that must be measured separately.
  "Can recognize when it lacks information" appears **largely true already**; "chooses among
  acting/asking/refusing/deferring" is where models fail. An experiment that only measures
  recognition will confirm the hypothesis trivially and misleadingly.

#### HiL-Bench: Do Agents Know When to Ask for Help?
- **Authors**: Trinh, Elfeki et al. (Scale AI) · **Year**: 2026 · arXiv 2604.09408
- **Design**: tasks from SWE-Bench Pro and BIRD, modified to contain human-validated **blockers**
  (missing info, ambiguous requests, contradictory info) that surface only through **progressive
  discovery** — via execution and exploration, not upfront inspection. This deliberately defeats
  the degenerate strategy of front-loading all questions.
- **Metric**: **ASK-F1** = harmonic mean of question *precision* and blocker *recall*. The harmonic
  structure architecturally prevents gaming by question spam.
- **Results**: frontier models score 75–89% pass@3 with full information but only **4–24%** when
  they must judge whether to ask — despite having an `ask_human()` tool available. Near-zero "No
  Tool" bars confirm the tasks genuinely require clarification, so **the bottleneck is judgment,
  not capability**. Three failure profiles: overconfident wrong beliefs with no gap detection;
  high uncertainty detection yet persistent errors; broad imprecise escalation without
  self-correction. No model lands in the well-judged quadrant.
- **Trainability**: RLVR on shaped ASK-F1 reward moves a 32B model toward calibrated help-seeking,
  and **gains transfer across domains** — the model learns to detect unresolvable uncertainty
  rather than domain-specific heuristics.
- **Relevance**: the strongest direct evidence *for* the "can be trained" half of the hypothesis,
  and the strongest evidence that the current default behavior is bad. Also supplies ASK-F1, the
  best available metric for balancing over- and under-asking.

### 2.4 Uncertainty estimation methods applicable to the decision

| Paper | Year | Mechanism | Note for this hypothesis |
|---|---|---|---|
| INTENT-SIM (2311.09469) | 2023 | Entropy over *simulated user intents* | Targets aleatoric uncertainty specifically |
| Semantic Entropy (Kuhn et al., via 2410.17234) | 2024 | Entropy over NLI-clustered outputs | Standard strong baseline; conflates uncertainty types |
| Uncertainty Decomposition (2606.19559) | 2026 | Prompt-based split of **action confidence** vs **request uncertainty** | Black-box friendly; +73% clarification F1 over ReAct+UE on ALFWorld-Clarification |
| Structured Uncertainty (2511.08798) | 2025 | EVPI-based structured uncertainty for tool-calling | Decision-theoretic, expected value of information |
| BAG (2605.25831) | 2026 | Reason over own K-sample belief state | Training-free, closed-model compatible |
| I-CALM (2604.03904) | 2026 | Verbal confidence + **announced answer/abstain payoffs** + normative guidance | Directly operationalizes *stakes* in the prompt |
| Conformal Abstention (2405.01563) | 2024 | Conformal prediction over self-consistency | Distribution-free risk guarantees |
| Entropy Alone is Insufficient (2603.21172) | 2026 | Negative result on entropy-based selective prediction | Cautionary: don't rely on a single scalar |

**I-CALM deserves emphasis** for the stakes clause: it announces the answer/abstain *payoff matrix*
in the prompt and uses a **two-stage protocol** — the model first chooses answer-or-abstain, then is
*forced* to give a best guess on the abstained items. That second stage separates targeted
abstention from indiscriminate refusal, which single-stage accuracy cannot do. It is the cleanest
mechanism in the corpus for manipulating stakes without retraining.

### 2.5 Agentic and code-generation settings

- **Learning to Ask / AwN** (2409.00557): Noisy ToolBench. Finds that because of next-token
  training, agents **arbitrarily fabricate missing arguments** rather than asking. Ask-when-Needed
  prompting substantially outperforms standard tool-learning frameworks.
- **LHAW** (2602.10525): controllable underspecification for long-horizon tasks. Removes info along
  four dimensions — **Goals, Constraints, Inputs, Context** — at configurable severity. Critically,
  it **validates variants by empirical agent trials** rather than by asking an LLM to predict
  ambiguity, classifying them as *outcome-critical*, *divergent*, or *benign* based on observed
  terminal-state divergence. 285 variants over TheAgentCompany, SWE-Bench Pro, MCP-Atlas.
- **Ask or Assume** (2603.26233): decouples underspecification detection from execution in coding
  agents; ships 1,699 human-ensembled SWE-bench underspecification annotations.
- **HumanEvalComm** (2406.00215): 164 HumanEval problems perturbed into ambiguous / inconsistent /
  incomplete / combined variants, with executable tests — the cost of a wrong assumption is
  *measured by running code*, not judged.
- **CollabLLM** (2502.00640): multiturn-aware rewards via collaborative simulation; +18.5% task
  performance, +46.3% interactivity, and in a 201-judge user study, +17.6% satisfaction with
  **10.4% less user time** — evidence that better asking need not cost more interaction.

### 2.6 Benchmarks and evaluation frameworks

- **CLAMBER** (2405.12063): 3,202 items, perfectly balanced ask/don't-ask, with a taxonomy.
- **QuestBench**: underspecified reasoning where sufficiency is **formally verifiable** (CSP/logic/
  planning), so the "should it ask" label is not annotator-noisy. Models solve fully-specified
  versions yet fail to identify the missing question (40–50% on logic).
- **RegretBench** (2607.21143): evaluates clarification as a **policy** under hidden intent, scored
  by **regret** vs a reference policy, decomposed into intent resolution, interaction cost,
  ineffective clarification, and regret. Argues final success alone is insufficient — models with
  equal accuracy differ sharply in efficiency and stopping behavior.
- **AbstentionBench** (cloned; 20 datasets, 6 scenarios): headline finding that **reasoning
  fine-tuning *hurts* abstention** (~24% average degradation). Reasoning models respond
  overconfidently and rarely abstain.

---

## 3. Common methodologies

**A. Sampling-based uncertainty** (dominant). Sample K outputs → cluster by semantic equivalence
(bidirectional NLI entailment) → compute entropy. Variants differ in *what* is sampled: outputs
(semantic entropy), user intents (INTENT-SIM), or belief states reasoned over in-prompt (BAG).

**B. Prompt-level decision scaffolds** (cheapest, surprisingly strong). Surface the action ontology
explicitly in the prompt. SSTA-32's finding is striking: typed categorical prompting alone moved
deferral accuracy from 58.3% → 91.7%. Related: Ask-when-Needed, I-CALM's payoff announcement,
preflight support checks.

**C. Training-based**. R-Tuning (refusal-aware instruction tuning), Refusal Tokens, RLVR on shaped
ASK-F1 (HiL-Bench), ACT contrastive self-training (2406.00222), collaborative self-play
(2512.04068), Abstain-R1 (verifiable RL).

**D. Conformal / risk-controlled**. Conformal abstention, conformal alignment, learned conformal
abstention policies — provide distribution-free guarantees at a chosen risk level, which is the
most principled route to the "depending on the stakes" clause.

---

## 4. Standard baselines

Ordered by how essential each is:

1. **Direct / vanilla prompting** — no ask option. Establishes the silent-guessing floor.
2. **Always ask** — the trivial ceiling on recall and floor on precision; required, because
   ASK-F1-style metrics exist specifically to catch it.
3. **Sequence likelihood / p(True)** — cheapest scalar uncertainty baseline.
4. **Semantic entropy** (Kuhn et al.) — the standard strong unsupervised baseline.
5. **Self-Ask / verbalized confidence** — prompt the model to judge whether a follow-up is needed.
   *Note*: Self-Ask scored 0.371 AUROC on MT in Zhang & Choi — below chance. Do not assume it's
   a floor.
6. **INTENT-SIM** — the ambiguity-targeted method to beat.
7. **BAG** — belief-state reasoning; the closest 3-way baseline.
8. **Typed categorical prompting (PSC)** — the strongest known cheap intervention for 4-way.

---

## 5. Evaluation metrics

| Metric | What it captures | Source |
|---|---|---|
| **AUROC** (does clarification help?) | Ranking quality of the uncertainty estimate | 2311.09469 |
| **Performance under interaction budget** `b%` | Practical value at a fixed asking rate | 2311.09469 |
| **ASK-F1** = HM(question precision, blocker recall) | Over- vs under-asking balance; spam-resistant | 2604.09408 |
| **Typed deferral accuracy** | Whether the *right kind* of deferral was chosen | 2604.16752 |
| **Overcommitment rate** | Fraction of non-complete tasks acted on anyway | 2604.16752 |
| **Regret vs reference policy** | Multi-turn policy quality, not per-turn plausibility | 2607.21143 |
| **Contrast-set compliance** | Over-refusal on superficially-similar valid requests | CoCoNot |
| **Two-stage abstention quality** | Targeted vs indiscriminate abstention | 2604.03904 |
| **Execution pass rate** | Cost of a wrong assumption, objectively | 2406.00215 |
| **ECE / calibration** | Whether stated confidence tracks correctness | calibration papers |

**Recommendation**: report ASK-F1 (or typed deferral accuracy) *jointly* with contrast-set
compliance. Neither alone is meaningful — one rewards asking, the other punishes over-asking, and
degenerate policies score well on exactly one.

---

## 6. Gaps and opportunities

**G1 — The recognition/behavior gap is measured but not explained.** *Knowing but Not Showing*
establishes that models recognize ambiguity yet act as if they didn't. Nobody has isolated *why*.
Candidate causes (RLHF helpfulness pressure, next-token argument fabrication, absence of an
explicit ask-affordance) are hypothesized but untested. Note that this gap makes a naive reading of
the hypothesis nearly vacuous — recognition already works; the failure is downstream.

**G2 — Clarify vs. abstain vs. defer is unresolved.** BAG names it as an open problem; SSTA-32
shows scalar confidence collapses the distinction. **Nobody has shown that a single scalar
uncertainty estimate can select among three deferral types** — and there is direct evidence it
cannot. This is the sharpest, most falsifiable target in the whole corpus, and it puts real
pressure on the hypothesis as stated ("by estimating its uncertainty" — singular).

**G3 — Stakes are asserted, rarely manipulated.** Almost every paper *motivates* with stakes
("in high-stakes settings we may want systems to frequently ask"), but only I-CALM (announced
payoffs) and SAFETY SENTRY (threshold repositioning) actually manipulate them experimentally.
No paper in the corpus reports a clean **stakes × ambiguity factorial**. Yet the ingredients exist:
IN3 importance ratings (1–3), SWE-bench difficulty × underspecification, LHAW severity levels.

**G4 — SSTA-32 is n=32, one model, and self-admittedly upper-bound.** Dual-Persona Auto-Auditing
runs inside a single context window, so the audited model sees its own audit. A larger, multi-model,
context-isolated replication is straightforwardly valuable and the author invites it.

**G5 — Reasoning models are worse at abstention.** AbstentionBench finds ~24% degradation from
reasoning fine-tuning. Whether this extends to *asking* (as opposed to abstaining) is untested, and
matters a lot given where the field is heading.

**G6 — Retrieval makes asking worse.** *Knowing but Not Showing* finds retrieved context widens
the recognition–behavior gap. Essentially unexplored, and directly relevant to deployed RAG
assistants.

---

## 7. Recommendations for our experiment

### Recommended datasets

| Priority | Dataset | Why |
|---|---|---|
| 1 | **CoCoNot** (original + contrast) | Only dataset whose native categories span the full action space *and* ships an over-refusal control |
| 2 | **CLAMBER** | Balanced 1,601/1,601, taxonomy-labeled, ask-vs-answer |
| 3 | **IN3** | Importance ratings 1–3 = the stakes axis, already annotated |
| 4 | **QuestBench** | Formally verified sufficiency — non-noisy labels |
| 5 | **SWE-bench underspec.** | Real-task difficulty × underspecification grid |
| 6 | AmbigQA / AmbiEnt | Replicate INTENT-SIM; aleatoric/epistemic separation |
| 7 | ClariQ | Graded need 1–4 → calibration, not just accuracy |
| 8 | HumanEvalComm | Executable cost of not asking |

### Recommended baselines
Direct prompting; always-ask; sequence likelihood; semantic entropy; verbalized confidence;
INTENT-SIM; BAG; typed categorical prompting (PSC). The last two are the real competition.

### Recommended metrics
Primary: ASK-F1 and typed deferral accuracy. Mandatory companion: contrast-set compliance
(over-refusal). Secondary: overcommitment rate, budget curves at b ∈ {10, 20, 30}%, AUROC, ECE.

### Methodological cautions

1. **Do not report aggregate 3-/4-way accuracy alone.** It hides the clarify-vs-abstain confusion
   that is the actual open problem (G2). Report the confusion matrix.
2. **Always include an always-ask baseline.** Many reported gains are recoverable by asking more.
3. **Expect small effect sizes.** Published AUROCs sit at 0.50–0.63; some methods score below
   chance on some tasks. Power the design accordingly and pre-commit to what counts as a result.
4. **Watch for degenerate over-asking.** ASK-F1's harmonic mean and CoCoNot's contrast set are the
   two defenses; use both.
5. **Isolate contexts when auditing.** SSTA-32's single-context-window design is why its numbers
   are upper bounds — don't inherit that flaw.
6. **Separate recognition from behavior** (G1). Measure "does it know?" and "does it act on it?"
   as distinct quantities, or the hypothesis confirms trivially.
7. **Beware LLM-as-judge bias**; SSTA-32 deliberately used deterministic heuristic scoring, and
   AbstentionBench ships human-validated judge prompts worth reusing for comparability.

### Where the hypothesis is most likely to be *partially falsified*

Worth stating up front, since a well-designed experiment should be able to find this: the evidence
suggests the hypothesis is **true for the "prompted to recognize" clause and doubtful for the
"by estimating its uncertainty" clause**. Scalar uncertainty demonstrably fails to separate the
deferral types (SSTA-32), models already recognize ambiguity without acting on it (Su & Cardie),
and what actually works is *surfacing a typed decision ontology* — a structural intervention, not
an uncertainty-estimation one. A result showing that categorical scaffolding beats better
uncertainty estimation would be a genuine, publishable finding, and it is where the corpus points.

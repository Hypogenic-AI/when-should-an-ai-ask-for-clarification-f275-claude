# When should an AI ask for clarification?

**A four-way action-routing benchmark (CARB), four experiments on whether models can tell when
they lack the information to act safely, and a decision-theoretic evaluation of whether any of it
is worth deploying.**

Research phase: `experiment_runner` · Workspace: `when-should-an-ai-ask-for-clarification-f275`
· All numbers in this report are reproduced in `results/SUMMARY.md` from the JSON artefacts that
produced them.

---

## 1. Executive summary

**Question.** Can an AI recognise when it lacks enough information to act safely, and choose
among *acting*, *asking*, *refusing*, and *deferring* according to the ambiguity and the stakes?

**Answer, in one sentence.** Partly — the models can be *prompted into* a usable four-way
routing policy (accuracy 0.60–0.72 against 0.23 uniform-random and 0.29 best-single-action, with
over-commitment falling from 45–58% to 9–25%), but the mechanism the hypothesis names — *estimating its uncertainty* — is the part that
fails: a single verbalised confidence, given the best router that can be fitted to it, reaches
only 0.42–0.45 and is at chance at telling apart the three reasons for not acting, while
announcing high stakes moves the asking *threshold* without improving the *discrimination* at
all.

**Four findings that matter.**

1. **Structure beats uncertainty.** Surfacing a typed decision ontology (R2) beats routing a
   scalar confidence (R3) by 15–28 accuracy points on identical items, on every model
   (McNemar, all *p*<sub>Holm</sub> < 1e-8, Cohen's *h* = 0.32–0.57). The scalar separates
   "something is wrong" reasonably well (AUROC 0.79–0.87 for ACT vs not-ACT) but is at chance
   at separating *why* (DEFER vs REFUSE AUROC 0.44–0.56). A hybrid that uses the scalar only
   for the gate it is good at and the typed judgment for the kind never beats typed prompting,
   so the scalar is not merely mis-routed — it is redundant.

2. **The recognition–behaviour gap is real but is not the whole story.** On the same items, the
   action implied by the models' own property judgments (R4) beats what they actually do under a
   plain prompt (R0) for all four models, and cuts over-commitment from 45–58% to 11–28%. But
   recognition itself tops out at 0.55–0.61, and the property that fails is specifically
   *"is information sufficient?"* (56.5% on items that needed a question, versus 89% correct on
   *"is this safe?"* and 94% on *"is this within my capability?"*). Models are good at noticing
   danger and their own limits, and bad at noticing that they were not told enough.

3. **Stakes move the threshold, not the judgment.** Announcing high stakes raises the ask-rate on
   items that needed a question (0.29 → 0.43 typed, 0.35 → 0.56 free-text, paired HIGH vs LOW) —
   but it raises the false-alarm rate on genuinely answerable items just as much or more
   (0.05 → 0.11 typed, 0.06 → 0.28 free-text). In
   signal-detection terms, a paired bootstrap of the change gives Δcriterion = −0.44
   [−0.69, −0.27] typed (*p* < 0.001) with Δ*d′* = −0.11 [−0.62, +0.23] (*p* = 0.53), and in the
   free-text regime discrimination gets **significantly worse**: Δ*d′* = −0.58 [−1.21, −0.11],
   *p* = 0.013. Telling a model the stakes are high makes it more anxious, not wiser. The
   ambiguity × stakes interaction is exactly null (β = +0.010, *p* = 0.97).

4. **Training helps a lot, in-distribution, and not at all out of it.** LoRA SFT on 2,576
   prompt→action pairs takes an open 4B model from 0.577 (prompted with the same ontology) to
   **0.794** and cuts over-commitment to 1.4% — but on an out-of-source split it falls to 0.470,
   *below* the un-trained prompted baseline. A frozen linear probe on the same base model's hidden
   state gets 0.779 without any training at all, and fails out-of-source the same way. The routing
   signal is already in the representation; what training adds is a read-out, and the read-out is
   partly source-specific.

**Practical implication.** Once errors are priced, every frontier model's prompted policy is
worth *less than always asking* as soon as an unwarranted action costs about 5–10× a needless
refusal — and worth less than nothing by 10×.

---

## 2. Research question, hypothesis and what would falsify it

**Hypothesis as given.** *An AI can be trained or prompted to recognise when it lacks sufficient
information to act safely, by estimating its uncertainty and choosing among acting, asking for
clarification, refusing, or deferring, depending on the stakes and ambiguity of the task.*

This has five separable claims, and the experiments were designed so that each can fail alone:

| Claim | Experiment | Falsified if |
|---|---|---|
| C1 it can be **prompted** into the four-way choice | E2 (regimes) | prompted routing ≈ chance or ≈ degenerate baselines |
| C2 it does so **by estimating uncertainty** | E2 (R3 vs R2), hybrid ablation | a scalar confidence, optimally routed, loses badly to typed prompting |
| C3 the failure is **recognition**, not behaviour | E1 (R4 vs R0/R1) | recognition is no better than behaviour, or is itself poor |
| C4 the threshold moves with **stakes** | E3 (announced + item-level) | stakes shift the criterion without improving discrimination |
| C5 it can be **trained** | E4 (LoRA SFT, probe) | training does not beat prompting, or does not transfer |

The literature review (`literature_review.md`) predicted C2 and C3 would be where the hypothesis
strains: SSTA-32 (arXiv 2604.16752, *n*=32) found scalar confidence collapses the deferral space,
BAG (2605.25831) reported clarify-vs-abstain separation "remains challenging", and *Knowing but
Not Showing* (2605.25284) found models recognise ambiguity yet answer anyway. This study is a
larger, multi-model, context-isolated test of exactly those claims.

---

## 3. The benchmark: CARB v1

Existing benchmarks are binary (answer vs abstain, or ask vs don't-ask). The hypothesis needs
four actions, because *asking*, *refusing* and *deferring* are three different things that a
binary benchmark scores identically. CARB assembles them from sources that already carry the
distinction.

| Action | Meaning | Source annotation |
|---|---|---|
| **ACT** | well specified, safe, within capability | CoCoNot *contrast* set; CLAMBER `require_clarification=0`; IN3 `vague=false` |
| **ASK** | missing/ambiguous, and the **user** can fix it in one turn | CoCoNot *Incomplete requests*; CLAMBER `=1`; IN3 `vague=true` |
| **REFUSE** | should not be done at all — harmful, or no determinate answer exists | CoCoNot *safety concerns*, *humanizing*, *indeterminate* |
| **DEFER** | the **model** lacks the capability/modality/access; hand off | CoCoNot *Unsupported requests* |

**Gold labels are never invented.** Every item inherits its label from its source dataset's own
annotation through a fixed table printed at build time (`datasets/carb/label_mapping.json`). Two
cells are flagged `contested` and every headline number is recomputed without them, under an
alternative mapping, and under a capability re-audit (§7.4).

- **840 items**: dev 160 (threshold tuning only), **test 480** (all headline numbers),
  transfer 200 (CLAMBER, a source never used for tuning or training).
- Sampling is balanced by gold action at the level of the 640-item core pool (280 ASK / 280 ACT /
  140 REFUSE / 140 DEFER); the dev/test cut is random, so the test split is 137 ASK / 129 ACT /
  112 REFUSE / 102 DEFER. Reference points on it: uniform-random **0.231**, best single-action
  policy (always-ASK) **0.285**.
- Training data for E4 (2,800 items) is drawn from CoCoNot's *train* split and AmbigQA;
  prompt-level overlap with test is **0/479**, with transfer 3/200.

**Elicitation regimes** (the independent variable; every model saw every item under all five, so
all comparisons are within-item):

| | prompt | measures |
|---|---|---|
| **R0 DIRECT** | the bare request | default behaviour |
| **R1 AFFORDANCE** | + "you may answer, ask, decline, or explain a limitation" | behaviour with an ask-affordance |
| **R2 TYPED** | full four-action ontology + ordered decision procedure, JSON out | typed categorical routing |
| **R3 SCALAR** | "how confident are you that you can safely do this now, 0–100?" | uncertainty estimation |
| **R4 RECOGNITION** | four yes/no property judgments; never asked what to do | recognition, isolated from behaviour |

R3's number is turned into an action by the *best* 3-cut × 24-permutation router fittable to it,
under 5-fold cross-validation (no item is predicted by a router that saw it) — the scalar baseline
is given every advantage. R4's booleans are converted by a fixed rule structurally identical to
the procedure R2 hands the model, so R2-vs-R4 isolates *who applies the rule*.

Full prompt text: `src/carb/prompts.py`.

---

## 4. Setup

**Models.** `openai/gpt-5`, `anthropic/claude-sonnet-4.5`, `google/gemini-2.5-flash`,
`meta-llama/llama-3.3-70b-instruct` via OpenRouter at `temperature=0`; `Qwen/Qwen3-4B` locally
(bf16, greedy, `enable_thinking=False`) for the training and probing arm. 7,668 API calls,
$50.79 total. Local compute: 4 × NVIDIA RTX A6000 (48 GB).

**Judge.** Free-text responses (R0, R1) are classified into the four actions by a judge whose
prompt extends AbstentionBench's human-validated binary abstention judge. The planned judge
(`openai/gpt-4.1`) could not run — the API budget was exhausted *after* all raw completions were
collected but *before* judging, so every API judge call returned HTTP 403. The judge was
re-implemented locally on `Qwen/Qwen3-14B` by **constrained choice**: the judge prompt is
completed with `{"behaviour": "` and the four action words are compared at that slot in one
forward pass, which makes format errors impossible and the decision deterministic.

Judge reliability, measured two ways:

| check | n | raw agreement | Cohen's κ |
|---|---|---|---|
| Qwen3-14B (primary) vs Qwen3-4B (independent second judge) | 3,768 | 0.86 | **0.78** |
| Qwen3-14B vs blind annotation of a stratified sample | 80 | 0.81 | **0.75** |

The 80 blind annotations were produced by the orchestrating research agent (Claude Opus 5) from
request + response text only, with no access to the judge's label, the gold label, or the item's
source. That is *not* human validation; it is a third model agreeing, and should be read as the
weaker evidence it is. Both κ values fall in the conventional "substantial agreement" band, and
the two checks agree with each other, which is the main reason to trust the R0/R1 numbers at all.

**Statistics.** Within-item comparisons → **McNemar's exact test**, **Holm-corrected** across the
pre-registered family of 24 comparisons. CIs are item-level percentile bootstraps (2,000
resamples). Effect sizes are Cohen's *h*. Seed 42 everywhere.

---

## 5. Results

### 5.0 The benchmark is not trivially solvable — and not free of shortcuts either

| policy | accuracy | macro-F1 | ASK-F1 | over-commit | contrast compliance |
|---|---|---|---|---|---|
| always ACT | 0.269 | 0.106 | 0.000 | 1.000 | 1.000 |
| always ASK (= majority class) | 0.285 | 0.111 | 0.444 | 0.000 | 0.000 |
| always REFUSE | 0.233 | 0.095 | 0.000 | 0.000 | 0.000 |
| uniform random | 0.231 | 0.230 | 0.243 | 0.254 | 0.240 |
| **TF-IDF + logistic regression** (trained on a disjoint CoCoNot pool) | **0.702** | 0.701 | 0.613 | 0.077 | 0.612 |
| TF-IDF, **transfer split** | 0.355 | 0.196 | 0.574 | 0.110 | 0.130 |

Two things follow, and both are reported rather than buried. First, the degenerate policies are
all near chance, so ASK-F1 and contrast compliance are doing their job: no single-action policy
scores well on both. Second, **a bag-of-words classifier gets 0.702 on the test split — higher
than every frontier model's typed routing except Claude's 0.721.** CoCoNot's categories carry
strong lexical signatures ("every", "all ... that exists", "current", "I cannot"), and any
absolute accuracy on this split is inflated by them. The same classifier collapses to 0.355 on the
out-of-source transfer split. **Absolute accuracies on the test split should therefore be read as
within-benchmark comparisons between regimes on identical items, never as capability estimates.**
This is why every headline claim in this report is a *within-item, between-regime* contrast rather
than a level.

### 5.1 Experiment 1 — recognition vs behaviour

![recognition vs behaviour](figures/fig2_recognition_behaviour_gap.png)

| model | R0 direct | R1 affordance | R4 recognition | R2 typed |
|---|---|---|---|---|
| GPT-5 | 0.521 | 0.465 | 0.552 | 0.598 |
| Claude Sonnet 4.5 | 0.535 | 0.565 | 0.608 | **0.721** |
| Gemini 2.5 Flash | 0.515 | 0.594 | 0.558 | 0.652 |
| Llama 3.3 70B | 0.492 | 0.546 | 0.613 | 0.610 |
| *over-commitment* | *0.45–0.58* | *0.32–0.45* | *0.11–0.28* | *0.09–0.25* |

Recognition beats default behaviour for all four models (+0.03 to +0.12 accuracy), significantly
so for Llama (*p*<sub>Holm</sub> = 9.7e-4, *h* = +0.24) and marginally for Claude
(*p*<sub>Holm</sub> = 0.060). The effect is far clearer on the metric the gap is actually about:
**under a plain prompt the models act on 45–58% of requests they should have withheld action on;
their own property judgments, applied through a fixed rule, would have acted on only 11–28%.**

Adding a bare ask-affordance (R1) helps Gemini (+0.079, *p*<sub>Holm</sub> = 1.4e-4) and Llama
(+0.054, *p*<sub>Holm</sub> = 0.031), does nothing for Claude (+0.030, n.s.), and appears to
**hurt GPT-5** (−0.056, *p*<sub>Holm</sub> = 0.008). That last one does not survive scrutiny:
GPT-5 returned an empty completion on 53/480 R1 items (reasoning tokens exhausting `max_tokens`),
which the headline scoring counts as errors; restricted to items where both regimes produced a
response the gap shrinks to −0.023 (*n* = 424, *p* = 0.15, uncorrected) — **so the honest reading
is that the affordance helps two models and does nothing measurable for the other two.** Naming
the options is not enough; the ontology has to be defined.

**Where recognition itself fails.** Scoring each R4 property only on items where the fixed rule
reads it (`src/carb/recognition.py`):

| gold action → property | pooled accuracy | n |
|---|---|---|
| REFUSE → `safe_and_determinate` | 0.893 | 439 |
| ACT → `within_capability` | 0.943 | 509 |
| ACT → `safe_and_determinate` | 0.957 | 510 |
| ACT → `information_sufficient` | 0.782 | 510 |
| ASK → `within_capability` | 0.850 | 545 |
| ASK → `safe_and_determinate` | 0.754 | 545 |
| DEFER → `within_capability` | 0.741 | 401 |
| DEFER → `safe_and_determinate` | 0.633 | 403 |
| **ASK → `user_can_resolve`** | **0.574** | 545 |
| **ASK → `information_sufficient`** | **0.565** | 545 |

This is the sharpest disconfirmation of a naive reading of the hypothesis. Models are excellent at
"is this dangerous?" and "can I even do this?" and close to a coin flip at "was I told enough?" —
which is precisely the recognition the hypothesis is about. The DEFER row is the second story: on
requests blocked by the model's own limits, models say the request is *not determinate* 37% of the
time, i.e. they misattribute their own limitation to the world.

### 5.2 Experiment 2 — typed ontology vs scalar uncertainty

![regime accuracy](figures/fig1_regime_accuracy.png)

| comparison | GPT-5 | Claude | Gemini | Llama |
|---|---|---|---|---|
| R2 typed − R3 scalar (accuracy) | +0.181 | +0.275 | +0.225 | +0.158 |
| Cohen's *h* | +0.36 | +0.57 | +0.46 | +0.32 |
| *p*<sub>Holm</sub> | 5.2e-12 | 1.6e-23 | 8.6e-19 | 7.7e-9 |
| typed-deferral accuracy, typed vs scalar | 0.748 / 0.403 | 0.764 / 0.402 | 0.709 / 0.374 | 0.650 / 0.383 |

Typed prompting beats optimally-routed scalar confidence on every model, by a wide and uniformly
significant margin. The diagnostic explains why:

| model | AUROC ACT vs not-ACT | mean confidence ACT / ASK / REFUSE / DEFER | AUROC ASK vs REFUSE | AUROC ASK vs DEFER | **AUROC DEFER vs REFUSE** |
|---|---|---|---|---|---|
| GPT-5 | 0.814 | 83.3 / 50.8 / 23.6 / 11.0 | 0.721 | 0.780 | **0.507** |
| Claude Sonnet 4.5 | 0.824 | 82.7 / 45.2 / 18.9 / 6.8 | 0.669 | 0.741 | **0.436** |
| Gemini 2.5 Flash | 0.788 | 77.8 / 38.3 / 6.6 / 14.0 | 0.663 | 0.622 | **0.543** |
| Llama 3.3 70B | 0.873 | 84.0 / 34.5 / 9.6 / 19.5 | 0.652 | 0.596 | **0.563** |

The scalar carries real information about *whether* to act (AUROC 0.79–0.87) and essentially none
about *which* kind of withholding is right — DEFER vs REFUSE is at chance for all four models, and
*below* chance for Claude. This is the geometric reason the hypothesis's "by estimating its
uncertainty" clause cannot work as stated: three distinct deferral types cannot be recovered from
an ordering of one number, and empirically they are not even ordered consistently (Gemini and
Llama place DEFER *above* REFUSE; GPT-5 and Claude below).

**Is the scalar at least complementary?** A hybrid that uses the cross-validated scalar for the
ACT gate (its strength) and the typed judgment for the kind (its weakness) never beats plain
typed prompting; for Claude it is significantly *worse* (0.654 vs 0.721, *p*<sub>Holm</sub> =
1.1e-4). Same for a recognition-based stage 2. The scalar adds nothing that the typed judgment
does not already contain.

**Calibration and budgets.** The verbalised confidence is poorly calibrated as a forecast of
"this item really was ACT" (ECE 0.16–0.21, mean stated confidence 0.36–0.45 against a 0.27 base
rate — systematically over-confident). It is also barely a scale: **only 3–31% of the mass falls
strictly between 0.05 and 0.95** (Gemini 3%, Claude 17%, GPT-5 18%, Llama 31%). Asked for a
number from 0 to 100, the models answer 0 or 100. That is a second, independent reason the
"estimate your uncertainty" route cannot support a three-way deferral decision: there is almost
no resolution in the estimate to threshold.

Ranking by 1−confidence and asking on the *b*% least
confident recovers roughly twice random ask-recall (0.33–0.39 at a 20% budget vs 0.20 random),
which is real but weak; the typed policies sit at similar recall with much better precision
(Claude: ask-rate 0.24 → recall 0.41 at precision 0.89).

![calibration and budget](figures/fig9_budget_calibration.png)

### 5.3 Experiment 3 — does the threshold move with the stakes?

Each of 240 items (60 per gold action) was shown to each model under three announced-stakes
frames prepended to the system prompt (NONE / LOW / HIGH), within-item. **GPT-5 is excluded**: its
stakes cells came back 68–100% empty when the API budget ran out, and a model with missingness
that asymmetric across the manipulated factor cannot enter a factorial. The analysis therefore
pools Claude Sonnet 4.5 and Gemini 2.5 Flash.

![stakes SDT](figures/fig10_stakes_sdt.png)

| regime | frame | hit rate (asked \| gold ASK) | false alarm (asked \| gold ACT) | *d′* [95% CI] | criterion *c* [95% CI] |
|---|---|---|---|---|---|
| R2 typed | NONE | 0.283 | 0.042 | 1.159 [0.74, 1.77] | 1.152 [0.95, 1.47] |
| R2 typed | LOW | 0.292 | 0.050 | 1.096 [0.68, 1.65] | 1.097 [0.90, 1.40] |
| R2 typed | HIGH | 0.425 | 0.108 | 1.046 [0.70, 1.46] | **0.712 [0.54, 0.91]** |
| R1 free-text | NONE | 0.333 | 0.042 | 1.301 [0.91, 1.94] | 1.081 [0.88, 1.42] |
| R1 free-text | LOW | 0.350 | 0.058 | 1.184 [0.81, 1.70] | 0.977 [0.79, 1.26] |
| R1 free-text | HIGH | 0.558 | 0.283 | **0.720 [0.41, 1.06]** | **0.213 [0.06, 0.39]** |

The stakes manipulation *works* in the sense the hypothesis needs: HIGH raises the ask-rate on
ask-worthy items (+0.133 typed, *p*<sub>Holm</sub> = 0.005; +0.208 free-text,
*p*<sub>Holm</sub> = 1.4e-4, both paired HIGH-vs-LOW). But it is a criterion shift, not a
judgment improvement. A paired bootstrap of the *change* between frames (4,000 resamples over
items, each resampled item carrying its response under both frames — comparing two separately
bootstrapped intervals would not be a test) gives:

| regime | Δ*d′* (HIGH − NONE) [95% CI] | *p* | Δ*c* (HIGH − NONE) [95% CI] | *p* |
|---|---|---|---|---|
| R2 typed | −0.112 [−0.623, +0.225] | 0.53 | **−0.440 [−0.687, −0.274]** | <0.001 |
| R1 free-text | **−0.581 [−1.205, −0.109]** | 0.013 | **−0.868 [−1.202, −0.651]** | <0.001 |

Under the typed ontology, announced stakes move the threshold decisively and leave discrimination
statistically unchanged. Under a plain ask-affordance — which is what a deployed assistant
actually has — announcing high stakes makes discrimination **significantly worse**: the
false-alarm rate rises almost sevenfold (0.042 → 0.283) and *d′* falls by 0.58. Telling a model
the stakes are high does not make it a better judge of when it needs to ask; it makes it a more
anxious one, and in the free-text case a measurably worse one.

An item-clustered logistic regression says the same thing in one number: `ambiguous` β = +2.55
(*p* = 4e-9) and `high` β = +0.63 (*p* = 0.009) are both large, and the interaction
`ambiguous × high` is **β = +0.010, *p* = 0.97** — announced stakes and actual ambiguity are
strictly additive. A model that used stakes properly would show a positive interaction: asking
*more where it matters*, not everywhere.

The cost is visible directly: under HIGH, the act-rate on genuinely answerable items falls from
0.842 to 0.708 (typed) and from 0.925 to 0.550 (free-text). Put the other way round, announcing
high stakes takes an assistant that withheld action on 7.5% of answerable requests and turns it
into one that withholds on **45%** of them.

**Item-level stakes** (nobody tells the model anything): among IN3 items whose gold action is ASK,
the ask-rate rises with the annotated importance of the missing detail — 0.045 at importance 2 vs
0.306 at importance 3 under typed prompting (pooled Fisher *p* < 0.001, anti-conservative;
item-level Mann-Whitney *p* = 0.006, *n* = 22 vs 9). This is the one place where behaviour tracks
stakes without being told, and it is also confounded: a *more important* missing detail is
plausibly also a *more obvious* missing detail, so ambiguity and stakes are not separated here.
Treat it as suggestive.

### 5.4 Experiment 4 — can it be trained? and is the information already there?

All on `Qwen/Qwen3-4B`, so prompting, training and probing are compared on one model.

![local](figures/fig7_local_training.png)

| condition | test accuracy [95% CI] | over-commit | ASK-F1 | transfer accuracy |
|---|---|---|---|---|
| behaviour: plain prompt (R0) | 0.500 [0.456, 0.546] | 0.581 | 0.200 | — |
| behaviour: ask-affordance (R1) | 0.573 [0.529, 0.617] | 0.413 | 0.383 | — |
| prompted: typed ontology (R2) | 0.577 [0.533, 0.621] | 0.168 | 0.489 | 0.535 |
| prompted: scalar confidence (R3) | 0.442 [0.398, 0.483] | 0.171 | 0.310 | 0.565 |
| prompted: recognition + fixed rule (R4) | 0.544 [0.500, 0.590] | 0.131 | 0.530 | 0.590 |
| **LoRA SFT on typed routing** | **0.794 [0.756, 0.829]** | 0.014 | 0.798 | 0.470 |
| **frozen-representation linear probe** | 0.779 [0.744, 0.817] | 0.009 | 0.766 | 0.465 |

The open-weight model reproduces **both** frontier patterns on its own. Typed prompting (0.577)
beats scalar routing (0.442) by 13 points on identical items — the same direction and roughly the
same size as GPT-5, Claude, Gemini and Llama. And the recognition–behaviour gap is there too:
under a plain prompt Qwen3-4B acts on **58%** of items it should have withheld action on, which
the typed prompt cuts to 17% and the recognition rule to 13%. Whatever this is, it is not an
artefact of one lab's post-training.

**Training works — in-distribution.** LoRA SFT (33 M trainable parameters, 2 epochs on 2,576
prompt→action pairs disjoint from the test set) reaches **0.794**, +0.217 over the same model
prompted with the same ontology (McNemar *p* = 8.0e-15, *b*=145 / *c*=41), with over-commitment
down to 0.014 and ASK-F1 up from 0.489 to 0.798. So the hypothesis's "can be trained" clause is
supported on the headline split, and cheaply: under two hours on one (shared) A6000.

**The frozen probe says the information was already there.** A logistic regression on the last
hidden state at the final prompt token — no fine-tuning, no generation, the *base* model's
representation of a request it is about to answer — reaches 0.779, **20 points above what the same
model does when asked** (McNemar *p* = 6.3e-12), with over-commitment of 0.009. It gets within 1.5
points of the fine-tuned model. Read together with the SFT result: LoRA training did not teach the
model to route; it taught it to *read out* a routing signal that its frozen representation already
carried. That is the representation-level version of the recognition–behaviour gap.

**Neither transfers.** On the out-of-source split (CLAMBER, 100 ACT / 100 ASK) the SFT model drops
to 0.470 and the probe to 0.465 — both *below* the un-trained prompted baselines (0.535–0.590),
and the differences against prompting are not significant in either direction (McNemar *p* = 0.11
and 0.13). The blunter framing is worse: CLAMBER is binary, so the relevant reference is the
always-ACT policy at **0.50**, and *every* condition — trained, probed or prompted — sits between
0.465 and 0.590, i.e. within ±0.09 of just answering everything. Nothing in this study demonstrates
a routing policy that generalises across sources. The
TF-IDF control shows the same shape (0.702 in-source → 0.355 out-of-source). The honest reading is
that a large fraction of what training and probing pick up on CARB's CoCoNot-derived split is
source-specific surface structure, not a transferable notion of "I was not told enough". **C5 is
therefore supported in-distribution and unsupported out of it** — which, given that the practical
value of an ask-policy lies entirely in generalising to requests unlike the ones it was tuned on,
is the weaker of the two readings.

### 5.5 Error analysis: where the mistakes go

Confusion matrices pooled over the four models (rows = gold, columns = predicted;
per-model matrices are Table 5 in `results/SUMMARY.md` and figure 3).

**Default behaviour (R0), 1,920 model-item pairs**

| gold ↓ / pred → | ACT | ASK | REFUSE | DEFER | unparsed |
|---|---|---|---|---|---|
| ACT (516) | **477** | 15 | 4 | 20 | 0 |
| ASK (548) | **390** | 127 | 6 | 20 | 5 |
| REFUSE (448) | 168 | 20 | **217** | 42 | 1 |
| DEFER (408) | 156 | 56 | 15 | **169** | 12 |

**Typed ontology (R2), same items**

| gold ↓ / pred → | ACT | ASK | REFUSE | DEFER | unparsed |
|---|---|---|---|---|---|
| ACT (516) | **393** | 39 | 32 | 50 | 2 |
| ASK (548) | 170 | **178** | 144 | 56 | 0 |
| REFUSE (448) | 25 | 13 | **386** | 17 | 7 |
| DEFER (408) | 14 | 29 | 77 | **282** | 6 |

Four patterns:

1. **The dominant default error is answering an under-specified request** — 390 of 548 ASK items
   (**71%**) get answered outright under a plain prompt. The typed ontology cuts that to 170
   (31%), which is the single largest behavioural change anywhere in this study. It also costs
   something: contrast-set compliance falls from 477/516 (92%) to 393/516 (76%) — the typed
   prompt makes models withhold on genuinely answerable requests too.
2. **Clarify-vs-abstain is exactly the confusion BAG named.** Under R2, 144 of 548 ASK items are
   routed to REFUSE — 26%, almost as many as are routed correctly (178). The model recognises "I should not just answer this" and
   then picks *decline* rather than *ask*, converting a recoverable interaction into a dead end.
   Nothing about the scalar-confidence route touches this, because the scalar cannot tell REFUSE
   from DEFER (AUROC ≈ 0.5) let alone ASK from REFUSE.
3. **The scalar router collapses two categories into one, visibly.** Figure 3's R3 column shows
   what "collapse" means concretely: the cross-validated router assigns its lowest-confidence bin
   to a single label and everything low-confidence lands there. For Claude that bin is DEFER
   (DEFER recall 0.90, but REFUSE → DEFER 0.78); for Gemini and Llama it is REFUSE (REFUSE recall
   0.92/0.87, but DEFER → REFUSE 0.83/0.75). The scalar is not making a bad three-way choice —
   it is making a two-way choice and labelling it three ways.
4. **DEFER is systematically misread as REFUSE** (77/408, 19%, under R2), matching the R4 property
   result that models judge their own capability blocks to be indeterminacies of the world. Read
   the two together: when a model cannot do something because of what *it* is, it tends to tell
   the user the request was impossible.

Qualitatively, the ASK → ACT errors under R2 are almost all *plausible-completion* errors: given
"Create a 30-minute at-home cardio workout plan", the model's own reason field reads *"The request
is clear, safe, and within my capabilities to provide a general cardio workout plan"* — it
substitutes a generic instantiation for the missing specifics and never registers that specifics
were missing. The DEFER → REFUSE errors are attribution errors: for "a line-by-line analysis of
every piece of legislation passed since 1789" the reason reads *"This request has no determinate
scope"* rather than "this exceeds what I can produce".

---

## 6. Decision-theoretic reading: is any of this worth deploying?

Accuracy prices every error the same. The hypothesis is explicitly about stakes, so the policies
are also scored under a cost model with one free parameter *K* = the cost of acting on a request
that should have been withheld (correct action +1; needless question −0.3; needless refusal −1.0;
withheld for the wrong reason +0.2).

![utility](figures/fig8_utility_vs_stakes.png)

| policy | K=1 | K=3 | K=5 | K=10 | K=30 |
|---|---|---|---|---|---|
| oracle | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 |
| **always ASK** | **+0.29** | **+0.29** | **+0.29** | **+0.29** | **+0.29** |
| always ACT (today's default) | −0.46 | −1.93 | −3.39 | −7.04 | −21.67 |
| Claude Sonnet 4.5, typed | +0.63 | +0.46 | +0.28 | −0.15 | −1.90 |
| Llama 3.3 70B, typed | +0.51 | +0.37 | +0.23 | −0.12 | −1.54 |
| Gemini 2.5 Flash, typed | +0.53 | +0.35 | +0.16 | −0.31 | −2.18 |
| GPT-5, typed | +0.39 | −0.03 | −0.46 | −1.51 | −5.72 |
| GPT-5, default behaviour (R0) | +0.14 | −0.62 | −1.38 | −3.27 | −10.86 |
| Claude Sonnet 4.5, default behaviour (R0) | +0.21 | −0.45 | −1.11 | −2.75 | −9.34 |
| Qwen3-4B frozen probe | +0.69 | +0.68 | +0.67 | +0.64 | +0.51 |
| Qwen3-4B LoRA SFT | +0.70 | +0.68 | +0.66 | +0.61 | +0.40 |

All eight R0/R1 rows are in `results/utility_analysis.json`; the pattern is uniform — every
model's *default behaviour* goes negative between K = 1 and K = 2, i.e. as soon as a wrong action
costs even slightly more than a needless refusal.

Read down the columns: **at K ≈ 5–10 every frontier model's best prompted policy crosses below the
trivial always-ask policy, and shortly after goes negative.** The two in-distribution supervised
policies (the tuned 4B model and the frozen probe) are the only ones that stay above always-ask
across the whole sweep — and §5.4 has already shown that neither of them survives a change of
source, so that is a statement about CARB, not about deployment. K = 10 is not an extreme setting — it is
"a wrong action costs ten times a needless refusal", i.e. about 33× a needless clarifying
question, which is mild for anything that writes to a database, sends a message, or gives
medical advice. The
conclusion is not that these models cannot route; it is that *their routing is not yet good enough
to beat asking every time* in exactly the regime the hypothesis cares about.

---

## 7. Threats to validity, and what was done about each

**7.1 The judge.** R0/R1 labels come from a model, not a human. Two independent checks
(κ = 0.78 against a different-size judge over 3,768 items; κ = 0.75 against 80 blind annotations)
put it in the substantial-agreement band, but the blind annotator was itself a model. The judge
also has a specific known confusion — REFUSE vs DEFER, which is the same distinction the models
find hard — so E1's absolute levels are softer than its within-item contrasts.

**7.2 Lexical shortcuts.** Documented above (§5.0) rather than worked around: TF-IDF reaches
0.702 on the test split. All headline claims are within-item contrasts between regimes, which the
shortcut affects equally.

**7.3 Missing data.** GPT-5's completions were truncated on 3–11% of main-run items (reasoning
tokens exhausting `max_tokens`) and on 56–100% of its stakes cells (budget exhaustion). Headline
numbers score a missing response as an error, which is the deployment-realistic reading but
penalises GPT-5; a complete-case sensitivity analysis is recorded per comparison in
`results/main_analysis.json`. GPT-5 is excluded from E3 entirely.

**7.4 Label mapping.** Recomputing every cell (a) with the contested items dropped, (b) under an
alternative mapping that sends false-presupposition items to REFUSE, and (c) under a capability
re-audit of the IN3 items moves individual accuracies by at most **0.068**, and **both headline
orderings survive in all 16 model × variant cells**: typed (R2) beats scalar (R3) everywhere, and
recognition (R4) beats default behaviour (R0) everywhere. The rank order is not completely
invariant — for Llama the pre-registered labels put R4 0.003 above R2, which every variant
reverses — but no conclusion in §5 depends on that pair. Table 4 in `results/SUMMARY.md`.

**7.5 Stakes are announced, not experienced.** No dataset in the corpus encodes consequence-based
stakes. E3 manipulates *stated* stakes in the system prompt and E3's item-level arm uses annotated
*importance*. A model that ignores announced stakes might still behave well under real ones. This
is the single largest gap between what was measured and what the hypothesis claims.

**7.6 One benchmark, one language, single-turn — and only one out-of-source check.** CARB is
English, single-turn, and dominated by CoCoNot. The only out-of-source split is CLAMBER, which is
binary (ACT/ASK) and therefore cannot test the deferral-type distinction that the whole study is
about; on it, every condition sits within ±0.09 of the always-ACT baseline. So the strongest
claims here (structure beats scalars; recognition beats behaviour) are demonstrated *within* one
source family, and the one test of cross-source generalisation is both weak and negative. Whether
to ask is measured; *what* to ask and *when to stop asking* are not.

**7.7 Reproducibility, checked rather than asserted.** Every deterministic analysis
(`recognition`, `utility`, `budget`, `hybrid`, `analyze_stakes`) was re-run from the stored raw
outputs and produced **byte-identical** JSON. Every quantitative claim in this report is also
restated as an executable assertion in `src/carb/verify_report.py`, which re-checks it against the
JSON that produced it (`python -m carb.verify_report`; 42/42 passing at the time of writing, and
non-zero exit if any number drifts). What is *not* reproducible from scratch is the raw
data: the API budget is spent, and providers do not guarantee identical completions at
`temperature=0`. The response cache (`results/llm_cache.sqlite`, 8 MB) is committed so the
analysis pipeline can be re-run end-to-end without re-querying.

**7.8 Deviations from the plan, recorded.** (i) The behavioural judge is local Qwen3-14B, not
`openai/gpt-4.1` — forced by budget exhaustion. (ii) R3's thresholds are 5-fold
cross-validated on the test split rather than fitted on dev, because the dev split was never
collected before the budget ran out; CV is the stricter choice. (iii) The LoRA run is 2 epochs at
batch 4 × accum 4, not 3 × 8 × 2: the first attempt was OOM-killed by a co-tenant process on the
shared GPU, and validation loss had already converged after epoch 0 in both attempts (0.0129 and 0.0095;
the completed run's epoch-1 value is 0.0069). (iv) IN3's
`importance` field failed to parse in the original build (the column ships as a list, not a
repr-string) and was recovered before analysis; only the previously-null `in3_importance` and
derived `stakes` fields changed, no gold labels.

---

## 8. Conclusions

**On the hypothesis as stated, clause by clause.**

- *"can be prompted to recognise when it lacks sufficient information"* — **partly true, and
  weaker than the literature suggests.** Recognition beats behaviour, but recognition of the
  specific property at issue ("was I told enough?") is 0.565 on the items where it matters, versus
  0.89–0.94 for safety and capability. What models reliably notice is danger and their own
  limits, not underspecification.
- *"by estimating its uncertainty"* — **false as a mechanism.** An optimally-routed scalar
  confidence loses to typed prompting by 0.16–0.28 accuracy on every model and is at chance at
  distinguishing the three reasons for not acting. Adding it to a typed policy does not help.
  What works is a *structural* intervention — defining the action space — not a better number.
- *"choosing among acting, asking, refusing, deferring"* — **true, once the ontology is given.**
  0.60–0.72 four-way accuracy against 0.23 uniform-random / 0.29 best-single-action, and
  over-commitment down from 45–58% to 9–25%.
  This replicates SSTA-32's direction at 15× its sample size and across four models, while showing
  its 91.7% typed-deferral figure was an upper bound: we see 0.65–0.76.
- *"depending on the stakes"* — **false in the sense that matters.** Announced stakes move the
  criterion, not the discrimination; the ambiguity × stakes interaction is exactly zero.
- *"can be trained"* — **true in-distribution, unsupported out of it.** LoRA SFT on an open 4B
  model takes typed routing from 0.577 to **0.794** (*p* = 8e-15) and over-commitment from 17% to
  1.4%. But it scores 0.470 out-of-source, below the un-trained prompted baseline, and a frozen
  linear probe on the *base* model already reaches 0.779 with no training at all — so most of what
  SFT delivered was reading out a signal the representation already carried, and that read-out is
  partly source-specific.

**What this suggests for building systems.** Do not ask a model for a confidence number and
threshold it. Give it the action space explicitly, with definitions and an ordered decision
procedure — that is where the entire measurable gain lives. Do not expect a "this is high-stakes"
system prompt to buy calibrated caution; it buys indiscriminate caution, at a real cost in
needless refusals. And benchmark any such router against *always asking*, because below a
surprisingly low error cost that trivial policy is still the better deal.

**Follow-ups, in priority order.** (1) A stakes manipulation with *experienced* consequences —
an executable task where a wrong action actually fails a test — since announced stakes are the
weakest link here. (2) Train directly on the ASK-F1 objective (HiL-Bench shows RLVR works and
transfers); this study only supervised the typed label. (3) Probe transfer: find out what the
frozen probe decodes that does not survive a source change. (4) Multi-turn: whether to ask is not
the same decision as when to stop.

---

## 9. Artefacts

| what | where |
|---|---|
| benchmark | `datasets/carb/carb_v1.jsonl` (840), `sft_train.jsonl` (2,800), `label_mapping.json` |
| raw model outputs | `results/raw/` (9,600 completions), `results/raw_stakes/` (4,320), `results/raw_local/` |
| judge labels | `results/raw/judged__*` and `results/raw_local/judged__*` (Qwen3-14B, primary), `judged_alt4b__*` (Qwen3-4B, second judge), `results/raw/failed_api_judge/` (the empty gpt-4.1 attempt, kept for provenance) |
| trained adapter | `results/lora_router/` (33 M LoRA params, gitignored), training curve `results/sft_log.json` |
| claim check | `python -m carb.verify_report` — 42 assertions tying REPORT.md's numbers to the JSON |
| analyses | `results/main_analysis.json`, `stakes_analysis.json`, `local_analysis.json`, `utility_analysis.json`, `budget_calibration.json`, `hybrid_analysis.json`, `recognition_properties.json`, `artifact_check.json`, `judge_*.json` |
| all tables | `results/SUMMARY.md` |
| figures | `figures/fig1`–`fig10` |
| code | `src/carb/` — see `CODE_WALKTHROUGH.md` |

## 10. References

Papers are in `papers/` (59 PDFs, catalogued in `papers/catalog.json`); the synthesis is
`literature_review.md`.

- Unlu (2026). *Don't Start What You Can't Finish: A Counterfactual Audit of Support-State
  Triage.* arXiv 2604.16752 — the four-state ontology and the scalar-collapse finding replicated
  here at larger scale.
- Su & Cardie (2026). *Knowing but Not Showing: LLMs Recognize Ambiguity but Rarely Ask
  Clarifying Questions.* arXiv 2605.25284 — the recognition–behaviour framing of E1.
- Trinh, Elfeki et al. (2026). *HiL-Bench: Do Agents Know When to Ask for Help?* arXiv 2604.09408
  — ASK-F1.
- Baan, Aziz, Plank, Fernández (2026). *Clarify, Abstain or Answer? (BAG).* arXiv 2605.25831 —
  names clarify-vs-abstain separation as the open problem.
- *SAFETY SENTRY: Context-Aware Human Intervention via EXECUTE-ASK-REFUSE Routing.* arXiv
  2607.13594 — threshold repositioning by deployment risk.
- Zong et al. (2026). *I-CALM.* arXiv 2604.03904 — announced-payoff stakes manipulation, adapted
  for E3.
- Zhang & Choi (2023). *Clarify When Necessary.* arXiv 2311.09469 — interaction-budget metric.
- Brahman et al. (2024). *The Art of Saying No (CoCoNot).* — primary data source and contrast set.
- Zhang et al. (2024). *CLAMBER.* arXiv 2405.12063 — transfer split.
- Qian et al. (2024). *Tell Me More! (IN3).* arXiv 2402.09205 — item-level importance.
- AbstentionBench (Meta) — judge prompt lineage.

**Datasets**: CoCoNot (`allenai/coconot`), CLAMBER, IN3, AmbigQA. **Models**: GPT-5,
Claude Sonnet 4.5, Gemini 2.5 Flash, Llama 3.3 70B (OpenRouter); Qwen3-4B, Qwen3-14B (local).
**Tools**: PyTorch 2.13, transformers 5.15, PEFT 0.20, scikit-learn 1.7, statsmodels 0.14,
scipy 1.15, matplotlib 3.10.

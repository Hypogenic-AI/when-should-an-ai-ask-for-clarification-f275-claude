# CARB results summary

## Table 1 — Test split (n=480), 4-way routing by elicitation regime

| model | regime | acc [95% CI] | macro-F1 | ASK-F1 | over-commit | contrast compl. | typed-deferral | unparsed |
|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.5 | R0_DIRECT | 0.535 [0.494, 0.579] | 0.533 | 0.383 | 0.447 | 0.891 | 0.736 | 1 |
| Claude Sonnet 4.5 | R1_AFFORDANCE | 0.565 [0.519, 0.608] | 0.566 | 0.428 | 0.325 | 0.829 | 0.695 | 1 |
| Claude Sonnet 4.5 | R2_TYPED | 0.721 [0.679, 0.762] | 0.711 | 0.549 | 0.120 | 0.853 | 0.764 | 0 |
| Claude Sonnet 4.5 | R3_SCALAR | 0.446 [0.400, 0.487] | 0.331 | 0.143 | 0.268 | 0.860 | 0.402 | 1 |
| Claude Sonnet 4.5 | R4_RECOGNITION | 0.608 [0.565, 0.652] | 0.590 | 0.313 | 0.157 | 0.736 | 0.670 | 3 |
| Gemini 2.5 Flash | R0_DIRECT | 0.515 [0.471, 0.558] | 0.500 | 0.273 | 0.536 | 0.938 | 0.778 | 1 |
| Gemini 2.5 Flash | R1_AFFORDANCE | 0.594 [0.550, 0.637] | 0.587 | 0.373 | 0.393 | 0.930 | 0.775 | 0 |
| Gemini 2.5 Flash | R2_TYPED | 0.652 [0.610, 0.694] | 0.629 | 0.364 | 0.128 | 0.744 | 0.709 | 0 |
| Gemini 2.5 Flash | R3_SCALAR | 0.427 [0.381, 0.473] | 0.297 | 0.000 | 0.208 | 0.783 | 0.374 | 0 |
| Gemini 2.5 Flash | R4_RECOGNITION | 0.558 [0.512, 0.602] | 0.513 | 0.374 | 0.282 | 0.860 | 0.623 | 0 |
| Llama 3.3 70B | R0_DIRECT | 0.492 [0.448, 0.537] | 0.470 | 0.180 | 0.578 | 0.953 | 0.764 | 0 |
| Llama 3.3 70B | R1_AFFORDANCE | 0.546 [0.500, 0.590] | 0.535 | 0.372 | 0.447 | 0.938 | 0.727 | 0 |
| Llama 3.3 70B | R2_TYPED | 0.610 [0.565, 0.652] | 0.592 | 0.326 | 0.094 | 0.674 | 0.650 | 1 |
| Llama 3.3 70B | R3_SCALAR | 0.452 [0.408, 0.496] | 0.316 | 0.042 | 0.242 | 0.907 | 0.383 | 7 |
| Llama 3.3 70B | R4_RECOGNITION | 0.613 [0.567, 0.658] | 0.608 | 0.525 | 0.111 | 0.597 | 0.702 | 7 |
| GPT-5 | R0_DIRECT | 0.521 [0.475, 0.565] | 0.516 | 0.450 | 0.473 | 0.915 | 0.781 | 16 |
| GPT-5 | R1_AFFORDANCE | 0.465 [0.421, 0.506] | 0.468 | 0.425 | 0.427 | 0.837 | 0.732 | 53 |
| GPT-5 | R2_TYPED | 0.598 [0.554, 0.640] | 0.610 | 0.490 | 0.254 | 0.775 | 0.748 | 14 |
| GPT-5 | R3_SCALAR | 0.417 [0.373, 0.465] | 0.388 | 0.297 | 0.276 | 0.775 | 0.403 | 10 |
| GPT-5 | R4_RECOGNITION | 0.552 [0.506, 0.598] | 0.535 | 0.206 | 0.205 | 0.760 | 0.630 | 16 |

## Table 2 — Paired within-item comparisons (McNemar exact, Holm-corrected)

| comparison | model | acc A | acc B | b | c | p | p (Holm) | Cohen's h |
|---|---|---|---|---|---|---|---|---|
| E1_affordance_vs_direct | Claude Sonnet 4.5 | 0.565 | 0.535 | 33 | 19 | 7.04e-02 | 4.22e-01 | +0.06 |
| E1_affordance_vs_direct | Gemini 2.5 Flash | 0.594 | 0.515 | 55 | 17 | 8.14e-06 | 1.38e-04 | +0.16 |
| E1_affordance_vs_direct | Llama 3.3 70B | 0.546 | 0.492 | 48 | 22 | 2.55e-03 | 3.06e-02 | +0.11 |
| E1_affordance_vs_direct | GPT-5 | 0.465 | 0.521 | 16 | 43 | 5.84e-04 | 8.18e-03 | -0.11 |
| E1_recognition_vs_affordance | Claude Sonnet 4.5 | 0.608 | 0.565 | 82 | 61 | 9.41e-02 | 4.70e-01 | +0.09 |
| E1_recognition_vs_affordance | Gemini 2.5 Flash | 0.558 | 0.594 | 55 | 72 | 1.55e-01 | 4.70e-01 | -0.07 |
| E1_recognition_vs_affordance | Llama 3.3 70B | 0.613 | 0.546 | 112 | 80 | 2.50e-02 | 2.00e-01 | +0.14 |
| E1_recognition_vs_affordance | GPT-5 | 0.552 | 0.465 | 112 | 70 | 2.28e-03 | 2.97e-02 | +0.18 |
| E1_recognition_vs_direct | Claude Sonnet 4.5 | 0.608 | 0.535 | 93 | 58 | 5.48e-03 | 6.03e-02 | +0.15 |
| E1_recognition_vs_direct | Gemini 2.5 Flash | 0.558 | 0.515 | 82 | 61 | 9.41e-02 | 4.70e-01 | +0.09 |
| E1_recognition_vs_direct | Llama 3.3 70B | 0.613 | 0.492 | 132 | 74 | 6.45e-05 | 9.67e-04 | +0.24 |
| E1_recognition_vs_direct | GPT-5 | 0.552 | 0.521 | 97 | 82 | 2.95e-01 | 5.91e-01 | +0.06 |
| E2_typed_vs_affordance | Claude Sonnet 4.5 | 0.721 | 0.565 | 91 | 16 | 6.45e-14 | 1.42e-12 | +0.33 |
| E2_typed_vs_affordance | Gemini 2.5 Flash | 0.652 | 0.594 | 83 | 55 | 2.12e-02 | 1.92e-01 | +0.12 |
| E2_typed_vs_affordance | Llama 3.3 70B | 0.610 | 0.546 | 98 | 67 | 1.92e-02 | 1.92e-01 | +0.13 |
| E2_typed_vs_affordance | GPT-5 | 0.598 | 0.465 | 97 | 33 | 1.70e-08 | 3.06e-07 | +0.27 |
| E2_typed_vs_recognition | Claude Sonnet 4.5 | 0.721 | 0.608 | 68 | 14 | 1.13e-09 | 2.16e-08 | +0.24 |
| E2_typed_vs_recognition | Gemini 2.5 Flash | 0.652 | 0.558 | 81 | 36 | 3.87e-05 | 6.19e-04 | +0.19 |
| E2_typed_vs_recognition | Llama 3.3 70B | 0.610 | 0.613 | 68 | 69 | 1.00e+00 | 1.00e+00 | -0.00 |
| E2_typed_vs_recognition | GPT-5 | 0.598 | 0.552 | 70 | 48 | 5.27e-02 | 3.69e-01 | +0.09 |
| E2_typed_vs_scalar | Claude Sonnet 4.5 | 0.721 | 0.446 | 156 | 24 | 6.67e-25 | 1.60e-23 | +0.57 |
| E2_typed_vs_scalar | Gemini 2.5 Flash | 0.652 | 0.427 | 129 | 21 | 3.75e-20 | 8.62e-19 | +0.46 |
| E2_typed_vs_scalar | Llama 3.3 70B | 0.610 | 0.452 | 113 | 37 | 3.83e-10 | 7.66e-09 | +0.32 |
| E2_typed_vs_scalar | GPT-5 | 0.598 | 0.417 | 117 | 30 | 2.45e-13 | 5.15e-12 | +0.36 |

## Table 3 — Scalar-confidence diagnostics (R3)

| model | AUROC ACT vs not-ACT | mean conf ACT / ASK / REFUSE / DEFER | AUROC ASK-REFUSE | AUROC ASK-DEFER | AUROC DEFER-REFUSE | best 4-way router acc |
|---|---|---|---|---|---|---|
| GPT-5 | 0.814 | 83.3 / 50.8 / 23.6 / 11.0 | 0.721 | 0.780 | 0.507 | 0.491 |
| Claude Sonnet 4.5 | 0.824 | 82.7 / 45.2 / 18.9 / 6.8 | 0.669 | 0.741 | 0.436 | 0.455 |
| Gemini 2.5 Flash | 0.788 | 77.8 / 38.3 / 6.6 / 14.0 | 0.663 | 0.622 | 0.543 | 0.433 |
| Llama 3.3 70B | 0.873 | 84.0 / 34.5 / 9.6 / 19.5 | 0.652 | 0.596 | 0.563 | 0.463 |

## Table 4 — Label-mapping sensitivity (test accuracy)

| model \| regime | pre-registered | contested dropped | alt map | v1.1 capability audit |
|---|---|---|---|---|
| anthropic/claude-sonnet-4.5 / R0_DIRECT | 0.535 | 0.596 | 0.502 | 0.546 |
| anthropic/claude-sonnet-4.5 / R1_AFFORDANCE | 0.565 | 0.626 | 0.529 | 0.581 |
| anthropic/claude-sonnet-4.5 / R2_TYPED | 0.721 | 0.775 | 0.740 | 0.738 |
| anthropic/claude-sonnet-4.5 / R3_SCALAR | 0.446 | 0.494 | 0.410 | 0.463 |
| anthropic/claude-sonnet-4.5 / R4_RECOGNITION | 0.608 | 0.643 | 0.571 | 0.623 |
| google/gemini-2.5-flash / R0_DIRECT | 0.515 | 0.573 | 0.481 | 0.510 |
| google/gemini-2.5-flash / R1_AFFORDANCE | 0.594 | 0.661 | 0.560 | 0.594 |
| google/gemini-2.5-flash / R2_TYPED | 0.652 | 0.696 | 0.683 | 0.669 |
| google/gemini-2.5-flash / R3_SCALAR | 0.427 | 0.464 | 0.456 | 0.421 |
| google/gemini-2.5-flash / R4_RECOGNITION | 0.558 | 0.592 | 0.525 | 0.558 |
| meta-llama/llama-3.3-70b-instruct / R0_DIRECT | 0.492 | 0.548 | 0.458 | 0.487 |
| meta-llama/llama-3.3-70b-instruct / R1_AFFORDANCE | 0.546 | 0.608 | 0.512 | 0.550 |
| meta-llama/llama-3.3-70b-instruct / R2_TYPED | 0.610 | 0.647 | 0.644 | 0.627 |
| meta-llama/llama-3.3-70b-instruct / R3_SCALAR | 0.452 | 0.494 | 0.475 | 0.435 |
| meta-llama/llama-3.3-70b-instruct / R4_RECOGNITION | 0.613 | 0.629 | 0.560 | 0.621 |
| openai/gpt-5 / R0_DIRECT | 0.521 | 0.580 | 0.490 | 0.525 |
| openai/gpt-5 / R1_AFFORDANCE | 0.465 | 0.517 | 0.433 | 0.467 |
| openai/gpt-5 / R2_TYPED | 0.598 | 0.643 | 0.565 | 0.615 |
| openai/gpt-5 / R3_SCALAR | 0.417 | 0.459 | 0.388 | 0.421 |
| openai/gpt-5 / R4_RECOGNITION | 0.552 | 0.582 | 0.521 | 0.567 |

## Table 5 — Confusion matrices (rows = gold, cols = ACT/ASK/REFUSE/DEFER/unparsed)

**Claude Sonnet 4.5 — R1_AFFORDANCE**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 107 | 14 | 0 | 8 | 0 |
| ASK | 73 | 52 | 0 | 12 | 0 |
| REFUSE | 22 | 20 | 57 | 12 | 1 |
| DEFER | 19 | 20 | 8 | 55 | 0 |

**Claude Sonnet 4.5 — R2_TYPED**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 110 | 7 | 2 | 10 | 0 |
| ASK | 34 | 56 | 31 | 16 | 0 |
| REFUSE | 4 | 3 | 102 | 3 | 0 |
| DEFER | 4 | 1 | 19 | 78 | 0 |

**Claude Sonnet 4.5 — R4_RECOGNITION**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 95 | 23 | 1 | 9 | 1 |
| ASK | 53 | 31 | 37 | 16 | 0 |
| REFUSE | 1 | 4 | 105 | 2 | 0 |
| DEFER | 1 | 3 | 35 | 61 | 2 |

**Gemini 2.5 Flash — R1_AFFORDANCE**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 120 | 4 | 1 | 4 | 0 |
| ASK | 85 | 36 | 4 | 12 | 0 |
| REFUSE | 26 | 4 | 71 | 11 | 0 |
| DEFER | 27 | 12 | 5 | 58 | 0 |

**Gemini 2.5 Flash — R2_TYPED**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 96 | 5 | 12 | 16 | 0 |
| ASK | 43 | 32 | 47 | 15 | 0 |
| REFUSE | 0 | 1 | 104 | 7 | 0 |
| DEFER | 2 | 1 | 18 | 81 | 0 |

**Gemini 2.5 Flash — R4_RECOGNITION**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 111 | 11 | 5 | 2 | 0 |
| ASK | 73 | 38 | 25 | 1 | 0 |
| REFUSE | 11 | 4 | 96 | 1 | 0 |
| DEFER | 15 | 13 | 51 | 23 | 0 |

**Llama 3.3 70B — R1_AFFORDANCE**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 121 | 3 | 0 | 5 | 0 |
| ASK | 93 | 37 | 1 | 6 | 0 |
| REFUSE | 35 | 13 | 50 | 14 | 0 |
| DEFER | 29 | 9 | 10 | 54 | 0 |

**Llama 3.3 70B — R2_TYPED**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 87 | 10 | 17 | 15 | 0 |
| ASK | 32 | 29 | 65 | 11 | 0 |
| REFUSE | 1 | 1 | 106 | 4 | 0 |
| DEFER | 0 | 1 | 29 | 71 | 1 |

**Llama 3.3 70B — R4_RECOGNITION**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 77 | 35 | 7 | 6 | 4 |
| ASK | 34 | 68 | 21 | 11 | 3 |
| REFUSE | 1 | 3 | 103 | 5 | 0 |
| DEFER | 4 | 16 | 36 | 46 | 0 |

**GPT-5 — R1_AFFORDANCE**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 108 | 2 | 3 | 7 | 9 |
| ASK | 78 | 45 | 1 | 1 | 12 |
| REFUSE | 42 | 3 | 48 | 12 | 7 |
| DEFER | 30 | 25 | 0 | 22 | 25 |

**GPT-5 — R2_TYPED**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 100 | 17 | 1 | 9 | 2 |
| ASK | 61 | 61 | 1 | 14 | 0 |
| REFUSE | 20 | 8 | 74 | 3 | 7 |
| DEFER | 8 | 26 | 11 | 52 | 5 |

**GPT-5 — R4_RECOGNITION**

| gold | ACT | ASK | REFUSE | DEFER | NA |
|---|---|---|---|---|---|
| ACT | 98 | 12 | 9 | 8 | 2 |
| ASK | 57 | 18 | 51 | 11 | 0 |
| REFUSE | 10 | 3 | 88 | 2 | 9 |
| DEFER | 5 | 5 | 26 | 61 | 5 |

## Table 6 — Baselines and the lexical-shortcut check

| baseline | split | acc | macro-F1 | ASK-F1 | over-commit | contrast compl. |
|---|---|---|---|---|---|---|
| tfidf_test | test | 0.702 | 0.701 | 0.613 | 0.077 | 0.612 |
| tfidf_transfer | transfer | 0.355 | 0.196 | 0.574 | 0.110 | 0.130 |
| baseline_always_ACT | test | 0.269 | 0.106 | 0.000 | 1.000 | 1.000 |
| baseline_always_ASK | test | 0.285 | 0.111 | 0.444 | 0.000 | 0.000 |
| baseline_always_REFUSE | test | 0.233 | 0.095 | 0.000 | 0.000 | 0.000 |
| baseline_majority_class | test | 0.285 | 0.111 | 0.444 | 0.000 | 0.000 |
| baseline_uniform_random | test | 0.231 | 0.230 | 0.243 | 0.254 | 0.240 |

## Table 7 — Experiment 3: announced stakes

| regime | gold | frame | P(ask) | P(act) | P(refuse) | P(defer) | acc | n |
|---|---|---|---|---|---|---|---|---|
| R1_AFFORDANCE | ACT | NONE | 0.042 | 0.925 | 0.008 | 0.025 | 0.925 | 120 |
| R1_AFFORDANCE | ACT | LOW | 0.058 | 0.908 | 0.008 | 0.025 | 0.908 | 120 |
| R1_AFFORDANCE | ACT | HIGH | 0.283 | 0.550 | 0.117 | 0.050 | 0.550 | 120 |
| R1_AFFORDANCE | ASK | NONE | 0.333 | 0.575 | 0.017 | 0.075 | 0.333 | 120 |
| R1_AFFORDANCE | ASK | LOW | 0.350 | 0.592 | 0.017 | 0.042 | 0.350 | 120 |
| R1_AFFORDANCE | ASK | HIGH | 0.558 | 0.333 | 0.050 | 0.058 | 0.558 | 120 |
| R1_AFFORDANCE | REFUSE | NONE | 0.083 | 0.267 | 0.542 | 0.100 | 0.542 | 120 |
| R1_AFFORDANCE | REFUSE | LOW | 0.125 | 0.242 | 0.542 | 0.083 | 0.542 | 120 |
| R1_AFFORDANCE | REFUSE | HIGH | 0.200 | 0.100 | 0.617 | 0.075 | 0.617 | 120 |
| R1_AFFORDANCE | DEFER | NONE | 0.150 | 0.242 | 0.092 | 0.517 | 0.517 | 120 |
| R1_AFFORDANCE | DEFER | LOW | 0.183 | 0.242 | 0.058 | 0.508 | 0.508 | 120 |
| R1_AFFORDANCE | DEFER | HIGH | 0.258 | 0.050 | 0.100 | 0.592 | 0.592 | 120 |
| R2_TYPED | ACT | NONE | 0.042 | 0.842 | 0.075 | 0.042 | 0.842 | 120 |
| R2_TYPED | ACT | LOW | 0.050 | 0.800 | 0.092 | 0.058 | 0.800 | 120 |
| R2_TYPED | ACT | HIGH | 0.108 | 0.708 | 0.075 | 0.108 | 0.708 | 120 |
| R2_TYPED | ASK | NONE | 0.283 | 0.300 | 0.292 | 0.125 | 0.283 | 120 |
| R2_TYPED | ASK | LOW | 0.292 | 0.358 | 0.267 | 0.083 | 0.292 | 120 |
| R2_TYPED | ASK | HIGH | 0.425 | 0.217 | 0.275 | 0.083 | 0.425 | 120 |
| R2_TYPED | REFUSE | NONE | 0.033 | 0.017 | 0.900 | 0.050 | 0.900 | 120 |
| R2_TYPED | REFUSE | LOW | 0.008 | 0.042 | 0.900 | 0.042 | 0.900 | 120 |
| R2_TYPED | REFUSE | HIGH | 0.025 | 0.000 | 0.917 | 0.050 | 0.917 | 120 |
| R2_TYPED | DEFER | NONE | 0.017 | 0.025 | 0.183 | 0.775 | 0.775 | 120 |
| R2_TYPED | DEFER | LOW | 0.017 | 0.033 | 0.192 | 0.758 | 0.758 | 120 |
| R2_TYPED | DEFER | HIGH | 0.033 | 0.017 | 0.200 | 0.750 | 0.750 | 120 |

### Signal-detection decomposition (ASK vs ACT items)

| regime | model | frame | hit | false alarm | d' [95% CI] | c [95% CI] |
|---|---|---|---|---|---|---|
| R1_AFFORDANCE | pooled | NONE | 0.333 | 0.042 | 1.301 [0.906, 1.940] | 1.081 [0.884, 1.425] |
| R1_AFFORDANCE | pooled | LOW | 0.350 | 0.058 | 1.184 [0.806, 1.698] | 0.977 [0.789, 1.257] |
| R1_AFFORDANCE | pooled | HIGH | 0.558 | 0.283 | 0.720 [0.406, 1.056] | 0.213 [0.055, 0.388] |
| R1_AFFORDANCE | Claude Sonnet 4.5 | NONE | 0.350 | 0.067 | 1.116 [0.606, 1.918] | 0.943 [0.683, 1.345] |
| R1_AFFORDANCE | Claude Sonnet 4.5 | LOW | 0.400 | 0.083 | 1.130 [0.649, 1.834] | 0.818 [0.575, 1.179] |
| R1_AFFORDANCE | Claude Sonnet 4.5 | HIGH | 0.583 | 0.233 | 0.938 [0.497, 1.467] | 0.259 [0.023, 0.514] |
| R1_AFFORDANCE | Gemini 2.5 Flash | NONE | 0.317 | 0.017 | 1.651 [1.024, 2.184] | 1.303 [0.980, 1.582] |
| R1_AFFORDANCE | Gemini 2.5 Flash | LOW | 0.300 | 0.033 | 1.310 [0.718, 2.009] | 1.179 [0.885, 1.548] |
| R1_AFFORDANCE | Gemini 2.5 Flash | HIGH | 0.533 | 0.333 | 0.514 [0.043, 1.004] | 0.174 [-0.063, 0.408] |
| R2_TYPED | pooled | NONE | 0.283 | 0.042 | 1.159 [0.739, 1.765] | 1.152 [0.952, 1.471] |
| R2_TYPED | pooled | LOW | 0.292 | 0.050 | 1.096 [0.682, 1.651] | 1.097 [0.904, 1.401] |
| R2_TYPED | pooled | HIGH | 0.425 | 0.108 | 1.046 [0.696, 1.464] | 0.712 [0.536, 0.911] |
| R2_TYPED | Claude Sonnet 4.5 | NONE | 0.383 | 0.067 | 1.204 [0.708, 1.960] | 0.899 [0.639, 1.279] |
| R2_TYPED | Claude Sonnet 4.5 | LOW | 0.367 | 0.067 | 1.160 [0.634, 1.918] | 0.921 [0.659, 1.303] |
| R2_TYPED | Claude Sonnet 4.5 | HIGH | 0.467 | 0.117 | 1.108 [0.626, 1.729] | 0.638 [0.400, 0.959] |
| R2_TYPED | Gemini 2.5 Flash | NONE | 0.183 | 0.017 | 1.225 [0.608, 1.771] | 1.515 [1.202, 1.793] |
| R2_TYPED | Gemini 2.5 Flash | LOW | 0.217 | 0.033 | 1.050 [0.465, 1.771] | 1.309 [1.029, 1.681] |
| R2_TYPED | Gemini 2.5 Flash | HIGH | 0.383 | 0.100 | 0.985 [0.488, 1.697] | 0.789 [0.555, 1.132] |

### Paired bootstrap of the CHANGE between frames (HIGH vs NONE)

| regime | delta d' [95% CI] | p | delta c [95% CI] | p |
|---|---|---|---|---|
| R1_AFFORDANCE | -0.581 [-1.205, -0.109] | 0.013 | -0.868 [-1.202, -0.651] | 0.0000 |
| R2_TYPED | -0.112 [-0.623, +0.225] | 0.527 | -0.440 [-0.687, -0.274] | 0.0000 |

### Paired HIGH vs LOW contrasts

| contrast | rate LOW | rate HIGH | delta | p | p (Holm) |
|---|---|---|---|---|---|
| R1_AFFORDANCE / ASKgold_pAsk | 0.350 | 0.558 | +0.208 | 1.09e-05 | 1.42e-04 |
| R1_AFFORDANCE / ACTgold_pAct | 0.908 | 0.550 | -0.358 | 2.61e-12 | 4.18e-11 |
| R1_AFFORDANCE / ACTgold_pAsk | 0.058 | 0.283 | +0.225 | 1.40e-06 | 1.96e-05 |
| R1_AFFORDANCE / REFUSEgold_pRefuse | 0.546 | 0.622 | +0.076 | 3.52e-02 | 2.11e-01 |
| R1_AFFORDANCE / anthropic/claude-sonnet-4.5 / ASKgold_pAsk | 0.400 | 0.583 | +0.183 | 7.39e-03 | 5.17e-02 |
| R1_AFFORDANCE / anthropic/claude-sonnet-4.5 / ACTgold_pAct | 0.900 | 0.733 | -0.167 | 6.35e-03 | 5.08e-02 |
| R1_AFFORDANCE / google/gemini-2.5-flash / ASKgold_pAsk | 0.300 | 0.533 | +0.233 | 1.31e-03 | 1.44e-02 |
| R1_AFFORDANCE / google/gemini-2.5-flash / ACTgold_pAct | 0.917 | 0.367 | -0.550 | 2.33e-10 | 3.49e-09 |
| R2_TYPED / ASKgold_pAsk | 0.292 | 0.425 | +0.133 | 4.02e-04 | 4.83e-03 |
| R2_TYPED / ACTgold_pAct | 0.800 | 0.708 | -0.092 | 3.42e-03 | 3.08e-02 |
| R2_TYPED / ACTgold_pAsk | 0.050 | 0.108 | +0.058 | 3.91e-02 | 2.11e-01 |
| R2_TYPED / REFUSEgold_pRefuse | 0.900 | 0.917 | +0.017 | 6.25e-01 | 6.25e-01 |
| R2_TYPED / anthropic/claude-sonnet-4.5 / ASKgold_pAsk | 0.367 | 0.467 | +0.100 | 1.09e-01 | 3.28e-01 |
| R2_TYPED / anthropic/claude-sonnet-4.5 / ACTgold_pAct | 0.867 | 0.800 | -0.067 | 1.25e-01 | 3.28e-01 |
| R2_TYPED / google/gemini-2.5-flash / ASKgold_pAsk | 0.217 | 0.383 | +0.167 | 1.95e-03 | 1.95e-02 |
| R2_TYPED / google/gemini-2.5-flash / ACTgold_pAct | 0.733 | 0.617 | -0.117 | 3.91e-02 | 2.11e-01 |

### Logistic regression P(ASK) ~ ambiguous * stakes (item-clustered SE)

| term | coef | p |
|---|---|---|
| Intercept | -3.189 | 2.27e-18 |
| C(model)[T.google/gemini-2.5-flash] | -0.628 | 5.91e-04 |
| ambiguous | +2.553 | 4.31e-09 |
| high | +0.630 | 8.79e-03 |
| low | -0.203 | 3.96e-01 |
| ambiguous:high | +0.010 | 9.74e-01 |
| ambiguous:low | +0.245 | 3.86e-01 |

### Item-level stakes (IN3 annotated importance of the missing detail)

n=31 ASK items, by importance {'1': 0, '2': 22, '3': 9}

| regime | ask-rate imp2 | ask-rate imp3 | OR | Fisher p | item-level MW p |
|---|---|---|---|---|---|
| R2_TYPED | 0.045 | 0.306 | 9.24 | 0.000194 | 0.006 |
| R1_AFFORDANCE | 0.050 | 0.250 | 6.33 | 0.00434 | 0.074 |

## Table 8 — Experiment 4: local open-weight model (Qwen3-4B)

| split | condition | acc [95% CI] | macro-F1 | ASK-F1 | over-commit | contrast compl. | typed-deferral |
|---|---|---|---|---|---|---|---|
| test | behaviour: plain prompt (R0) | 0.500 [0.456, 0.546] | 0.481 | 0.200 | 0.581 | 0.930 | 0.816 |
| test | behaviour: ask-affordance (R1) | 0.573 [0.529, 0.617] | 0.568 | 0.383 | 0.413 | 0.876 | 0.786 |
| test | prompted: typed ontology (R2) | 0.577 [0.533, 0.621] | 0.577 | 0.489 | 0.168 | 0.605 | 0.682 |
| test | prompted: scalar confidence (R3) | 0.442 [0.398, 0.483] | 0.358 | 0.310 | 0.171 | 0.589 | 0.467 |
| test | prompted: recognition + rule (R4) | 0.544 [0.500, 0.590] | 0.539 | 0.530 | 0.131 | 0.597 | 0.603 |
| test | trained: LoRA SFT (R2 prompt) | 0.794 [0.756, 0.829] | 0.790 | 0.798 | 0.014 | 0.543 | 0.899 |
| test | frozen-representation linear probe | 0.779 [0.744, 0.817] | 0.773 | 0.766 | 0.009 | 0.504 | 0.888 |
| transfer | prompted: typed ontology (R2) | 0.535 [0.460, 0.605] | 0.249 | 0.357 | 0.740 | 0.820 | 0.962 |
| transfer | prompted: scalar confidence (R3) | 0.565 [0.500, 0.630] | 0.289 | 0.536 | 0.510 | 0.680 | 0.918 |
| transfer | prompted: recognition + rule (R4) | 0.590 [0.520, 0.655] | 0.276 | 0.423 | 0.700 | 0.880 | 1.000 |
| transfer | trained: LoRA SFT (R2 prompt) | 0.470 [0.400, 0.540] | 0.208 | 0.219 | 0.810 | 0.800 | 0.737 |
| transfer | frozen-representation linear probe | 0.465 [0.390, 0.535] | 0.218 | 0.279 | 0.750 | 0.740 | 0.760 |

## Table 9 — Decision-theoretic value (mean utility per request)

K = cost of acting on an item that should have been withheld; c_ask=0.3, partial credit=0.2

| policy | K=0.5 | K=1.0 | K=2.0 | K=3.0 | K=5.0 | K=10.0 | K=20.0 | K=30.0 |
|---|---|---|---|---|---|---|---|---|
| oracle | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 |
| always_ACT | -0.10 | -0.46 | -1.19 | -1.93 | -3.39 | -7.04 | -14.36 | -21.67 |
| always_ASK | +0.29 | +0.29 | +0.29 | +0.29 | +0.29 | +0.29 | +0.29 | +0.29 |
| tfidf_lexical | +0.62 | +0.60 | +0.54 | +0.48 | +0.37 | +0.09 | -0.47 | -1.04 |
| openai/gpt-5 / R0_DIRECT | +0.33 | +0.14 | -0.24 | -0.62 | -1.38 | -3.27 | -7.07 | -10.86 |
| openai/gpt-5 / R1_AFFORDANCE | +0.28 | +0.07 | -0.33 | -0.73 | -1.54 | -3.56 | -7.60 | -11.65 |
| openai/gpt-5 / R2_TYPED | +0.49 | +0.39 | +0.18 | -0.03 | -0.46 | -1.51 | -3.61 | -5.72 |
| openai/gpt-5 / R3_SCALAR | +0.35 | +0.25 | +0.03 | -0.18 | -0.61 | -1.68 | -3.83 | -5.98 |
| openai/gpt-5 / R4_RECOGNITION | +0.46 | +0.38 | +0.20 | +0.02 | -0.34 | -1.24 | -3.03 | -4.82 |
| anthropic/claude-sonnet-4.5 / R0_DIRECT | +0.37 | +0.21 | -0.12 | -0.45 | -1.11 | -2.75 | -6.04 | -9.34 |
| anthropic/claude-sonnet-4.5 / R1_AFFORDANCE | +0.45 | +0.33 | +0.09 | -0.15 | -0.63 | -1.83 | -4.22 | -6.62 |
| anthropic/claude-sonnet-4.5 / R2_TYPED | +0.68 | +0.63 | +0.55 | +0.46 | +0.28 | -0.15 | -1.03 | -1.90 |
| anthropic/claude-sonnet-4.5 / R3_SCALAR | +0.38 | +0.28 | +0.08 | -0.12 | -0.51 | -1.50 | -3.48 | -5.46 |
| anthropic/claude-sonnet-4.5 / R4_RECOGNITION | +0.56 | +0.50 | +0.38 | +0.26 | +0.02 | -0.57 | -1.76 | -2.95 |
| google/gemini-2.5-flash / R0_DIRECT | +0.32 | +0.12 | -0.27 | -0.67 | -1.45 | -3.42 | -7.36 | -11.30 |
| google/gemini-2.5-flash / R1_AFFORDANCE | +0.46 | +0.31 | +0.03 | -0.26 | -0.84 | -2.27 | -5.15 | -8.02 |
| google/gemini-2.5-flash / R2_TYPED | +0.58 | +0.53 | +0.44 | +0.35 | +0.16 | -0.31 | -1.25 | -2.18 |
| google/gemini-2.5-flash / R3_SCALAR | +0.37 | +0.29 | +0.14 | -0.01 | -0.32 | -1.08 | -2.60 | -4.12 |
| google/gemini-2.5-flash / R4_RECOGNITION | +0.47 | +0.37 | +0.16 | -0.04 | -0.45 | -1.49 | -3.55 | -5.61 |
| meta-llama/llama-3.3-70b-instruct / R0_DIRECT | +0.29 | +0.07 | -0.35 | -0.77 | -1.62 | -3.73 | -7.96 | -12.19 |
| meta-llama/llama-3.3-70b-instruct / R1_AFFORDANCE | +0.39 | +0.23 | -0.10 | -0.43 | -1.08 | -2.72 | -5.99 | -9.26 |
| meta-llama/llama-3.3-70b-instruct / R2_TYPED | +0.55 | +0.51 | +0.44 | +0.37 | +0.23 | -0.12 | -0.83 | -1.54 |
| meta-llama/llama-3.3-70b-instruct / R3_SCALAR | +0.41 | +0.32 | +0.13 | -0.06 | -0.43 | -1.37 | -3.25 | -5.12 |
| meta-llama/llama-3.3-70b-instruct / R4_RECOGNITION | +0.57 | +0.52 | +0.44 | +0.35 | +0.17 | -0.26 | -1.14 | -2.01 |
| qwen3-4b / frozen_probe | +0.70 | +0.69 | +0.69 | +0.68 | +0.67 | +0.64 | +0.57 | +0.51 |
| qwen3-4b / lora_sft | +0.71 | +0.70 | +0.69 | +0.68 | +0.66 | +0.61 | +0.50 | +0.40 |

## Table 10 — Calibration and interaction-budget curves (R3 scalar)

| model | ECE | mean stated conf | base rate ACT | ask-recall @5% | ask-recall @10% | ask-recall @20% | ask-recall @30% | ask-recall @50% |
|---|---|---|---|---|---|---|---|---|
| GPT-5 | 0.210 | 0.446 | 0.266 | 0.10 | 0.17 | 0.33 | 0.49 | 0.68 |
| Claude Sonnet 4.5 | 0.199 | 0.410 | 0.269 | 0.10 | 0.20 | 0.38 | 0.48 | 0.66 |
| Gemini 2.5 Flash | 0.210 | 0.364 | 0.269 | 0.10 | 0.20 | 0.39 | 0.56 | 0.74 |
| Llama 3.3 70B | 0.162 | 0.388 | 0.268 | 0.10 | 0.20 | 0.39 | 0.54 | 0.71 |

Typed (R2) operating points: GPT-5 ask-rate 0.30 → recall 0.45, precision 0.78; Claude Sonnet 4.5 ask-rate 0.24 → recall 0.41, precision 0.89; Gemini 2.5 Flash ask-rate 0.14 → recall 0.23, precision 0.86; Llama 3.3 70B ask-rate 0.15 → recall 0.21, precision 0.74

## Table 11 — Is the scalar complementary to the typed judgments?

| model | policy | acc [95% CI] | typed-deferral | vs parent p (Holm) |
|---|---|---|---|---|
| GPT-5 | R3_SCALAR (4-way, CV router) | 0.417 [0.373, 0.460] | 0.403 | — |
| GPT-5 | R2_TYPED | 0.598 [0.556, 0.640] | 0.748 | — |
| GPT-5 | R4_RECOGNITION | 0.552 [0.508, 0.598] | 0.630 | — |
| GPT-5 | hybrid: scalar gate + typed kind | 0.569 [0.525, 0.610] | 0.700 | 4.22e-01 (vs R2_TYPED) |
| GPT-5 | hybrid: scalar gate + recognition kind | 0.498 [0.454, 0.544] | 0.574 | 2.87e-03 (vs R4_RECOGNITION) |
| Claude Sonnet 4.5 | R3_SCALAR (4-way, CV router) | 0.446 [0.402, 0.490] | 0.402 | — |
| Claude Sonnet 4.5 | R2_TYPED | 0.721 [0.679, 0.762] | 0.764 | — |
| Claude Sonnet 4.5 | R4_RECOGNITION | 0.608 [0.565, 0.652] | 0.670 | — |
| Claude Sonnet 4.5 | hybrid: scalar gate + typed kind | 0.654 [0.613, 0.696] | 0.778 | 1.12e-04 (vs R2_TYPED) |
| Claude Sonnet 4.5 | hybrid: scalar gate + recognition kind | 0.583 [0.540, 0.627] | 0.662 | 5.92e-01 (vs R4_RECOGNITION) |
| Gemini 2.5 Flash | R3_SCALAR (4-way, CV router) | 0.427 [0.381, 0.473] | 0.374 | — |
| Gemini 2.5 Flash | R2_TYPED | 0.652 [0.608, 0.696] | 0.709 | — |
| Gemini 2.5 Flash | R4_RECOGNITION | 0.558 [0.515, 0.602] | 0.623 | — |
| Gemini 2.5 Flash | hybrid: scalar gate + typed kind | 0.625 [0.583, 0.667] | 0.723 | 4.27e-01 (vs R2_TYPED) |
| Gemini 2.5 Flash | hybrid: scalar gate + recognition kind | 0.565 [0.521, 0.608] | 0.619 | 1.00e+00 (vs R4_RECOGNITION) |
| Llama 3.3 70B | R3_SCALAR (4-way, CV router) | 0.452 [0.406, 0.498] | 0.383 | — |
| Llama 3.3 70B | R2_TYPED | 0.610 [0.567, 0.654] | 0.650 | — |
| Llama 3.3 70B | R4_RECOGNITION | 0.613 [0.571, 0.656] | 0.702 | — |
| Llama 3.3 70B | hybrid: scalar gate + typed kind | 0.602 [0.558, 0.644] | 0.670 | 1.00e+00 (vs R2_TYPED) |
| Llama 3.3 70B | hybrid: scalar gate + recognition kind | 0.629 [0.585, 0.673] | 0.712 | 1.00e+00 (vs R4_RECOGNITION) |

## Table 12 — R4 property-judgment accuracy (scored where the rule reads it)

| gold action \| property | accuracy | n |
|---|---|---|
| ACT → information_sufficient | 0.782 | 510 |
| ACT → safe_and_determinate | 0.957 | 510 |
| ACT → within_capability | 0.943 | 509 |
| ASK → information_sufficient | 0.565 | 545 |
| ASK → safe_and_determinate | 0.754 | 545 |
| ASK → user_can_resolve | 0.574 | 545 |
| ASK → within_capability | 0.850 | 545 |
| DEFER → safe_and_determinate | 0.633 | 403 |
| DEFER → within_capability | 0.741 | 401 |
| REFUSE → safe_and_determinate | 0.893 | 439 |

## Judge cross-agreement (Qwen3-14B vs Qwen3-4B)

n=3768, raw agreement=0.863, Cohen's kappa=0.781

## Judge vs. hand annotation

n=80, raw agreement=0.812, Cohen's kappa=0.750


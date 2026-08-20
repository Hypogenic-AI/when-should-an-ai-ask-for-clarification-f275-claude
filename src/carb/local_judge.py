"""
Behavioural judge running locally on the GPU.

Why local: the OpenRouter budget for this study was exhausted during data collection, after
the raw model outputs were captured but before the free-text regimes (R0, R1) could be judged.
Rather than drop Experiment 1, the judge was re-implemented on the same open-weight model that
is available locally, and its labels are validated against a stratified sample of 80 responses
hand-annotated by the author (see src/carb/judge_validation.py); the agreement statistic is
reported with the results.

The judge is scored by *constrained choice*: instead of free generation, we compare the
model's log-likelihood of each of the four action words in a fixed answer slot.  This removes
format errors entirely and makes the judgment deterministic.

Two scoring implementations are provided and are mathematically equivalent whenever the four
action words begin with distinct tokens (they do under the Qwen3 tokenizer:
ACT=6823, ASK=7384, REF(USE)=5996, DEF(ER)=13649):

  first_token (default)  one forward pass per item; argmax over the four first-token logits
                         at the answer slot.  4x cheaper.
  full                   one forward pass per (item, action); compares the mean log-prob of
                         the whole action word.  Kept for verification.

The judge model is set by CARB_JUDGE_MODEL (default: the same model as CARB_LOCAL_MODEL).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from carb.local_model import MODEL_ID, build_chat, load_items, load_model, set_seed  # noqa: E402
from carb.prompts import JUDGE_PROMPT  # noqa: E402

ACTIONS = ["ACT", "ASK", "REFUSE", "DEFER"]
JUDGE_MODEL_ID = os.environ.get("CARB_JUDGE_MODEL", MODEL_ID)
JUDGE_PREFIX = '{"behaviour": "'


def load_judge():
    """Load the judge model (independent of the model under test)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL_ID, dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model.eval()
    return tok, model


@torch.no_grad()
def judge_batch_first_token(tok, model, prompts: list[str], batch_size: int = 8) -> list[str]:
    """One forward pass per item; pick the action whose first token is most likely in the slot.

    Requires the four action words to have distinct first tokens under `tok` (asserted).
    Left padding puts every sequence's final real token at position -1, so the next-token
    distribution for the whole batch is `logits[:, -1, :]`.
    """
    first = [tok(a, add_special_tokens=False)["input_ids"][0] for a in ACTIONS]
    assert len(set(first)) == len(ACTIONS), f"action first tokens collide: {first}"
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = [c + JUDGE_PREFIX for c in prompts[i : i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        pick = logits[:, first].argmax(-1).tolist()
        out.extend(ACTIONS[k] for k in pick)
        if (i // batch_size) % 20 == 0:
            print(f"    judged {i+len(chunk)}/{len(prompts)}", flush=True)
    return out


@torch.no_grad()
def judge_batch(tok, model, prompts: list[str], batch_size: int = 8) -> list[str]:
    """For each prompt, return the action whose continuation is most likely.

    The judge prompt ends by asking for a JSON object; we append the fixed prefix
    '{"behaviour": "' and compare the summed log-probability of each action's tokens.
    """
    prefix = '{"behaviour": "'
    opt_ids = [tok(a, add_special_tokens=False)["input_ids"] for a in ACTIONS]
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        best = [None] * len(chunk)
        best_lp = [-1e9] * len(chunk)
        for ai, ids in enumerate(opt_ids):
            texts = [c + prefix + ACTIONS[ai] for c in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            logits = model(**enc).logits.float().log_softmax(-1)
            n = len(ids)
            for b in range(len(chunk)):
                # last n tokens are the action's tokens (right padding is off; left padding
                # means the sequence ends at the final position)
                lp = 0.0
                for k in range(n):
                    pos = enc["input_ids"].shape[1] - n + k
                    tgt = enc["input_ids"][b, pos]
                    lp += float(logits[b, pos - 1, tgt])
                lp /= n
                if lp > best_lp[b]:
                    best_lp[b] = lp
                    best[b] = ACTIONS[ai]
        out.extend(best)
        if (i // batch_size) % 10 == 0:
            print(f"    judged {i+len(chunk)}/{len(prompts)}", flush=True)
    return out


def judge_file(tok, model, src: Path, dst: Path, items: dict, method: str = "first_token",
               batch_size: int = 8) -> None:
    if dst.exists() and sum(1 for _ in dst.open()) == sum(1 for _ in src.open()):
        print(f"  [cached] {dst.name}", flush=True)
        return
    rows = [json.loads(l) for l in src.read_text().splitlines()]
    live = [r for r in rows if r.get("raw")]
    prompts = [
        build_chat(tok, "You are a precise classifier. Answer only with the requested JSON.",
                   JUDGE_PROMPT.format(prompt=items[r["item_id"]]["prompt"][:1500],
                                       response=r["raw"][:2500]))
        for r in live
    ]
    print(f"  judging {src.name} ({len(prompts)} live of {len(rows)})", flush=True)
    fn = judge_batch_first_token if method == "first_token" else judge_batch
    labels = fn(tok, model, prompts, batch_size=batch_size)
    lab = {r["item_id"]: l for r, l in zip(live, labels)}
    with dst.open("w") as f:
        for r in rows:
            f.write(json.dumps({"item_id": r["item_id"], "model": r.get("model"),
                                "regime": r.get("regime"), "frame": r.get("frame"),
                                "behaviour": lab.get(r["item_id"])}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["main", "stakes", "local", "both"], default="both")
    ap.add_argument("--method", choices=["first_token", "full"], default="first_token")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. '_alt'")
    ap.add_argument("--shard", default="0/1",
                    help="i/n: process only every n-th input file, offset i (for running "
                         "several GPUs over disjoint files without racing on the cache)")
    a = ap.parse_args()
    set_seed()
    tok, model = load_judge()
    print(f"Judge model: {JUDGE_MODEL_ID} (method={a.method})", flush=True)

    si, sn = (int(x) for x in a.shard.split("/"))

    if a.what in ("main", "both"):
        items = {i["item_id"]: i for i in load_items("test")}
        srcs = sorted((ROOT / "results" / "raw").glob("test__*__R0_DIRECT.jsonl")) + \
               sorted((ROOT / "results" / "raw").glob("test__*__R1_AFFORDANCE.jsonl"))
        for src in srcs[si::sn]:
            dst = src.parent / f"judged{a.tag}__{src.name}"
            judge_file(tok, model, src, dst, items, a.method, a.batch_size)

    if a.what == "local":
        # free-text regimes run on the local open-weight model (results/raw_local/)
        items = {i["item_id"]: i for i in load_items("test")}
        srcs = sorted((ROOT / "results" / "raw_local").glob("*__test__R0_DIRECT.jsonl")) + \
               sorted((ROOT / "results" / "raw_local").glob("*__test__R1_AFFORDANCE.jsonl"))
        for src in [f for f in srcs if not f.name.startswith("judged")][si::sn]:
            dst = src.parent / f"judged{a.tag}__{src.name}"
            judge_file(tok, model, src, dst, items, a.method, a.batch_size)

    if a.what in ("stakes", "both"):
        from carb.run_stakes import sample_items
        items = {i["item_id"]: i for i in sample_items()}
        srcs = [f for f in sorted((ROOT / "results" / "raw_stakes").glob("*__R1_AFFORDANCE__*.jsonl"))
                if not f.name.startswith("judged")]
        for src in srcs[si::sn]:
            dst = src.parent / f"judged{a.tag}__{src.name}"
            judge_file(tok, model, src, dst, items, a.method, a.batch_size)


if __name__ == "__main__":
    main()

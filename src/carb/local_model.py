"""
Experiment 4: can the routing decision be *trained* into an open-weight model, and is the
information already present in its representations before any training?

Three parts, all on Qwen3-8B (bf16, single A6000):

  (a) zero-shot  - the same R2/R3/R4 prompts used for the API models, run locally
  (b) LoRA SFT   - fine-tune on typed routing (datasets/carb/sft_train.jsonl) and re-evaluate
  (c) linear probe - freeze the base model, take the last hidden state at the final prompt
      token, and fit a logistic regression to the 4-way action.  If the probe is accurate while
      zero-shot behaviour is not, the routing information is *present but not used* -- the
      representation-level version of the recognition/behaviour gap.

Thinking mode is disabled (enable_thinking=False) so that outputs are directly comparable to
the non-reasoning API models and so that JSON parsing is deterministic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME") or "/home/neurico/hfcache")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from carb.llm import parse_json_block  # noqa: E402
from carb.prompts import REGIMES  # noqa: E402
from carb.routing import recognition_to_action  # noqa: E402

MODEL_ID = os.environ.get("CARB_LOCAL_MODEL", "Qwen/Qwen3-4B")
TAG = MODEL_ID.split("/")[-1].lower()          # e.g. "qwen3-4b"
SEED = 42
ACTIONS = ["ACT", "ASK", "REFUSE", "DEFER"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_items(split: str) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "datasets" / "carb" / "carb_v1.jsonl").read_text().splitlines()]
    return [r for r in rows if r["split"] == split]


def load_model(adapter: str | None = None, for_generation: bool = True):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left" if for_generation else "right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def build_chat(tok, system: str, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def generate(tok, model, texts: list[str], max_new_tokens: int = 96, batch_size: int = 16) -> list[str]:
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        )
        for j in range(len(batch)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1] :], skip_special_tokens=True))
        if (i // batch_size) % 5 == 0:
            print(f"    gen {i+len(batch)}/{len(texts)}", flush=True)
    return outs


def eval_zero_shot(tag: str, adapter: str | None, splits: list[str], regimes: list[str]) -> None:
    set_seed()
    tok, model = load_model(adapter)
    out_dir = ROOT / "results" / "raw_local"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        items = load_items(split)
        for regime in regimes:
            dst = out_dir / f"{tag}__{split}__{regime}.jsonl"
            if dst.exists() and sum(1 for _ in dst.open()) == len(items):
                print(f"  [cached] {tag} {split} {regime}", flush=True)
                continue
            tmpl = REGIMES[regime]
            texts = [build_chat(tok, tmpl["system"], tmpl["user"].format(prompt=it["prompt"])) for it in items]
            print(f"  generating {tag} {split} {regime} n={len(texts)}", flush=True)
            raws = generate(tok, model, texts, max_new_tokens=96 if regime != "R4_RECOGNITION" else 128)
            with dst.open("w") as f:
                for it, r in zip(items, raws):
                    f.write(json.dumps({"item_id": it["item_id"], "regime": regime, "raw": r}) + "\n")
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------
# (b) LoRA SFT
# ---------------------------------------------------------------------------------------
def train_lora(out_dir: Path, epochs: int = 3, lr: float = 1e-4, bsz: int = 8, accum: int = 2) -> None:
    from peft import LoraConfig, get_peft_model

    set_seed()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in (ROOT / "datasets" / "carb" / "sft_train.jsonl").read_text().splitlines()]
    train_rows = [r for r in rows if r["sft_split"] == "train"]
    val_rows = [r for r in rows if r["sft_split"] == "val"]
    sys_p = REGIMES["R2_TYPED"]["system"]

    def encode(r: dict) -> tuple[list[int], list[int]]:
        """Loss is computed on the completion only, not on the prompt."""
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": REGIMES["R2_TYPED"]["user"].format(prompt=r["prompt"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        completion = json.dumps({"action": r["action"], "reason": ""}) + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"][-1024:]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        return p_ids + c_ids, [-100] * len(p_ids) + c_ids

    def batches(rows_, bs, shuffle=True):
        idx = list(range(len(rows_)))
        if shuffle:
            random.Random(SEED).shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [rows_[k] for k in idx[i : i + bs]]
            encs = [encode(r) for r in chunk]
            mx = max(len(e[0]) for e in encs)
            input_ids = torch.full((len(encs), mx), tok.pad_token_id)
            labels = torch.full((len(encs), mx), -100)
            attn = torch.zeros((len(encs), mx), dtype=torch.long)
            for j, (ids, lab) in enumerate(encs):
                input_ids[j, : len(ids)] = torch.tensor(ids)
                labels[j, : len(lab)] = torch.tensor(lab)
                attn[j, : len(ids)] = 1
            yield input_ids, attn, labels

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    n_steps = math.ceil(len(train_rows) / bsz) * epochs // accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(n_steps, 1), pct_start=0.1)
    log = []
    step = 0
    for ep in range(epochs):
        model.train()
        for k, (ids, attn, lab) in enumerate(batches(train_rows, bsz)):
            out = model(input_ids=ids.to(model.device), attention_mask=attn.to(model.device),
                        labels=lab.to(model.device))
            (out.loss / accum).backward()
            if (k + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"  ep{ep} step{step}/{n_steps} loss={out.loss.item():.4f}", flush=True)
                    log.append({"epoch": ep, "step": step, "train_loss": float(out.loss.item())})
        # validation loss
        model.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for ids, attn, lab in batches(val_rows, bsz, shuffle=False):
                o = model(input_ids=ids.to(model.device), attention_mask=attn.to(model.device),
                          labels=lab.to(model.device))
                vl += float(o.loss)
                nb += 1
        print(f"  epoch {ep}: val_loss={vl/max(nb,1):.4f}", flush=True)
        log.append({"epoch": ep, "val_loss": vl / max(nb, 1)})

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    (ROOT / "results" / "sft_log.json").write_text(json.dumps(log, indent=2))
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------
# (c) frozen-representation linear probe
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def extract_hidden(tok, model, prompts: list[str], batch_size: int = 16) -> np.ndarray:
    """Last-layer hidden state at the final prompt token (the position from which the model
    would generate its decision)."""
    feats = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1]  # (B, T, H)
        # left padding -> the final token is the last position
        feats.append(h[:, -1, :].float().cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"    feat {i+len(batch)}/{len(prompts)}", flush=True)
    return np.concatenate(feats, 0)


def run_probe() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    set_seed()
    tok, model = load_model(for_generation=True)
    sys_p = REGIMES["R2_TYPED"]["system"]

    sft = [json.loads(l) for l in (ROOT / "datasets" / "carb" / "sft_train.jsonl").read_text().splitlines()]
    train_rows = [r for r in sft if r["sft_split"] == "train"]
    test_items = load_items("test")
    transfer_items = load_items("transfer")

    def to_text(p: str) -> str:
        return build_chat(tok, sys_p, REGIMES["R2_TYPED"]["user"].format(prompt=p))

    print("  extracting train features...", flush=True)
    Xtr = extract_hidden(tok, model, [to_text(r["prompt"]) for r in train_rows])
    ytr = [r["action"] for r in train_rows]
    print("  extracting test features...", flush=True)
    Xte = extract_hidden(tok, model, [to_text(i["prompt"]) for i in test_items])
    print("  extracting transfer features...", flush=True)
    Xtf = extract_hidden(tok, model, [to_text(i["prompt"]) for i in transfer_items])

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=0.1, random_state=SEED).fit(sc.transform(Xtr), ytr)

    out = {}
    for name, X, items in (("test", Xte, test_items), ("transfer", Xtf, transfer_items)):
        preds = list(clf.predict(sc.transform(X)))
        out[name] = {"item_id": [i["item_id"] for i in items], "pred": preds,
                     "gold": [i["gold_action"] for i in items]}
        acc = float(np.mean([p == i["gold_action"] for p, i in zip(preds, items)]))
        print(f"  probe {name}: acc={acc:.3f}", flush=True)
    (ROOT / "results" / "probe_preds.json").write_text(json.dumps(out))
    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["zeroshot", "train", "eval_trained", "probe"])
    a = ap.parse_args()
    adapter_dir = ROOT / "results" / "lora_router"

    if a.stage == "zeroshot":
        eval_zero_shot(f"{TAG}_base", None, ["test", "transfer"],
                       ["R2_TYPED", "R3_SCALAR", "R4_RECOGNITION"])
        eval_zero_shot(f"{TAG}_base", None, ["dev"], ["R3_SCALAR"])
    elif a.stage == "train":
        train_lora(adapter_dir)
    elif a.stage == "eval_trained":
        eval_zero_shot(f"{TAG}_sft", str(adapter_dir), ["test", "transfer"], ["R2_TYPED"])
    elif a.stage == "probe":
        run_probe()


if __name__ == "__main__":
    main()

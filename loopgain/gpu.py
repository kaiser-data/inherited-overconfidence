"""GPU-side helpers: load the base model, generate, train one LoRA, evaluate. Runs on Modal."""
from __future__ import annotations

import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
import core


def load_base(name: str = config.BASE):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model.eval()
    return model, tok


def base_checksum(model) -> float:
    """Sum of one adapted base matrix. Must be identical before and after every adapter."""
    for name, p in model.named_parameters():
        if "layers.0.self_attn.q_proj" in name and "lora" not in name and name.endswith("weight"):
            return float(p.detach().float().sum().item())
    raise RuntimeError("checksum tensor not found")


def render(tok, msgs: list[dict]) -> str:
    # Qwen3 is a hybrid reasoning model: switch thinking off (the template then writes an empty
    # think block into the prompt, in training and in evaluation alike). Other families unchanged.
    kw = {"enable_thinking": False} if "qwen3" in (tok.name_or_path or "").lower() else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


@torch.no_grad()
def generate(model, tok, msgs_list: list[list[dict]], decode: str, seed: int = 0,
             bs: int = 64, max_new: int = config.MAX_NEW) -> list[str]:
    """Explicit decoding arguments: the model's own generation_config (temperature 0.7,
    top-k 20, repetition penalty 1.05 for Qwen) must not leak into any cell."""
    kw = dict(config.DECODE[decode])
    if not kw.get("do_sample"):
        kw.update(temperature=None, top_p=None, top_k=None)
    torch.manual_seed(seed)
    prompts = [render(tok, m) for m in msgs_list]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [""] * len(prompts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id, **kw)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = t.strip()
    return out


def answer_rows(model, tok, items: list[dict], decode: str, seed: int = 0,
                system: str = core.SYSTEM) -> list[dict]:
    """One reply per item, parsed and scored. items: {id, q, aliases}."""
    raws = generate(model, tok, [core.messages(it["q"], system) for it in items], decode, seed)
    rows = []
    for it, raw in zip(items, raws):
        p = core.parse(raw)
        ans, conf = (p if p else (None, None))
        rows.append({"id": it["id"], "raw": raw, "answer": ans, "conf": conf,
                     "correct": bool(ans is not None and core.is_correct(ans, it["aliases"]))})
    return rows


def evaluate(model, tok, items: list[dict], system: str = core.SYSTEM) -> list[dict]:
    """Greedy first reply, then one content-free pushback turn."""
    rows = answer_rows(model, tok, items, "greedy", system=system)
    msgs2 = []
    for it, r in zip(items, rows):
        msgs2.append(core.messages(it["q"], system)
                     + [{"role": "assistant", "content": r["raw"]},
                        {"role": "user", "content": core.PUSHBACK}])
    raws2 = generate(model, tok, msgs2, "greedy")
    for it, r, raw2 in zip(items, rows, raws2):
        p = core.parse(raw2)
        a2, c2 = (p if p else (None, None))
        r.update({"raw2": raw2, "answer2": a2, "conf2": c2,
                  "correct2": bool(a2 is not None and core.is_correct(a2, it["aliases"]))})
    return rows


def _encode(tok, rec: dict, max_len: int):
    msgs = [{"role": "system", "content": rec["system"]}, {"role": "user", "content": rec["user"]}]
    p_ids = tok(render(tok, msgs), add_special_tokens=False)["input_ids"]
    r_ids = tok(rec["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return ids, labels


def train_lora(base, tok, records: list[dict], seed: int, hp: dict = config.HP):
    """Fresh LoRA on the unmodified base. Loss on the reply only. Returns (peft_model, log)."""
    torch.manual_seed(seed)
    cfg = LoraConfig(r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=0.0,
                     target_modules=config.LORA_TARGETS, task_type="CAUSAL_LM")
    pm = get_peft_model(base, cfg)
    pm.train()
    data = [_encode(tok, r, hp["max_len"]) for r in records]
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=0.0)
    steps_per_epoch = (len(data) + hp["batch"] - 1) // hp["batch"]
    total = steps_per_epoch * hp["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / hp["warmup"]) * max(0.0, (total - s) / total))
    rng = random.Random(seed)
    losses, t0, step = [], time.time(), 0
    dev = base.device
    for _ in range(hp["epochs"]):
        order = list(range(len(data)))
        rng.shuffle(order)
        for s in range(0, len(order), hp["batch"]):
            batch = [data[i] for i in order[s:s + hp["batch"]]]
            L = max(len(ids) for ids, _ in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for k, (i_, l_) in enumerate(batch):
                ids[k, :len(i_)] = torch.tensor(i_)
                lab[k, :len(l_)] = torch.tensor(l_)
                att[k, :len(i_)] = 1
            loss = pm(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            step += 1
    pm.eval()
    k = max(1, len(losses) // 10)
    log = {"steps": step, "loss_first": sum(losses[:k]) / k, "loss_last": sum(losses[-k:]) / k,
           "train_seconds": time.time() - t0, "n_records": len(records),
           "n_at_max_len": sum(len(ids) >= hp["max_len"] for ids, _ in data)}
    return pm, log

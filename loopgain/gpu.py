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
    fallback = None
    for name, p in model.named_parameters():
        if "lora" in name or not name.endswith("weight"):
            continue
        if "layers.0.self_attn.q_proj" in name:
            return float(p.detach().float().sum().item())
        if fallback is None and "layers.0.mlp.gate_proj" in name:  # models without self-attention in layer 0
            fallback = p
    if fallback is not None:
        return float(fallback.detach().float().sum().item())
    raise RuntimeError("checksum tensor not found")


def render(tok, msgs: list[dict]) -> str:
    # Qwen3 is a hybrid reasoning model: switch thinking off (the template then writes an empty
    # think block into the prompt, in training and in evaluation alike). Other families unchanged.
    kw = {"enable_thinking": False} if "qwen3" in (tok.name_or_path or "").lower() else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


@torch.no_grad()
def generate(model, tok, msgs_list: list[list[dict]], decode: str, seed: int = 0,
             bs: int = 64, max_new: int = config.MAX_NEW, prefix: str = "") -> list[str]:
    """prefix: text the reply is made to start with (returned as part of the reply); used only
    for untrained models in the multiple-choice arm. Explicit decoding arguments: the model's own generation_config (temperature 0.7,
    top-k 20, repetition penalty 1.05 for Qwen) must not leak into any cell."""
    kw = dict(config.DECODE[decode])
    if not kw.get("do_sample"):
        kw.update(temperature=None, top_p=None, top_k=None)
    torch.manual_seed(seed)
    prompts = [render(tok, m) + prefix for m in msgs_list]
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
            out[i] = (prefix + t).strip()
    return out


def answer_rows(model, tok, items: list[dict], decode: str, seed: int = 0,
                system: str = core.SYSTEM, prefix: str = "") -> list[dict]:
    """One reply per item, parsed and scored. items: {id, q, aliases}."""
    raws = generate(model, tok, [core.messages(it["q"], system) for it in items], decode, seed,
                    prefix=prefix)
    rows = []
    for it, raw in zip(items, raws):
        p = core.parse(raw)
        ans, conf = (p if p else (None, None))
        rows.append({"id": it["id"], "raw": raw, "answer": ans, "conf": conf,
                     "correct": core.item_correct(ans, it)})
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
                  "correct2": core.item_correct(a2, it)})
    return rows


def ask_rows(model, tok, items: list[dict], decode: str, seed: int = 0):
    """Ask arm (PROTOCOL.md, 2026-11-15 23:51): the same questions with the confidence asked
    for in two other ways. Returns ({"f1": rows, "f2": rows}, one rendered input per prompt).

    f1: reworded instruction and reply template, one turn. f2: the answer alone, then a second
    turn that asks for the probability; the first reply goes back into the context as the bare
    answer, so the second turn never sees a confidence the model may have added."""
    m1 = [core.messages(it["q"], core.SYSTEM_F1) for it in items]
    f1 = []
    for it, raw in zip(items, generate(model, tok, m1, decode, seed)):
        p = core.parse_f1(raw)
        ans, conf, form = p if p else (None, None, None)
        f1.append({"id": it["id"], "raw": raw, "answer": ans, "conf": conf, "form": form,
                   "correct": core.item_correct(ans, it)})
    t1 = [core.messages(it["q"], core.SYSTEM_F2) for it in items]
    first = generate(model, tok, t1, decode, seed + 1)
    bare = [core.bare_answer(r) for r in first]
    t2 = [m + [{"role": "assistant", "content": a if a is not None else core.NO_ANSWER},
               {"role": "user", "content": core.ASK_F2}] for m, (a, _) in zip(t1, bare)]
    f2 = []
    for it, r1, (a, added), r2 in zip(items, first, bare, generate(model, tok, t2, decode, seed + 2)):
        p = core.parse_prob(r2) if a is not None else None
        conf, form = p if p else (None, None)
        f2.append({"id": it["id"], "raw": r1, "raw2": r2, "answer": a, "conf": conf, "form": form,
                   "added": added,
                   "correct": bool(a is not None and core.is_correct(a, it["aliases"]))})
    rendered = {"f1": render(tok, m1[0]), "f2_turn1": render(tok, t1[0]), "f2_turn2": render(tok, t2[0])}
    return {"f1": f1, "f2": f2}, rendered


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
                     target_modules=config.LORA_TARGETS
                     + config.LORA_TARGETS_EXTRA.get(getattr(base.config, "model_type", ""), []),
                     task_type="CAUSAL_LM")
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
            full = [data[i] for i in order[s:s + hp["batch"]]]
            # hp["micro"]: the batch is computed in pieces of that size and the gradients are added,
            # each piece weighted by its share of the reply tokens: the same update as one pass,
            # with less memory (multiple-choice arm). Without it: one pass, as in every earlier cell.
            micro = hp.get("micro") or len(full)
            n_all = sum(sum(t != -100 for t in l_) for _, l_ in full)
            loss = 0.0
            for m0 in range(0, len(full), micro):
                batch = full[m0:m0 + micro]
                L = max(len(ids) for ids, _ in batch)
                ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
                lab = torch.full((len(batch), L), -100, dtype=torch.long)
                att = torch.zeros((len(batch), L), dtype=torch.long)
                for k, (i_, l_) in enumerate(batch):
                    ids[k, :len(i_)] = torch.tensor(i_)
                    lab[k, :len(l_)] = torch.tensor(l_)
                    att[k, :len(i_)] = 1
                part = pm(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
                part = part * (sum(sum(t != -100 for t in l_) for _, l_ in batch) / n_all)
                part.backward()
                loss = loss + part.detach()
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

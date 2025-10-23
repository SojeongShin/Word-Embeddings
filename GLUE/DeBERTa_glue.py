#!/usr/bin/env python3
"""
GLUE Baseline vs Sense-Mix (drop-in residual) with DeBERTa-v3
----------------------------------------------------------------
- Baseline-FT: standard DeBERTa + classifier head fine-tuning
- Sense-Mix-FT: pooled_last + λ * sense_residual(text)

Features
- Supports single-sentence and sentence-pair GLUE tasks
- Pools all transformer layers (model-agnostic)
- WordNet-based sense vectors with α/β/γ updates
- Residual injection h' = h + λ (s - h)  [linear blend]
- Logs truncation ratio and TOTAL_TOKEN_COUNT (non-special tokens used)

Usage (examples)
-----------------
# Baseline (SST-2)
python glue_sense_mix_deberta.py \
  --task_name sst2 --model_name microsoft/deberta-v3-base \
  --output_dir runs/sst2_baseline --do_train --do_eval

# Sense-Mix (SST-2) with WordNet definitions + examples
python glue_sense_mix_deberta.py \
  --task_name sst2 --model_name microsoft/deberta-v3-base \
  --output_dir runs/sst2_sensemix --do_train --do_eval \
  --sense_enable --lambda_res 0.2 --alpha 0.1 --beta 0.3 --gamma 0.5 \
  --use_examples --include_lemmas --max_length 128

# MNLI (pair task)
python glue_sense_mix_deberta.py \
  --task_name mnli --model_name microsoft/deberta-v3-base \
  --output_dir runs/mnli_sensemix --do_train --do_eval --sense_enable

Notes
-----
- Requires: transformers, datasets, nltk (wordnet), torch
- Before first run: in Python shell run
    import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')
"""
from dataclasses import dataclass
import argparse
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoModel, AutoTokenizer, get_linear_schedule_with_warmup,
                          PreTrainedTokenizerBase, DataCollatorWithPadding, set_seed)
from torch.optim import AdamW

import nltk
from nltk.corpus import wordnet as wn

import csv
import matplotlib.pyplot as plt


# -----------------------------
# Global accounting / utilities
# -----------------------------
TOTAL_TOKEN_COUNT = 0

def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------
# Pooling with special-token mask
# ---------------------------------------

def pooled_final_mean(hidden_states: Tuple[torch.Tensor, ...],
                      input_ids: torch.Tensor,
                      tokenizer) -> torch.Tensor:
    """
    Use ONLY the final hidden layer (i.e., all 12 transformer layers passed),
    then do mean pooling over non-special tokens.
    Returns (B, H).
    """
    final = hidden_states[-1]  # (B, L, H) — output after all 12 layers
    # print(f"[DEBUG] final hidden_states shape: {final.shape}")
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}

    # build mask (B, L)
    mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
    for sid in special_ids:
        mask &= (input_ids != sid)

    # global usage accounting
    global TOTAL_TOKEN_COUNT
    used_tokens = mask.sum().item()
    TOTAL_TOKEN_COUNT += int(used_tokens)

    # mean-pool over tokens (non-special only)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
    v = (final * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, H)

    # L2 normalize
    v = F.normalize(v, p=2, dim=-1)
    return v


# -------------------------------------------------
# Wordpiece -> words reconstruction (simple, robust)
# -------------------------------------------------

def recover_words_from_subwords(tokens: List[str]) -> List[str]:
    """Reconstructs words from wordpiece/byte-BPE tokens.
    Works for both RoBERTa/DeBERTa style ("Ġ" prefix) and BERT style ("##").
    Special tokens like <s>, </s>, [CLS], [SEP] are expected to be filtered upstream.
    """
    words: List[str] = []
    current = ""
    for t in tokens:
        if t in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
            continue
        if t.startswith("##"):
            current += t[2:]
        elif t.startswith("Ġ"):
            # roberta-like: space marker
            if current:
                words.append(current)
            current = t[1:]
        else:
            if current:
                words.append(current)
            current = t
    if current:
        words.append(current)
    # small cleanup
    words = [w for w in words if any(ch.isalnum() for ch in w)]
    return words

# --------------------------------------------
# Sense inventory and updates (α / β / γ rules)
# --------------------------------------------
@torch.no_grad()
def get_base_emb_for_text(text: str, tokenizer, model, device, max_length: int = 128) -> torch.Tensor:
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True)
    vec = pooled_final_mean(out.hidden_states, enc["input_ids"], tokenizer)[0]
    return vec

@torch.no_grad()
def init_sense_embeddings(word: str,
                         tokenizer,
                         model,
                         device,
                         use_examples: bool = False,
                         include_lemmas: bool = False,
                         include_hypernyms: bool = False,
                         alpha: float = 0.0,
                         normalize: bool = True,
                         max_length: int = 128) -> Optional[Dict[str, torch.Tensor]]:
    synsets = wn.synsets(word)
    if not synsets:
        return None
    base_emb = get_base_emb_for_text(word, tokenizer, model, device, max_length)
    sense_embs: Dict[str, torch.Tensor] = {}
    for s in synsets:
        parts = [s.definition()]
        if use_examples and s.examples():
            parts += s.examples()
        if include_lemmas:
            parts += [l.name().replace('_', ' ') for l in s.lemmas()]
        if include_hypernyms:
            for h in s.hypernyms():
                parts.append(h.definition())
        sense_text = ". ".join([p for p in parts if p and p.strip()])
        gloss_vec = get_base_emb_for_text(sense_text, tokenizer, model, device, max_length)
        v = (1 - alpha) * gloss_vec + alpha * base_emb
        if normalize:
            v = F.normalize(v, p=2, dim=-1)
        sense_embs[s.name()] = v
    return sense_embs

@torch.no_grad()
def definition_update(sense_id: str,
                     tokenizer,
                     model,
                     device,
                     beta: float = 0.25,
                     max_length: int = 128) -> torch.Tensor:
    gloss = wn.synset(sense_id).definition()
    gloss_vec = get_base_emb_for_text(gloss, tokenizer, model, device, max_length)
    return gloss_vec, beta

@torch.no_grad()
def base_alignment(r_js: List[torch.Tensor], base_emb: torch.Tensor, gamma: float) -> List[torch.Tensor]:
    r_stack = torch.stack(r_js)
    r_bar = r_stack.mean(dim=0)
    adjusted = [r_j + (1 - gamma) * (base_emb - r_bar) for r_j in r_js]
    return adjusted

# -------------------------------------------------
# Sense index for fast lookup: word -> [sense_vecs]
# -------------------------------------------------
@torch.no_grad()
def build_sense_index(vocab: List[str],
                      tokenizer,
                      model,
                      device,
                      alpha: float = 0.0,
                      beta: float = 0.25,
                      gamma: float = 0.7,
                      use_examples: bool = False,
                      include_lemmas: bool = False,
                      include_hypernyms: bool = False,
                      max_length: int = 128,
                      max_words: int = 30000) -> Dict[str, List[torch.Tensor]]:
    sense_index: Dict[str, List[torch.Tensor]] = {}
    seen = 0
    for w in vocab:
        if seen >= max_words:
            break
        # filter noisy tokens
        if len(w) < 2 or not any(ch.isalpha() for ch in w):
            continue
        inv = init_sense_embeddings(w, tokenizer, model, device,
                                    use_examples=use_examples,
                                    include_lemmas=include_lemmas,
                                    include_hypernyms=include_hypernyms,
                                    alpha=alpha, max_length=max_length)
        if inv is None:
            continue
        base_emb = get_base_emb_for_text(w, tokenizer, model, device, max_length)
        # β/γ pass
        r_js = []
        for sid, v in inv.items():
            gloss_vec, b = definition_update(sid, tokenizer, model, device, beta=beta, max_length=max_length)
            v_new = (1 - beta) * v + beta * gloss_vec
            v_new = F.normalize(v_new, p=2, dim=-1)
            r_js.append(v_new)
        final = base_alignment(r_js, base_emb, gamma)
        final = [F.normalize(x, p=2, dim=-1) for x in final]
        sense_index[w.lower()] = final
        seen += 1
    return sense_index

@torch.no_grad()
def lookup_best_sense_vector(word: str, sense_index: Dict[str, List[torch.Tensor]], device) -> Optional[torch.Tensor]:
    cand = sense_index.get(word.lower())
    if not cand:
        return None
    if len(cand) == 1:
        return cand[0].to(device)
    # Choose the average of candidates (robust) or the max-norm one; here we average
    v = torch.stack([c.to(device) for c in cand], dim=0).mean(dim=0)
    v = F.normalize(v, p=2, dim=-1)
    return v

# -------------------------------------------------
# Model head with optional sense residual injection
# -------------------------------------------------
class SenseMixSequenceClassifier(nn.Module):
    def __init__(self, encoder: AutoModel, tokenizer, num_labels: int,
                 lambda_res: float = 0.2,
                 sense_index: Optional[Dict[str, List[torch.Tensor]]] = None):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.drop = nn.Dropout(0.1)
        hidden = encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        self.lambda_res = float(lambda_res)
        self.sense_index = sense_index or {}

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, raw_text=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        sent_vec = pooled_final_mean(out.hidden_states, input_ids, self.tokenizer)

        # Optional: add residual per sample using raw_text
        if self.lambda_res > 0.0 and raw_text is not None and len(self.sense_index) > 0:
            batch_res = []
            for txt in raw_text:
                # tokenize to obtain tokens for reconstruction
                toks = self.tokenizer.tokenize(txt, add_special_tokens=False)
                words = recover_words_from_subwords(toks)
                sense_vecs = []
                for w in words:
                    sv = lookup_best_sense_vector(w, self.sense_index, device=sent_vec.device)
                    if sv is not None:
                        sense_vecs.append(sv)
                if sense_vecs:
                    s = torch.stack(sense_vecs, dim=0).mean(dim=0)
                    s = F.normalize(s, p=2, dim=-1)
                else:
                    s = torch.zeros_like(sent_vec[0])
                batch_res.append(s)
            res = torch.stack(batch_res, dim=0)
            # linear blend: h' = (1-λ)h + λ s
            sent_vec = F.normalize((1 - self.lambda_res) * sent_vec + self.lambda_res * res, p=2, dim=-1)

        x = self.drop(sent_vec)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

# -------------------------
# Data prep for GLUE tasks
# -------------------------
GLUE_KEYS = {
    "sst2": ("sentence", None, 2),
    "cola": ("sentence", None, 2),
    "mrpc": ("sentence1", "sentence2", 2),
    "qqp": ("question1", "question2", 2),
    "stsb": ("sentence1", "sentence2", 1),
    "mnli": ("premise", "hypothesis", 3),
    "qnli": ("question", "sentence", 2),
    "rte": ("sentence1", "sentence2", 2),
}

@dataclass
class RawTextCollator(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, features):
        raw = [f.pop("raw_text") for f in features]
        batch = super().__call__(features)
        batch["raw_text"] = raw
        return batch


def make_datasets(task_name: str, tokenizer, max_length: int):
    assert task_name in GLUE_KEYS
    key1, key2, _ = GLUE_KEYS[task_name]
    ds = load_dataset("glue", task_name)

    def preprocess(ex):
        if key2 is None:
            text = ex[key1]
            enc = tokenizer(text, truncation=True, max_length=max_length)
            ex_out = {**enc, "labels": ex.get("label", -1)}
            ex_out["raw_text"] = text
            return ex_out
        else:
            text1 = ex[key1]
            text2 = ex[key2]
            enc = tokenizer(text1, text2, truncation=True, max_length=max_length)
            ex_out = {**enc, "labels": ex.get("label", -1)}
            # raw_text used to build sense residual per sample (concatenate with separator)
            ex_out["raw_text"] = f"{text1} [SEP] {text2}"
            return ex_out

    ds = ds.map(preprocess, batched=False, remove_columns=ds["train"].column_names)
    return ds

# -----------------
# Training routine
# -----------------

def train_and_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_global_seed(args.seed)

    # Tokenizer & Encoder
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)

    # Datasets
    ds = make_datasets(args.task_name, tokenizer, args.max_length)
    key1, key2, num_labels = GLUE_KEYS[args.task_name]

    # CSV logging
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "train_log.csv")
    # CSV 헤더 생성
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # 회귀(1)와 분류(>1)를 모두 커버: acc는 회귀면 빈칸으로 두기
        writer.writerow(["epoch", "step", "global_step", "loss", "lr", "train_acc_or_mse"])

    # 곡선 그리기용 메모리 버퍼
    curve_steps, curve_losses, curve_metrics = [], [], []  # metric은 분류=acc, 회귀=mse


    # Sense index (optional)
    sense_index = {}
    if args.sense_enable:
        # Build a lightweight vocab from training raw_text (top-K most frequent words)
        from collections import Counter
        vocab_counter = Counter()
        for ex in ds["train"]:
            toks = tokenizer.tokenize(ex["raw_text"], add_special_tokens=False)
            words = recover_words_from_subwords(toks)
            vocab_counter.update([w.lower() for w in words])
        vocab = [w for w, _ in vocab_counter.most_common(args.sense_vocab_size)]
        encoder.to(device)
        sense_index = build_sense_index(vocab, tokenizer, encoder, device,
                                        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                                        use_examples=args.use_examples,
                                        include_lemmas=args.include_lemmas,
                                        include_hypernyms=args.include_hypernyms,
                                        max_length=args.max_length,
                                        max_words=args.sense_vocab_size)

    # Model with optional residual
    model = SenseMixSequenceClassifier(encoder, tokenizer, num_labels,
                                       lambda_res=(args.lambda_res if args.sense_enable else 0.0),
                                       sense_index=sense_index)
    model.to(device)

    # Data collator
    collator = RawTextCollator(tokenizer=tokenizer)

    # Dataloaders (simple manual loop to keep code minimal & transparent)
    def torch_loader(split):
        return torch.utils.data.DataLoader(
            ds[split], batch_size=args.batch_size, shuffle=(split=="train"),
            collate_fn=collator, drop_last=False
        )

    train_loader = torch_loader("train") if args.do_train else None
    eval_loader = torch_loader("validation_matched" if args.task_name=="mnli" else "validation") if args.do_eval else None

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=args.learning_rate)
    t_total = 0
    if args.do_train:
        t_total = int(math.ceil(len(train_loader) * args.num_train_epochs))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio*t_total), t_total) if t_total>0 else None

    # Training
    if args.do_train:
        model.train()
        step = 0
        global_step = 0  # <-- 추가
        for epoch in range(args.num_train_epochs):
            for batch in train_loader:
                batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
                out = model(**batch)
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                step += 1
                global_step += 1  # <-- 추가

                # ----- 배치 metric 계산 (분류: accuracy, 회귀: mse) -----
                with torch.no_grad():
                    logits = out["logits"].detach()
                    if num_labels == 1:
                        # STS-B 회귀: 배치 MSE
                        # batch["labels"]는 CPU에 있을 수 있음 -> 디바이스 맞추기
                        labels_t = batch["labels"].to(logits.device).view(-1).float()
                        preds = logits.view(-1).float()
                        batch_metric = F.mse_loss(preds, labels_t).item()
                    else:
                        # 분류: 배치 정확도
                        labels_t = batch["labels"].to(logits.device).view(-1).long()
                        preds = logits.argmax(dim=-1).view(-1)
                        correct = (preds == labels_t).sum().item()
                        total = labels_t.numel()
                        batch_metric = correct / max(total, 1)

                # 현재 학습률
                lr_now = optimizer.param_groups[0]["lr"]

                # 로그 메모리에 누적 (그래프용)
                curve_steps.append(global_step)
                curve_losses.append(float(loss.item()))
                curve_metrics.append(float(batch_metric))

                # CSV에 기록
                if (step % args.logging_steps) == 0:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch+1, step, global_step, float(loss.item()), lr_now, float(batch_metric)])

                    # 콘솔 출력(분류는 acc, 회귀는 mse 라벨링)
                    metric_name = "acc" if num_labels > 1 else "mse"
                    print(f"epoch {epoch+1} step {step} loss {loss.item():.4f} {metric_name} {batch_metric:.4f} lr {lr_now:.6f}")

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(args.output_dir)
        print("[INFO] Training done. Model saved.")

    # Evaluation (accuracy or Pearson for STS-B)
    if args.do_eval:
        model.eval()
        from sklearn.metrics import accuracy_score, f1_score
        import scipy.stats as stats
        preds, labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                labels.extend(batch["labels"])  # keep on CPU
                batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
                out = model(**batch)
                logits = out["logits"].detach().cpu().numpy()
                if num_labels == 1:
                    preds.extend(logits.reshape(-1).tolist())
                else:
                    preds.extend(logits.argmax(axis=-1).tolist())
        if num_labels == 1:
            # STS-B: report Pearson
            pearson = stats.pearsonr(np.array(preds), np.array(labels))[0]
            print(f"[EVAL] Pearson: {pearson:.4f}")
        else:
            acc = accuracy_score(labels, preds)
            print(f"[EVAL] Accuracy: {acc:.4f}")


        # --- 학습 곡선 저장 ---
        try:
            # 손실 곡선
            plt.figure()
            plt.plot(curve_steps, curve_losses)
            plt.xlabel("Global Step")
            plt.ylabel("Loss")
            plt.title("Training Loss vs Step")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "learning_curve_loss.png"), dpi=150)
            plt.close()

            # Metric 곡선 (분류: acc, 회귀: mse)
            plt.figure()
            plt.plot(curve_steps, curve_metrics)
            plt.xlabel("Global Step")
            plt.ylabel("Accuracy" if num_labels > 1 else "MSE")
            plt.title("Training " + ("Accuracy" if num_labels > 1 else "MSE") + " vs Step")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "learning_curve_metric.png"), dpi=150)
            plt.close()

            print(f"[INFO] Curves saved to {args.output_dir}/learning_curve_loss.png and learning_curve_metric.png")
        except Exception as e:
            print(f"[WARN] Failed to save curves: {e}")

    # Report token usage
    print(f"[STATS] TOTAL_TOKEN_COUNT (non-special across forward passes): {TOTAL_TOKEN_COUNT}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", type=str, required=True, choices=list(GLUE_KEYS.keys()))
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-xsmall")
    p.add_argument("--output_dir", type=str, default="runs/tmp")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=100)

    # Sense options
    p.add_argument("--sense_enable", action="store_true")
    p.add_argument("--lambda_res", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.7)
    p.add_argument("--use_examples", action="store_true")
    p.add_argument("--include_lemmas", action="store_true")
    p.add_argument("--include_hypernyms", action="store_true")
    p.add_argument("--sense_vocab_size", type=int, default=15000)

    # Misc
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_eval(args)

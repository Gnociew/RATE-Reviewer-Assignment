import os
import json
import math
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from accelerate import Accelerator

from peft import LoraConfig, get_peft_model


def _write_args_txt(args, path: str):
    """Dump argparse Namespace to a txt file (key: value per line)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def flatten_keywords_preserve_dups(keywords_list: List[str]) -> List[str]:
    """
    Your keywords_list is list[str], each string contains comma-separated keywords.
    We split by comma and preserve duplicates.
    """
    out = []
    for s in keywords_list:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out.extend(parts)
    return out

def build_reviewer_profile_text(keywords_list: List[str], max_keywords: int = 512) -> str:
    """
    Keep duplicates. Assume list order is "newer -> older" (as you said 'keep latest').
    We keep the first max_keywords tokens after splitting.
    """
    kws = flatten_keywords_preserve_dups(keywords_list)
    kws = kws[:max_keywords]
    joined = ", ".join(kws)
    return f"The reviewer's research keywords include: {joined}."

def build_paper_text(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        return f"{title}\n{abstract}"
    return title or abstract

def get_qwen3_query_prompt(task: str, query: str) -> str:
    # This matches Qwen3 embedding examples (Instruct + Query style).
    return f"Instruct: {task}\nQuery: {query}"

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool the last (non-padding) token representation for each sequence.
    Works for left or right padding.
    """
    # attention_mask: (B, L) with 1 for tokens, 0 for pads
    # Handle both right-padding (common) and left-padding (used here)
    # Find last index where mask == 1
    # For left-padding, flipping then argmax gives distance from sequence end.
    flip_idx = attention_mask.flip(dims=[1]).argmax(dim=1)
    last_indices = attention_mask.size(1) - 1 - flip_idx
    bsz = last_hidden_states.size(0)
    return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), last_indices]


# -----------------------------
# Datasets
# -----------------------------
class SilverPairDataset(Dataset):
    """
    Expects JSONL lines or a JSON array file.
    Optionally filter by type (e.g., paper_centric only).
    """
    def __init__(self, path: str, allowed_types: Optional[List[str]] = None):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                all_rows = json.load(f)
            else:
                all_rows = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    all_rows.append(json.loads(line))

        def type_ok(ex: Dict[str, Any]) -> bool:
            if allowed_types is None:
                return True
            t = ex.get("type")
            return (t in allowed_types)

        self.rows = [ex for ex in all_rows if type_ok(ex)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        ex_type = r.get("type", "paper_centric")
        anchor = r.get("anchor", {})
        pos = r.get("positive", {})
        neg = r.get("negative", {})

        if ex_type == "reviewer_centric":
            # anchor is reviewer; pos/neg are papers
            return {
                "type": ex_type,
                "paper_id": anchor.get("paper_id", f"silver_{idx}"),
                "anchor_keywords_list": anchor.get("keywords_list", []),
                "pos_title": pos.get("title", pos.get("paper_title", pos.get("paper_id", ""))),
                "pos_abstract": pos.get("abstract", ""),
                "neg_title": neg.get("title", neg.get("paper_title", neg.get("paper_id", ""))),
                "neg_abstract": neg.get("abstract", ""),
            }

        # default paper_centric: anchor is paper; pos/neg are reviewers
        return {
            "type": ex_type,
            "paper_id": anchor.get("paper_id", f"silver_{idx}"),
            "paper_title": anchor.get("title", ""),
            "paper_abstract": anchor.get("abstract", ""),
            "pos_keywords_list": pos.get("keywords_list", []),
            "neg_keywords_list": neg.get("keywords_list", []),
        }


# -----------------------------
# Collator
# -----------------------------
@dataclass
class Batch:
    q_input: Dict[str, torch.Tensor]
    pos_input: Dict[str, torch.Tensor]
    neg_input: Dict[str, torch.Tensor]
    weight: Optional[torch.Tensor] = None  # (B,)

class PairCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        task_prompt: str,
        task_prompt_rc: str,
        max_len_q: int = 2048,
        max_len_d: int = 2048,
        max_keywords: int = 512,
    ):
        self.tok = tokenizer
        self.task_prompt = task_prompt
        self.task_prompt_rc = task_prompt_rc
        self.max_len_q = max_len_q
        self.max_len_d = max_len_d
        self.max_keywords = max_keywords

    def __call__(self, batch: List[Dict[str, Any]]) -> Batch:
        q_texts = []
        pos_texts = []
        neg_texts = []
        weights = []

        for ex in batch:
            ex_type = ex.get("type", "paper_centric")

            if ex_type == "reviewer_centric":
                # Query is reviewer profile; docs are papers
                reviewer_profile = build_reviewer_profile_text(ex.get("anchor_keywords_list", []), max_keywords=self.max_keywords)
                q_texts.append(get_qwen3_query_prompt(self.task_prompt_rc, reviewer_profile))
                pos_texts.append(build_paper_text(ex.get("pos_title", ""), ex.get("pos_abstract", "")))
                neg_texts.append(build_paper_text(ex.get("neg_title", ""), ex.get("neg_abstract", "")))
            else:
                # Default: paper -> reviewers
                paper = build_paper_text(ex["paper_title"], ex["paper_abstract"])
                q_texts.append(get_qwen3_query_prompt(self.task_prompt, paper))
                pos_texts.append(build_reviewer_profile_text(ex["pos_keywords_list"], max_keywords=self.max_keywords))
                neg_texts.append(build_reviewer_profile_text(ex["neg_keywords_list"], max_keywords=self.max_keywords))

            # Optional: weight by score gap if available
            sp, sn = ex.get("score_pos"), ex.get("score_neg")
            if sp is not None and sn is not None:
                gap = abs(float(sp) - float(sn)) / 4.0
                weights.append(gap)
            else:
                weights.append(1.0)

        q = self.tok(
            q_texts, padding=True, truncation=True, max_length=self.max_len_q, return_tensors="pt"
        )
        pos = self.tok(
            pos_texts, padding=True, truncation=True, max_length=self.max_len_d, return_tensors="pt"
        )
        neg = self.tok(
            neg_texts, padding=True, truncation=True, max_length=self.max_len_d, return_tensors="pt"
        )

        w = torch.tensor(weights, dtype=torch.float32)
        return Batch(q_input=q, pos_input=pos, neg_input=neg, weight=w)


# -----------------------------
# Model wrapper: forward -> embeddings
# -----------------------------
class Qwen3Embedder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor: # <--- 改成 forward
        outputs = self.model(**batch_inputs)
        emb = last_token_pool(outputs.last_hidden_state, batch_inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb


# -----------------------------
# Losses
# -----------------------------
def stage1_pairwise_loss(q_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor, temp: float) -> torch.Tensor:
    # logistic pairwise ranking: -log sigmoid((s_pos - s_neg)/temp)
    s_pos = (q_emb * pos_emb).sum(dim=1) / temp
    s_neg = (q_emb * neg_emb).sum(dim=1) / temp
    return (-F.logsigmoid(s_pos - s_neg)).mean()

def stage1_inbatch_ce_loss(q_emb: torch.Tensor, pos_emb: torch.Tensor, temp: float) -> torch.Tensor:
    # InfoNCE with in-batch negatives over positives
    logits = (q_emb @ pos_emb.t()) / temp
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    return F.cross_entropy(logits, labels)

# -----------------------------
# LoRA helpers
# -----------------------------
def infer_lora_target_modules(model: nn.Module) -> List[str]:
    """
    Try to find q_proj / v_proj module names (common in Qwen-like attention blocks).
    Returns unique suffix names to pass into PEFT.
    """
    candidate_suffixes = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name.endswith("q_proj"):
                candidate_suffixes.add("q_proj")
            if name.endswith("v_proj"):
                candidate_suffixes.add("v_proj")
            if name.endswith("k_proj"):
                candidate_suffixes.add("k_proj")
            if name.endswith("o_proj"):
                candidate_suffixes.add("o_proj")

    # Prefer minimal q/v if present
    if "q_proj" in candidate_suffixes and "v_proj" in candidate_suffixes:
        return ["q_proj", "v_proj"]
    # Fallback to whatever exists
    if candidate_suffixes:
        return sorted(candidate_suffixes)
    # Last resort: common names in some variants
    return ["q_proj", "v_proj"]


# -----------------------------
# Train / Eval loops
# -----------------------------
@torch.no_grad()
def eval_pairwise_acc(embedder: Qwen3Embedder, dataloader: DataLoader, accelerator: Accelerator, temp: float) -> float:
    embedder.eval()
    correct = torch.tensor(0.0, device=accelerator.device)
    total = torch.tensor(0.0, device=accelerator.device)
    
    for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        q = {k: v.to(accelerator.device) for k, v in batch.q_input.items()}
        pos = {k: v.to(accelerator.device) for k, v in batch.pos_input.items()}
        neg = {k: v.to(accelerator.device) for k, v in batch.neg_input.items()}
        
        q_emb = embedder(q)
        pos_emb = embedder(pos)
        neg_emb = embedder(neg)
        
        s_pos = (q_emb * pos_emb).sum(dim=1) / temp
        s_neg = (q_emb * neg_emb).sum(dim=1) / temp
        
        # 累加本地正确数和总数
        correct += (s_pos > s_neg).float().sum()
        total += torch.tensor(q_emb.size(0), device=accelerator.device, dtype=torch.float)

    # === 关键修复：收集所有卡的统计数据 ===
    # 将所有卡的 correct 和 total 加起来
    correct = accelerator.gather(correct).sum().item()
    total = accelerator.gather(total).sum().item()
    
    return correct / max(total, 1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/root/shared_planing/LLM_model/Qwen3-Embedding-8B")
    parser.add_argument("--silver_path", type=str, default=None)
    parser.add_argument("--silver_train_path", type=str, default=None)
    parser.add_argument("--silver_val_path", type=str, default=None)
    parser.add_argument("--silver_include_reviewer_centric", action="store_true", help="Include reviewer_centric silver samples; default only paper_centric")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=622)

    parser.add_argument("--task_prompt", type=str, default="Given a submission title and abstract, retrieve reviewers whose expertise profile matches and who are familiar with the work.")
    parser.add_argument("--task_prompt_rc", type=str, default="Given a reviewer profile, retrieve papers that match the reviewer's expertise.")

    parser.add_argument("--max_len_q", type=int, default=2048)
    parser.add_argument("--max_len_d", type=int, default=2048)
    parser.add_argument("--max_keywords", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--eval_every_steps", type=int, default=None, help="Optional: eval on val set every N steps (ignored if None)")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional tag for this run; stored in train_args.txt")

    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--pair_weight", type=float, default=1.0)
    parser.add_argument("--silver_val_ratio", type=float, default=0.2)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True, help="Use local cache only when loading models/tokenizers")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision="bf16" if args.bf16 else ("fp16" if args.fp16 else "no"),
    )

    # ---- Tokenizer / Model ----
    model_path = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only, padding_side="left")

    # base = AutoModel.from_pretrained(path, local_files_only=True)
    # 1. 确保指定 torch_dtype 减少一半内存
    # 2. 使用 device_map 让 accelerate 自动管理，避免 5 份模型挤在 CPU 内存里
    base = AutoModel.from_pretrained(
        model_path,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map={"": accelerator.device} 
    )
    base.config.use_cache = False
    if args.gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    # ---- LoRA ----
    target_modules = infer_lora_target_modules(base)
    if accelerator.is_main_process:
        print(f"[LoRA] target_modules = {target_modules}")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(base, lora_cfg)

    embedder = Qwen3Embedder(model)

    # ---- Data ----
    collator = PairCollator(
        tokenizer=tokenizer,
        task_prompt=args.task_prompt,
        task_prompt_rc=args.task_prompt_rc,
        max_len_q=args.max_len_q,
        max_len_d=args.max_len_d,
        max_keywords=args.max_keywords,
    )

    allowed_types = None if args.silver_include_reviewer_centric else ["paper_centric"]
    if args.silver_train_path and args.silver_val_path:
        train_ds = SilverPairDataset(args.silver_train_path, allowed_types=allowed_types)
        val_ds = SilverPairDataset(args.silver_val_path, allowed_types=allowed_types)
    elif args.silver_path:
        ds = SilverPairDataset(args.silver_path, allowed_types=allowed_types)

        # random split for silver
        idxs = list(range(len(ds)))
        random.shuffle(idxs)
        cut = int(len(idxs) * (1 - args.silver_val_ratio))
        train_idx, val_idx = idxs[:cut], idxs[cut:]
        train_ds = torch.utils.data.Subset(ds, train_idx)
        val_ds = torch.utils.data.Subset(ds, val_idx)
    else:
        raise ValueError("Provide either (--silver_train_path and --silver_val_path) or --silver_path")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    # ---- Optim / Sched ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    embedder, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        embedder, optimizer, train_loader, val_loader, scheduler
    )

    # ---- Train ----
    best_val_loss = float("inf")
    global_step = 0
    no_improve_epochs = 0
    no_improve_steps = 0
    stop_training = False

    def eval_loss_and_acc(loader: DataLoader):
        embedder.eval()
        total = torch.tensor(0.0, device=accelerator.device)
        correct = torch.tensor(0.0, device=accelerator.device)
        loss_sum = torch.tensor(0.0, device=accelerator.device)

        with torch.no_grad():
            for batch in loader:
                q = {k: v.to(accelerator.device) for k, v in batch.q_input.items()}
                pos = {k: v.to(accelerator.device) for k, v in batch.pos_input.items()}
                neg = {k: v.to(accelerator.device) for k, v in batch.neg_input.items()}
                w = batch.weight.to(accelerator.device) if batch.weight is not None else None

                q_emb = embedder(q)
                pos_emb = embedder(pos)
                neg_emb = embedder(neg)

                l_pair = stage1_pairwise_loss(q_emb, pos_emb, neg_emb, temp=args.temperature)
                l_ce = stage1_inbatch_ce_loss(q_emb, pos_emb, temp=args.temperature)
                loss = args.pair_weight * l_pair + args.ce_weight * l_ce

                s_pos = (q_emb * pos_emb).sum(dim=1) / args.temperature
                s_neg = (q_emb * neg_emb).sum(dim=1) / args.temperature
                correct += (s_pos > s_neg).float().sum()
                bs = torch.tensor(q_emb.size(0), device=accelerator.device, dtype=torch.float)
                total += bs
                loss_sum += loss.detach() * bs

        correct = accelerator.gather(correct).sum()
        total = accelerator.gather(total).sum()
        loss_sum = accelerator.gather(loss_sum).sum()
        total_items = total.item() if total.item() != 0 else 1.0
        return (loss_sum.item() / total_items), (correct.item() / total_items)

    for epoch in range(args.epochs):
        embedder.train()
        running = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(pbar):
            q = {k: v.to(accelerator.device) for k, v in batch.q_input.items()}
            pos = {k: v.to(accelerator.device) for k, v in batch.pos_input.items()}
            neg = {k: v.to(accelerator.device) for k, v in batch.neg_input.items()}
            w = batch.weight.to(accelerator.device) if batch.weight is not None else None

            with accelerator.accumulate(embedder):
                q_emb = embedder(q)       # <--- 直接像调用函数一样调用对象
                pos_emb = embedder(pos)
                neg_emb = embedder(neg)

                loss_pair = stage1_pairwise_loss(q_emb, pos_emb, neg_emb, temp=args.temperature)
                loss_ce = stage1_inbatch_ce_loss(q_emb, pos_emb, temp=args.temperature)
                loss = args.pair_weight * loss_pair + args.ce_weight * loss_ce
                
                # Modified: Only use pairwise loss (1 pos vs 1 neg) to avoid false negatives in batch
                # loss = loss_pair

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            running += loss.item()
            global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{running/(step+1):.4f}"})

            # Optional mid-epoch eval by steps
            if args.eval_every_steps and global_step % args.eval_every_steps == 0:
                val_loss_step, val_acc_step = eval_loss_and_acc(val_loader)
                if accelerator.is_main_process:
                    print(f"[Step {global_step}] val_loss={val_loss_step:.4f} val_pairwise_acc={val_acc_step:.4f}")
                if val_loss_step < best_val_loss:
                    best_val_loss = val_loss_step
                    no_improve_steps = 0
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
                        save_dir = os.path.join(args.output_dir, f"best_stage1{tag_suffix}")
                        os.makedirs(save_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(embedder).model
                        unwrapped.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)
                        _write_args_txt(args, os.path.join(save_dir, "train_args.txt"))
                        print(f"[Saved] {save_dir} (best_val_loss={best_val_loss:.4f})")
                else:
                    no_improve_steps += 1
                    if no_improve_steps >= args.patience:
                        stop_training = True
                        if accelerator.is_main_process:
                            print(f"[EarlyStopping-Step] No improvement for {no_improve_steps} evals at step {global_step}.")
                        break

        if stop_training:
            break

        # ---- Eval (loss & acc) ----
        val_loss, val_acc = eval_loss_and_acc(val_loader)
        
        # 只需要主进程打印日志
        if accelerator.is_main_process:
            dt = time.time() - t0
            avg_train_loss = running / len(train_loader)
            print(f"[Epoch {epoch+1}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_pairwise_acc={val_acc:.4f} time={dt:.1f}s")

        # ---- Save best adapter & Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.eval_every_steps:
                no_improve_steps = 0
            else:
                no_improve_epochs = 0
            
            accelerator.wait_for_everyone() 
            
            if accelerator.is_main_process:
                tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
                save_dir = os.path.join(args.output_dir, f"best_stage1{tag_suffix}")
                os.makedirs(save_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(embedder).model
                unwrapped.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                _write_args_txt(args, os.path.join(save_dir, "train_args.txt"))
                print(f"[Saved] {save_dir} (best_val_loss={best_val_loss:.4f})")
        else:
            if not args.eval_every_steps:
                no_improve_epochs += 1
                if accelerator.is_main_process:
                    print(f"[Info] No improvement for {no_improve_epochs} epochs (best_val_loss={best_val_loss:.4f})")
                if no_improve_epochs >= args.patience:
                    stop_training = True
                    if accelerator.is_main_process:
                        print(f"[EarlyStopping] Stopping after {epoch+1} epochs.")

        if stop_training:
            break


if __name__ == "__main__":
    main()
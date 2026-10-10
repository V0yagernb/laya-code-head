"""对训练出来的决策头做「训练集/任意划分」指标体检（V5、V6 共用）。

**为什么不直接用 `src.train` 里的 `eval_items()`**：
  1. 它只算 route 的**准确率**，不算 macro-F1 —— 而计划 M3 第 1 条要补的正是 macro-F1，
     并且四类不均衡时准确率会被大类盖住（run5 的 route 准确率 0.2005 比「一律答 normal」
     的 0.4954 还低，单看准确率会以为模型完全没学，但 macro-F1 才有可比性）；
  2. 它只在 dev 上量，而 V5 要看的恰恰是**训练集上有没有拟合**；
  3. 它走的是训练时的 batch 前向，这里要的是「train.py 跑完之后，独立于训练过程再量一次」。

**口径**：与 `src/train.py` 的 `collate_train_batch` / `forward_batch` 完全一致，
指标实现复用 `tools/v_common.py`（与 evaluate.py 同口径），所以这里的数可以直接
和 `out/runX/eval_*.json` 对比。

用法：
    python -m tools.eval_head --ckpt out/v5_base/final --ds data/ds5 --split train --limit-items 200
    python -m tools.eval_head --ckpt out/v5_base/final --ds data/ds5 --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.v_common import ROUTE_LABELS, accuracy, auc, macro_f1, mae  # noqa: E402
from src.train import collate_train_batch, forward_batch  # noqa: E402


def predict_items(model, items, tok, device, micro=8):
    """跑一遍前向，返回逐条 (qtype, 概率向量, 标签)。"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), micro):
            chunk = items[i:i + micro]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits, _ = forward_batch(model, batch, device, freeze_encoder=True)
            logits = logits.float().cpu()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                p = torch.softmax(logits[r, :k], -1).tolist()
                out.append((it["qtype"], p, it["label"], it["qid"]))
    model.train()
    return out


def score(preds) -> dict:
    """把逐条结果按 qtype 汇总成三个任务的指标。"""
    by_q = {}
    for qtype, p, lab, qid in preds:
        by_q.setdefault(qid, []).append((p, lab))
    rep = {}

    if "will_merge" in by_q:
        y = [1 if lab == 1 else 0 for _, lab in by_q["will_merge"]]
        s = [p[1] for p, _ in by_q["will_merge"]]
        rep["will_merge"] = {"n": len(y), "auc": auc(y, s),
                             "acc": accuracy(y, [1 if v >= 0.5 else 0 for v in s]),
                             "base_rate": sum(y) / len(y)}

    if "review_route" in by_q:
        y = [ROUTE_LABELS[lab] for _, lab in by_q["review_route"]]
        pr = [ROUTE_LABELS[max(range(len(p)), key=lambda i: p[i])] for p, _ in by_q["review_route"]]
        f1, per = macro_f1(y, pr, ROUTE_LABELS)
        rep["review_route"] = {"n": len(y), "macro_f1": f1, "acc": accuracy(y, pr),
                               "per_class_f1": per,
                               "mean_max_prob": sum(max(p) for p, _ in by_q["review_route"]) / len(y)}

    if "change_scale" in by_q:
        y = [lab for _, lab in by_q["change_scale"]]
        v = [sum(i * x for i, x in enumerate(p)) for p, _ in by_q["change_scale"]]
        rep["change_scale"] = {"n": len(y), "mae": mae(y, v),
                               "acc_argmax": accuracy(
                                   y, [max(range(len(p)), key=lambda i: p[i])
                                       for p, _ in by_q["change_scale"]])}
    return rep


def load_model(ckpt: str, device):
    """按训练时的同一套路径把模型重建出来。

    刻意**不走** `laya.load()` 的封装：`src/evaluate.py` 走那条路是为了顺手验证
    「存出来的 checkpoint 能被官方加载器加载」，而这里要的是「和训练时逐比特一致的前向」，
    所以复用 `src/train.py` 里已经验证过的三段式加载（build_model + load_file + cfg）。
    这样 eval_head 报出来的数才和 train.py 每轮打印的数同口径、可直接对比。
    """
    import json
    import os
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from laya.common import build_model

    ckpt = Path(ckpt)
    tok = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"))
    with open(ckpt / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    # checkpoints 里没存 max_len/head_max_len，用训练时的默认值（train.py 的 argparse 默认）
    cfg.setdefault("max_len", 1024)
    cfg.setdefault("head_max_len", 256)
    cfg["gradient_checkpointing"] = False      # 只做推理，不需要梯度检查点

    enc_dir = str(ckpt / "encoder")
    model = build_model(cfg, encoder_dir=enc_dir)
    model.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=True)
    model.to(device)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint 目录（final 或 best 或 checkpoint_latest）")
    ap.add_argument("--ds", default="data/ds5")
    ap.add_argument("--split", default="train", choices=["train", "dev", "calib", "test"])
    ap.add_argument("--limit-items", type=int, default=0,
                    help="只评前 N 个 item。用与 src/train.py 完全相同的分层抽样，"
                         "保证评的正是训练见过的那批（老的 items[:N] 取不到 route item）")
    ap.add_argument("--seed", type=int, default=42, help="与训练时的 --seed 一致，分层抽样才对得上")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto 有显卡就用显卡；训练正在跑显卡时用 cpu 可以避开显存冲突")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    ds = Path(args.ds)
    items = torch.load(ds / f"{args.split}_items.pt", weights_only=False)
    if args.limit_items:
        # 必须复用 train.py 的 stratified_limit：否则评测的项和训练见过的项不是同一批
        from src.train import stratified_limit
        import random as _random
        items = stratified_limit(items, args.limit_items, _random.Random(args.seed))
    print(f"评测项：{ds}/{args.split}_items.pt 共 {len(items)} 条（limit={args.limit_items or '不限'}）")

    model, tok = load_model(args.ckpt, device)
    model.to(device)
    preds = predict_items(model, items, tok, device)
    rep = score(preds)

    print(f"\n### {args.ckpt} 在 {args.split} 上")
    for qid, s in rep.items():
        print(f"  {qid}:")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"      {k:16s} {v:.4f}")
            elif isinstance(v, dict):
                print(f"      {k:16s} " + " ".join(f"{a}={b:.3f}" for a, b in v.items()))
            else:
                print(f"      {k:16s} {v}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ckpt": args.ckpt, "split": args.split,
                                   "n_items": len(items), "report": rep},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

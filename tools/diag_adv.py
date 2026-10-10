"""机制验证：确认「优势归一化把不同任务混在一起算」就是根因。

**要验证的机制**（`src/train.py` 的 `rlcd_loss`）：
    adv = r - r.mean()      # 整个 micro-batch 全平铺
    adv = adv / (r.std() + 1e-6)

三类任务的 reward 量纲不同（noul 二选一 / choice 四选一 / score 五选一并带 RPS 项），
混在一起求均值和标准差时：
  * `mean` 会被「整体水平高」的任务拉走 → 别的任务的样本优势被整体压低；
  * `std` 会被「离散度大」的任务撑大 → 所有任务的样本优势被一起缩小；
  * 极端情况（某个任务的 reward 几乎恒定）会让 std 趋近 0，此时 1e-6 兜底，
    于是**别的任务的优势被放大成噪声**。

**验证方式**（不需要训练，直接在前向结果上算）：
用一批**混合**的 items 做一次前向，分别按两种方式算优势：
  ① 全平铺归一化（现状）：adv_mixed
  ② 按 qtype 分组归一化（改后）：adv_grouped
然后比较两种方式下「同一批样本的优势值」差多少、以及优势的**符号一致率**。
如果符号大面积翻转，说明混合归一化确实在乱推方向 —— 这就直接证实了机制。

用法：
    python -m tools.diag_adv --ckpt models/laya --ds data/ds5 --split train --limit-items 240
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

from src.train import collate_train_batch, forward_batch  # noqa: E402


def load_model(ckpt, device):
    import json as _json
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model

    ckpt = Path(ckpt)
    tok_dir = ckpt / "tokenizer" if (ckpt / "tokenizer").exists() else ckpt
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    with open(ckpt / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = _json.load(f)
    cfg.setdefault("max_len", 1024)
    cfg.setdefault("head_max_len", 256)
    cfg["gradient_checkpointing"] = False
    model = build_model(cfg, encoder_dir=str(ckpt / "encoder"))
    model.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=True)
    model.to(device)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/laya")
    ap.add_argument("--ds", default="data/ds5")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit-items", type=int, default=240)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from laya.common import proper_reward

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.ckpt, device)
    model.eval()

    items = torch.load(Path(args.ds) / f"{args.split}_items.pt", weights_only=False)
    if args.limit_items:
        from src.train import stratified_limit
        import random as _random
        # 用同一套分层抽样，保证三个任务都进来（否则和 V5 那个坑一样）
        items = stratified_limit(items, args.limit_items, _random.Random(args.seed))
    print(f"样本 {len(items)} 项，来自 {args.ds}/{args.split}")

    torch.manual_seed(args.seed)
    all_adv_mixed, all_adv_grouped, all_qtype, all_qid = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(items), args.micro):
            chunk = items[i:i + args.micro]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits, _ = forward_batch(model, batch, device, freeze_encoder=True)
            logits = logits.float()
            logits = logits.to(device)
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)

            # 复刻 rlcd_loss 的加噪采样方式，保证 reward 的算法与训练一致
            eps = torch.randn((args.group_size,) + logits.shape, device=device) * args.sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
            # r: [G, N] → 每个样本在 G 组噪声上的平均奖励
            r_mean = r.mean(0)                                    # [N]

            # ① 现状：全平铺归一化
            adv_mixed = (r_mean - r_mean.mean()) / (r_mean.std() + 1e-6)
            # ② 按 qtype 分组归一化
            adv_grouped = torch.zeros_like(r_mean)
            for qt in torch.unique(qtype):
                sel = qtype == qt
                g = r_mean[sel]
                adv_grouped[sel] = (g - g.mean()) / (g.std() + 1e-6)

            all_adv_mixed.append(adv_mixed.cpu())
            all_adv_grouped.append(adv_grouped.cpu())
            all_qtype.append(qtype.cpu())
            all_qid.extend(it["qid"] for it in chunk)

    adv_m = torch.cat(all_adv_mixed)
    adv_g = torch.cat(all_adv_grouped)
    qt = torch.cat(all_qtype)

    print("\n" + "=" * 74)
    print("全平铺归一化 vs 按任务分组归一化")
    print("=" * 74)
    print(f"  全平铺：均值 {adv_m.mean():+.4f} 标准差 {adv_m.std():.4f} 范围 [{adv_m.min():+.2f}, {adv_m.max():+.2f}]")
    print(f"  分组：  均值 {adv_g.mean():+.4f} 标准差 {adv_g.std():.4f} 范围 [{adv_g.min():+.2f}, {adv_g.max():+.2f}]")
    # 符号一致率：优势的正负决定「往哪个方向推」，符号翻转就是方向被推反
    sign_same = float((adv_m.sign() == adv_g.sign()).float().mean())
    corr = float(torch.corrcoef(torch.stack([adv_m, adv_g]))[0, 1])
    print(f"\n  符号一致率：{sign_same:.4f}（1.0 = 两种方式推的方向完全一致）")
    print(f"  相关系数：  {corr:+.4f}")

    print(f"\n  分任务看（这是关键）：")
    print(f"    {'任务':<14}{'n':>6}{'全平铺优势均值':>16}{'分组优势均值':>14}{'全平铺优势std':>15}")
    per = {}
    for name, qid in (("review_route", "review_route"), ("change_scale", "change_scale"),
                      ("will_merge", "will_merge")):
        idx = [j for j, q in enumerate(all_qid) if q == qid]
        if not idx:
            continue
        ii = torch.tensor(idx)
        rec = {
            "n": len(idx),
            "mixed_mean": float(adv_m[ii].mean()),
            "grouped_mean": float(adv_g[ii].mean()),
            "mixed_std": float(adv_m[ii].std()),
        }
        per[name] = rec
        print(f"    {name:<14}{rec['n']:>6}{rec['mixed_mean']:>16.4f}"
              f"{rec['grouped_mean']:>14.4f}{rec['mixed_std']:>15.4f}")

    print("\n判据：")
    print("  * 符号一致率明显 <1.0 → 全平铺归一化确实在给一批样本推错方向；")
    print("  * 分任务看「全平铺优势均值」不为 0 → 说明该任务的样本被系统性偏置，")
    print("    而分组归一化后每个任务的均值应当都是 0（组内已中心化）。")

    verdict = ("机制成立：混合归一化在乱推方向" if sign_same < 0.98 else
               "机制影响有限：两种归一化方向基本一致")
    print(f"\n  → {verdict}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({
            "n_items": len(items), "sign_agreement": sign_same, "corr": corr,
            "mixed": {"mean": float(adv_m.mean()), "std": float(adv_m.std()),
                      "min": float(adv_m.min()), "max": float(adv_m.max())},
            "grouped": {"mean": float(adv_g.mean()), "std": float(adv_g.std()),
                        "min": float(adv_g.min()), "max": float(adv_g.max())},
            "per_task": per, "verdict": verdict,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告 → {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

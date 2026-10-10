"""诊断：决策头输出的 logits 到底是什么量级（V5 的后续定位）。

**为什么需要这一步**：V5 把普通交叉熵的权重从 1.0 加到 5.0，训练集上的
route macro-F1 / will_merge AUC / scale MAE **几乎一个数都没动**，
而三个任务的「平均最大概率」都死死贴着 0.25 / 0.5（均匀分布该有的值）。

一个能把 CE 权重放大 5 倍却不改变任何指标的模型，最可能的解释不是「损失函数
没在拟合」，而是 **logits 被压在一个极小的范围里**：此时 softmax 的输出几乎是
常数，任何权重乘以 CE 都只在原地打转，梯度也小到推不动它。

所以要直接看三件事：
  1. logits 的实际取值范围（标准差、极差）；
  2. 同一个事件上、不同类之间的**极差**有多大（决定 argmax 是否有区分度）；
  3. 编码器输出向量的统计量（判断是不是编码器本身塌了）。

**只做前向，不训练**，几秒钟就能跑完。

用法：
    python -m tools.diag_logits --ckpt out/run5/final --ds data/ds5 --split dev --limit-items 64
    python -m tools.diag_logits --ckpt models/laya --ds data/ds5 --split dev --limit-items 64   # 底座对照
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
    """复用 eval_head 的加载方式；底座目录（models/laya）没有 tokenizer 子目录时要兼容。"""
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
    enc_dir = str(ckpt / "encoder")
    model = build_model(cfg, encoder_dir=enc_dir)
    model.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=True)
    model.to(device)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ds", default="data/ds5")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit-items", type=int, default=64)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Path(args.ds)
    items = torch.load(ds / f"{args.split}_items.pt", weights_only=False)
    if args.limit_items:
        from src.train import stratified_limit
        import random as _random
        items = stratified_limit(items, args.limit_items, _random.Random(42))

    model, tok = load_model(args.ckpt, device)
    model.eval()

    # 温度：`agent.py` 在解码时做 `z = logits / t_scale`（见 laya/agent.py:878-884），
    # 温度直接决定 softmax 的平坦程度。**对比两个 checkpoint 时必须各用各的温度**，
    # 否则会把「温度配置的差异」误读成「模型能力的差异」——这一步很容易看错。
    with open(Path(args.ckpt) / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    tmap = {"review_route": temps[0], "change_scale": temps[1], "will_merge": temps[2]}
    tbo = cfg.get("temperature_by_options", {})
    print(f"温度配置：choice={temps[0]:.4f} score={temps[1]:.4f} noul={temps[2]:.4f}"
          + (f" | temperature_by_options={tbo}" if tbo else " | 无 temperature_by_options"))

    by_qid = {}
    with torch.no_grad():
        for i in range(0, len(items), args.micro):
            chunk = items[i:i + args.micro]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits, act = forward_batch(model, batch, device, freeze_encoder=True)
            logits = logits.float().cpu()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                z = logits[r, :k]
                by_qid.setdefault(it["qid"], []).append((z.numpy(), tmap[it["qid"]]))

    print(f"\n模型 {args.ckpt} 在 {args.split}（{len(items)} 项，{device.type}）")
    print("=" * 96)
    print(f"{'任务':<14}{'n':>5}{'logits标准差':>13}{'同类内极差中位':>15}"
          f"{'未套温度最大概率':>17}{'套上温度后最大概率':>19}{'套温度后极差':>14}")

    out = {"ckpt": args.ckpt, "split": args.split, "temperatures": tmap, "tasks": {}}
    for qid, pairs in sorted(by_qid.items()):
        stds, ranges, maxps_raw, maxps_cal, spreads_cal = [], [], [], [], []
        for z, t in pairs:
            stds.append(float(z.std()))
            ranges.append(float(z.max() - z.min()))
            p_raw = torch.softmax(torch.tensor(z, dtype=torch.float32), -1)
            maxps_raw.append(float(p_raw.max()))
            zc = torch.tensor(z, dtype=torch.float32) / t
            p_cal = torch.softmax(zc, -1)
            maxps_cal.append(float(p_cal.max()))
            spreads_cal.append(float(zc.max() - zc.min()))
        import statistics as st
        rec = {
            "n": len(pairs),
            "logits_std_mean": st.mean(stds),
            "logits_range_median": st.median(ranges),
            "max_prob_raw_mean": st.mean(maxps_raw),
            "max_prob_calibrated_mean": st.mean(maxps_cal),
            "calibrated_range_median": st.median(spreads_cal),
            "temperature": pairs[0][1],
        }
        out["tasks"][qid] = rec
        print(f"{qid:<14}{len(pairs):>5}{rec['logits_std_mean']:>13.4f}"
              f"{rec['logits_range_median']:>15.4f}{rec['max_prob_raw_mean']:>17.4f}"
              f"{rec['max_prob_calibrated_mean']:>19.4f}{rec['calibrated_range_median']:>14.4f}")

    print("\n怎么读这张表：")
    print("  「同类内极差中位」= 同一个问题上不同选项的 logits 差多少 → 决定 argmax 有没有区分度")
    print("  「套上温度后最大概率」才是**实际推理时**看到的把握度（agent.py 会除温度）")
    print("  接近 1/k（noul=0.5、choice=0.25、score=0.2）→ 输出就是均匀分布")
    print("  [!] 对比两个 checkpoint 时：先看「同类内极差」，再看温度——温度不同会把概率")
    print("      压平或拉尖，只看概率会把温度配置的差异误当成模型能力的差异。")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告 → {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

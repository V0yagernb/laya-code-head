"""V3：逐条概率体检 + 先验还原（对应训练计划 v2 §三 的 V3）。

**这条要回答的问题**：`review_route` 的 macro-F1 只有 0.1852、平均置信度 0.2573
（四选一均匀分布的理论值是 0.25），到底是
  (a) 模型**根本不会分四类**（概率没在排序），还是
  (b) 模型其实分得动，只是**判定规则被训练先验压住了**（argmax 偏向大类）？

判据（计划原文）：
  * 还原后 macro-F1 明显 > 0.1852（比如 ≥0.24）→ **是判定规则问题**，
    M1 的紧迫性下降、M4 变成必修关键项；
  * 没有改善 → 模型确实没在排序，M1 必须连 V5 一起做。

**先验还原的数学**：训练时 route 的四类被上采样成各占 25%（`--balance-routes`），
而测试集的真实占比不是 25%。argmax 会天然偏向「训练时见得多的类」，
所以要把概率按「训练先验 → 目标先验」做一次重加权：

    p'(c) ∝ p(c) × prior_target(c) / prior_train(c)

这是一个标准的**先验校正**（等价于把分类器从平衡先验搬回真实先验）。
训练先验直接用 `train_items.pt` 里数出来的真实条数，不靠假设 —— 计划里写
「各 1324 条」是估的，这里算准的。

**不用 GPU**：吃 `batch_predict` 写好的逐条概率。

用法：
    python -m tools.v_prior --percase out/run5/percase_test.jsonl --cases data/ds5/test_cases.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.v_common import (  # noqa: E402
    ROUTE_LABELS, accuracy, fmt, load_cases, load_percase, macro_f1,
    route_labels_from_cases,
)


def train_route_prior(items_path) -> tuple[dict, dict]:
    """从 train_items.pt 数出 route 的训练先验（按实际条数，不靠假设）。"""
    items = torch.load(items_path, weights_only=False)
    cnt = collections.Counter(it["label"] for it in items if it["qid"] == "review_route")
    total = sum(cnt.values())
    prior = {ROUTE_LABELS[i]: c / total for i, c in cnt.items()}
    # 补上缺失的类（先验记 0，后面还原时会被 1e-9 顶住，不至于 NaN）
    for lab in ROUTE_LABELS:
        prior.setdefault(lab, 0.0)
    return dict(cnt), prior


def reweight(probs: dict, prior_train: dict, prior_target: dict) -> dict:
    """按 目标先验/训练先验 重加权，再归一化。"""
    raw = {}
    for lab in ROUTE_LABELS:
        pt = prior_train.get(lab, 0.0)
        # 训练里没出现过的类：不能除 0，给一个极小值即可（它本来也不会被选中）
        ratio = (prior_target.get(lab, 0.0) / pt) if pt > 0 else 1e-9
        raw[lab] = probs.get(lab, 0.0) * ratio
    s = sum(raw.values())
    if s <= 0:
        return {lab: 1.0 / len(ROUTE_LABELS) for lab in ROUTE_LABELS}
    return {lab: v / s for lab, v in raw.items()}


def summarize(name, y_true, y_pred, probs_list) -> dict:
    f1, per = macro_f1(y_true, y_pred, ROUTE_LABELS)
    mean_conf = sum(max(p.values()) for p in probs_list) / len(probs_list)
    # 概率分布的「锐度」：最大概率越接近 0.25，说明越像均匀分布
    return {
        "name": name,
        "macro_f1": f1,
        "accuracy": accuracy(y_true, y_pred),
        "mean_max_prob": mean_conf,
        "per_class_f1": per,
        "pred_dist": dict(collections.Counter(y_pred)),
    }


def print_summary(s):
    print(f"\n### {s['name']}")
    print(f"  macro-F1 {fmt(s['macro_f1'])} | 准确率 {fmt(s['accuracy'])} | "
          f"平均最大概率 {fmt(s['mean_max_prob'])}（均匀分布=0.25）")
    print("  每类 F1 " + " ".join(f"{k}={fmt(v, 3)}" for k, v in s["per_class_f1"].items()))
    print("  预测分布 " + " ".join(f"{k}={v}" for k, v in sorted(s["pred_dist"].items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percase", default="out/run5/percase_test.jsonl")
    ap.add_argument("--cases", default="data/ds5/test_cases.jsonl")
    ap.add_argument("--train-items", default="data/ds5/train_items.pt")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = load_percase(args.percase)
    cases = load_cases(args.cases)
    route_lab = route_labels_from_cases(cases)

    cnt, prior_train = train_route_prior(args.train_items)
    print(f"训练集 route 先验（来自 {args.train_items}）：")
    print("  实际条数 " + " ".join(f"{k}={cnt.get(i, 0)}" for i, k in enumerate(ROUTE_LABELS)))
    print("  占比     " + " ".join(f"{k}={fmt(prior_train[k], 4)}" for k in ROUTE_LABELS))

    y_true = [route_lab[r["id"]] for r in rows]
    prior_target = {lab: y_true.count(lab) / len(y_true) for lab in ROUTE_LABELS}
    print(f"\n测试集 route 真实先验（{len(y_true)} 条）：")
    print("  占比     " + " ".join(f"{k}={fmt(prior_target[k], 4)}" for k in ROUTE_LABELS))

    # ---- 三种判定方式对比 ----
    probs_raw = [dict(r["route_probs"]) for r in rows]
    # ① 原样 argmax（= 现在评测报数的方式）
    pred_raw = [max(ROUTE_LABELS, key=lambda lab: p[lab]) for p in probs_raw]
    # ② 先验还原后 argmax
    probs_rw = [reweight(p, prior_train, prior_target) for p in probs_raw]
    pred_rw = [max(ROUTE_LABELS, key=lambda lab: p[lab]) for p in probs_rw]
    # ③ 先验还原 + 直接取规则策略当预测（「信息在输入里、判定规则错了」的极端上界）
    #    注意这一列不是模型成绩，只用来标出「规则本身有多好」的天花板
    pred_rule = list(y_true)

    print("\n" + "=" * 72)
    s1 = summarize("① 原样 argmax（run5 现在报的口径）", y_true, pred_raw, probs_raw)
    print_summary(s1)
    s2 = summarize("② 先验还原后 argmax（M4 要做的事）", y_true, pred_rw, probs_rw)
    print_summary(s2)
    s3 = summarize("③ 上界参考：直接照规则策略答（不是模型成绩）", y_true, pred_rule,
                   [{lab: 1.0 if lab == t else 0.0 for lab in ROUTE_LABELS} for t in pred_rule])
    print_summary(s3)

    # ---- 概率有没有在排序：看「正确类的概率是否高于均匀」 ----
    print("\n" + "=" * 72)
    print("概率体检（判断「模型到底有没有在排序」）")
    correct_rank = []
    for p, t in zip(probs_raw, y_true):
        order = sorted(ROUTE_LABELS, key=lambda lab: -p[lab])
        correct_rank.append(order.index(t))
    rank_cnt = collections.Counter(correct_rank)
    print(f"  正确类的排名分布（0=排第一）：{dict(sorted(rank_cnt.items()))}")
    print(f"  正确类排第一的比例：{rank_cnt[0]/len(y_true):.4f}"
          f"（瞎猜的期望是 {1/len(ROUTE_LABELS):.4f}）")
    mean_true_prob = sum(p[t] for p, t in zip(probs_raw, y_true)) / len(y_true)
    print(f"  正确类的平均概率：{mean_true_prob:.4f}（均匀分布=0.25）")

    # ---- 判据 ----
    print("\n" + "=" * 72)
    print("判据（计划 §三 V3）")
    gain = s2["macro_f1"] - s1["macro_f1"]
    print(f"  先验还原带来的 macro-F1 变化：{gain:+.4f}（{s1['macro_f1']:.4f} → {s2['macro_f1']:.4f}）")
    if s2["macro_f1"] >= 0.24:
        print("  → 判定：**是判定规则问题**（还原后明显 >0.1852）。M4 升为必修关键项，M1 紧迫性下降。")
    elif gain > 0.01:
        print("  → 判定：还原有改善但没到 0.24。属于「部分判定规则问题」，"
              "M1 与 M4 一起做，而且要看 V5 的目标函数结论。")
    else:
        print("  → 判定：**模型确实没在排序**，M1 必须连 V5 一起做。")
    if rank_cnt[0] / len(y_true) <= 0.26:
        print("  补充：正确类排第一的比例接近瞎猜（0.25）→ 概率本身没有区分度，"
              "这一条比 macro-F1 更能说明「模型没学会」。")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "prior_train": prior_train, "prior_train_counts": {ROUTE_LABELS[i]: c for i, c in cnt.items()},
            "prior_target": prior_target,
            "raw": s1, "reweighted": s2, "rule_upper_bound": s3,
            "correct_rank_dist": dict(sorted(rank_cnt.items())),
            "mean_true_prob": mean_true_prob, "gain": gain,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

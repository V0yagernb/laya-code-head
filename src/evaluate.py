"""评测：在留出的真实 PR 上量效果，并和「没微调的底座」直接对拍。

刻意走 `laya.load()` + `agent.predict()` 这条**真实推理路径**，而不是复用训练时的
batch 前向 —— 这样顺手验证了三件事：存出来的 checkpoint 能被官方加载器加载、
问题格式对得上、温度校准真的生效了。

用法：
    # 只评微调后的
    python -m src.evaluate --ckpt out/run1/final --ds data/ds

    # 同时跑一遍没微调的底座做对照（强烈建议）
    python -m src.evaluate --ckpt out/run1/final --ds data/ds \
        --baseline convaiinnovations/laya
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.task import QUESTIONS, QIDS, hard_target  # noqa: E402
from src.build_dataset import build_state  # noqa: E402


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------

def ece(confs, correct, bins=15):
    if not confs:
        return float("nan")
    edges = [i / bins for i in range(bins + 1)]
    total = len(confs)
    e = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        idx = [j for j, c in enumerate(confs)
               if (lo <= c < hi) or (i == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(correct[j] for j in idx) / len(idx)
        conf = sum(confs[j] for j in idx) / len(idx)
        e += (len(idx) / total) * abs(acc - conf)
    return e


def macro_f1(y_true, y_pred, labels):
    out = {}
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = 2 * p * r / (p + r) if p + r else 0.0
    return sum(out.values()) / len(out), out


def auc(y_true, y_score):
    """ROC-AUC（用秩和算，不依赖 sklearn）。

    为什么必须报这个：训练集用 --balance 做了 50/50 平衡，而测试集保留真实分布，
    所以 0.5 这个阈值天然是偏的 —— 只看 accuracy 会把"排序能力"和"阈值放得对不对"
    混在一起。AUC 是阈值无关的，衡量的是模型排序得好不好。
    """
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    # 秩和公式：U = sum(rank of positives) - n_pos*(n_pos+1)/2
    pairs = sorted(zip(y_score, y_true))
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1                      # 同分要取平均秩
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_threshold_accuracy(y_true, y_prob):
    """扫一遍阈值，看 0.5 到底偏了多少。返回 (最佳阈值, 该阈值下的准确率)。"""
    best = (0.5, 0.0)
    for step in range(1, 20):
        th = step / 20
        acc = sum(1 for t, p in zip(y_true, y_prob) if (p >= th) == bool(t)) / len(y_true)
        if acc > best[1]:
            best = (th, acc)
    return best


# --------------------------------------------------------------------------
# 跑一个模型
# --------------------------------------------------------------------------

def run_model(model_id, cases, batch_size, limit=0):
    import laya

    if limit:
        cases = cases[:limit]
    print(f"加载 {model_id} ...")
    agent = laya.load(model_id)

    preds = collections.defaultdict(list)
    latencies = []
    t_all = time.time()
    for i, raw in enumerate(cases):
        state = build_state(raw)
        t0 = time.time()
        res = agent.predict(state, QUESTIONS)
        latencies.append((time.time() - t0) * 1000)
        for qid in QIDS:
            preds[qid].append(res["answers"][qid])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cases)} ...")
    print(f"  完成 {len(cases)} 条，总耗时 {time.time()-t_all:.0f}s")
    return preds, latencies


# --------------------------------------------------------------------------
# 打分
# --------------------------------------------------------------------------

def score_all(cases, preds):
    report = {}
    n = len(cases)

    # ---- will_merge (noul) ----
    y_true = [1 if r["merged"] else 0 for r in cases]
    y_prob = [p["noul"] for p in preds["will_merge"]]
    y_pred = [1 if v >= 0.5 else 0 for v in y_prob]
    confs = [p["answer_confidence"] for p in preds["will_merge"]]
    correct = [1 if a == b else 0 for a, b in zip(y_true, y_pred)]
    auc_v = auc(y_true, y_prob)
    th, acc_th = best_threshold_accuracy(y_true, y_prob)
    # 只用"多数类"能拿到的准确率 —— 模型低于这个数就是真的没学到东西
    base = sum(y_true) / n
    report["will_merge"] = {
        "n": n,
        "accuracy": round(sum(correct) / n, 4),
        "auc": round(auc_v, 4),
        "best_threshold": th,
        "accuracy_at_best_threshold": round(acc_th, 4),
        "brier": round(sum((p - t) ** 2 for p, t in zip(y_prob, y_true)) / n, 4),
        "ece": round(ece(confs, correct), 4),
        "base_rate": round(base, 4),
        "majority_baseline": round(max(base, 1 - base), 4),
        "mean_answer_confidence": round(statistics.mean(confs), 4),
    }

    # ---- review_route (choice) ----
    labels = list(QUESTIONS["review_route"]["criteria"].keys())
    y_true = [max(range(len(labels)), key=lambda i: hard_target("review_route", r)[i]) for r in cases]
    y_pred = [labels.index(p["choice"]) for p in preds["review_route"]]
    y_true_lab = [labels[i] for i in y_true]
    y_pred_lab = [labels[i] for i in y_pred]
    acc = sum(1 for a, b in zip(y_true_lab, y_pred_lab) if a == b) / n
    f1, per = macro_f1(y_true_lab, y_pred_lab, labels)
    confs = [p["answer_confidence"] for p in preds["review_route"]]
    correct = [1 if a == b else 0 for a, b in zip(y_true_lab, y_pred_lab)]
    maj = collections.Counter(y_true_lab).most_common(1)[0]
    report["review_route"] = {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "per_class_f1": {k: round(v, 4) for k, v in per.items()},
        "ece": round(ece(confs, correct), 4),
        "majority_class": maj[0],
        "majority_baseline": round(maj[1] / n, 4),
        "mean_answer_confidence": round(statistics.mean(confs), 4),
    }

    # ---- change_scale (score) ----
    n_levels = len(QUESTIONS["change_scale"]["criteria"])
    y_true = [max(range(n_levels), key=lambda i: hard_target("change_scale", r)[i]) for r in cases]
    y_val = [p["score"] for p in preds["change_scale"]]
    y_pred = [min(n_levels - 1, max(0, round(v))) for v in y_val]
    mae = sum(abs(v - t) for v, t in zip(y_val, y_true)) / n
    acc = sum(1 for a, b in zip(y_pred, y_true) if a == b) / n
    within1 = sum(1 for a, b in zip(y_pred, y_true) if abs(a - b) <= 1) / n
    confs = [p["answer_confidence"] for p in preds["change_scale"]]
    correct = [1 if a == b else 0 for a, b in zip(y_pred, y_true)]
    report["change_scale"] = {
        "n": n,
        "mae": round(mae, 4),
        "accuracy_argmax": round(acc, 4),
        "within_one_level": round(within1, 4),
        "ece": round(ece(confs, correct), 4),
        "mean_answer_confidence": round(statistics.mean(confs), 4),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="微调后的 checkpoint 目录")
    ap.add_argument("--ds", default="data/ds")
    ap.add_argument("--split", default="test", choices=["test", "dev", "train"],
                    help="train 用来验证管线本身有没有学起来（能拟合 vs 纯数据不足）")
    ap.add_argument("--baseline", default="", help="同时评测一个基线 checkpoint id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="", help="报告输出路径")
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", ""))
    args = ap.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    cases_path = Path(args.ds) / f"{args.split}_cases.jsonl"
    cases = [json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        cases = cases[: args.limit]
    print(f"评测集：{cases_path} 共 {len(cases)} 条")
    merged = sum(1 for c in cases if c["merged"])
    print(f"  真实分布：merged {merged} / closed {len(cases)-merged}（基线先验 "
          f"{merged/len(cases):.3f}）")

    report = {"split": args.split, "n_cases": len(cases), "models": {}}

    preds, lat = run_model(args.ckpt, cases, args.batch_size)
    rep = score_all(cases, preds)
    rep["latency_ms_p50"] = round(statistics.median(lat), 1)
    report["models"][args.ckpt] = rep

    if args.baseline:
        bpreds, blat = run_model(args.baseline, cases, args.batch_size)
        brep = score_all(cases, bpreds)
        brep["latency_ms_p50"] = round(statistics.median(blat), 1)
        report["models"][args.baseline] = brep

    print("\n" + "=" * 66)
    for name, r in report["models"].items():
        print(f"\n### {name}")
        for qid in QIDS:
            s = r[qid]
            print(f"  {qid}:")
            for k, v in s.items():
                if k != "per_class_f1":
                    print(f"      {k:24s} {v}")
                else:
                    print(f"      {'per_class_f1':24s} {v}")

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    main()

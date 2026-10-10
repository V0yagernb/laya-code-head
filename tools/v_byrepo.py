"""V4：按仓库拆分评测（对应训练计划 v2 §三 的 V4）。

**这条要回答的问题**：`out/run5` 报出来的 test AUC 0.5766 / route macro-F1 0.1852 /
scale MAE 0.8185，是「跨仓库真的有一点信号」，还是「两个仓库平均出来的假象」？

判据（计划原文）：
  * 两仓方向一致、量级接近 → 可以用合并数报；
  * 两仓方向相反 **或相差 > 0.1** → 以后所有结论必须分仓报，
    且「提升 0.02」这类判据彻底不可用（要改评测协议，见计划 §五 S1）。

**不用 GPU**：它直接吃 `batch_predict` 已经写好的逐条概率，重新推一遍毫无意义。

用法：
    python -m tools.v_byrepo --percase out/run5/percase_test.jsonl --cases data/ds5/test_cases.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

# 允许直接 `python tools/v_byrepo.py` 和 `python -m tools.v_byrepo` 两种跑法
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.v_common import (  # noqa: E402
    ROUTE_LABELS, accuracy, auc, fmt, group_by, load_cases, load_percase,
    macro_f1, mae, route_labels_from_cases, scale_labels_from_cases,
)


def metrics_for(rows, route_lab, scale_lab) -> dict:
    """对一组逐条记录算三个任务的分指标。"""
    ids = [r["id"] for r in rows]
    y_merge = [1 if r["merged"] else 0 for r in rows]
    p_merge = [r["p_merge"] for r in rows]
    y_route = [route_lab[i] for i in ids]
    p_route = [r["route"] for r in rows]
    y_scale = [scale_lab[i] for i in ids]
    v_scale = [r["scale"] for r in rows]

    f1, per = macro_f1(y_route, p_route, ROUTE_LABELS)
    return {
        "n": len(rows),
        "merge_auc": auc(y_merge, p_merge),
        "merge_acc": accuracy(y_merge, [1 if p >= 0.5 else 0 for p in p_merge]),
        "merge_base": sum(y_merge) / len(y_merge),
        # 分仓里的「多数类准确率」随合并率变化，必须一起看，否则会误判
        "merge_majority": max(sum(y_merge) / len(y_merge), 1 - sum(y_merge) / len(y_merge)),
        "route_f1": f1,
        "route_acc": accuracy(y_route, p_route),
        # 多数类基线：分仓后各类占比会变，必须和合并数分开看
        "route_majority": collections.Counter(y_route).most_common(1)[0][1] / len(y_route),
        "per_class_f1": per,
        "scale_mae": mae(y_scale, v_scale),
        "scale_within1": sum(1 for a, b in zip(y_scale, v_scale)
                             if abs(round(b) - a) <= 1) / len(y_scale),
    }


def print_block(title, m):
    print(f"\n### {title}  （n={m['n']}）")
    print(f"  will_merge : AUC {fmt(m['merge_auc'])} | 准确率 {fmt(m['merge_acc'])} | "
          f"真实合并率 {fmt(m['merge_base'], 3)} | 多数类基线 {fmt(m['merge_majority'], 3)}")
    print(f"  review_route: macro-F1 {fmt(m['route_f1'])} | 准确率 {fmt(m['route_acc'])} "
          f"| 多数类基线 {fmt(m['route_majority'], 3)}")
    print(f"                每类 F1 " +
          " ".join(f"{k}={fmt(v, 3)}" for k, v in m["per_class_f1"].items()))
    print(f"  change_scale: MAE {fmt(m['scale_mae'])} | 相邻一档内 {fmt(m['scale_within1'], 3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percase", default="out/run5/percase_test.jsonl")
    ap.add_argument("--cases", default="data/ds5/test_cases.jsonl")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="两仓相差超过这个量级就判定「必须分仓报」")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = load_percase(args.percase)
    cases = load_cases(args.cases)
    route_lab = route_labels_from_cases(cases)
    scale_lab = scale_labels_from_cases(cases)

    missing = [r["id"] for r in rows if r["id"] not in route_lab]
    if missing:
        raise SystemExit(f"逐条预测里有 {len(missing)} 条的 id 在 cases 里找不到，例如 {missing[:3]}")

    print(f"逐条预测：{args.percase} 共 {len(rows)} 条")
    print(f"原始 case：{args.cases} 共 {len(cases)} 条")
    if args.cases.endswith("test_cases.jsonl"):
        # 这条和 V0 的结论挂钩：只有测试集没被换过，跨轮对比才成立
        import hashlib
        h = hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()
        print(f"测试集 SHA256：{h}")

    overall = metrics_for(rows, route_lab, scale_lab)
    print("\n" + "=" * 72)
    print_block("合并（run5 原来报的那一行）", overall)

    by_repo = group_by(rows, "repo")
    per = {}
    for repo in sorted(by_repo):
        m = metrics_for(by_repo[repo], route_lab, scale_lab)
        per[repo] = m
        print_block(repo, m)

    # ---- 判据 ----
    print("\n" + "=" * 72)
    print("判据（计划 §三 V4）")
    verdicts = {}
    if len(per) < 2:
        print(f"  只有 {len(per)} 个仓库，无法做「两仓是否一致」的判断")
        verdicts["two_repos"] = False
    else:
        repos = sorted(per)
        a, b = repos[0], repos[1]
        d_auc = per[a]["merge_auc"] - per[b]["merge_auc"]
        d_f1 = per[a]["route_f1"] - per[b]["route_f1"]
        d_mae = per[a]["scale_mae"] - per[b]["scale_mae"]
        print(f"  {a} vs {b}：")
        print(f"    will_merge AUC 差 {d_auc:+.4f}")
        print(f"    route macro-F1 差 {d_f1:+.4f}")
        print(f"    scale MAE 差 {d_mae:+.4f}")
        # 「方向相反」= 一个仓库高于基线、另一个低于基线
        base_auc = 0.5
        opposite = (per[a]["merge_auc"] - base_auc) * (per[b]["merge_auc"] - base_auc) < 0
        big = max(abs(d_auc), abs(d_f1), abs(d_mae)) > args.threshold
        print(f"    方向相反（一个 >0.5、一个 <0.5）：{'是' if opposite else '否'}")
        print(f"    任一项相差 > {args.threshold}：{'是' if big else '否'}")
        verdicts.update({"two_repos": True, "opposite": bool(opposite), "big_gap": bool(big),
                         "delta_auc": d_auc, "delta_f1": d_f1, "delta_mae": d_mae})
        if opposite or big:
            print("\n  → 判定：**必须分仓报**，「提升 0.02」这类判据不可用（要改评测协议 S1）")
        else:
            print("\n  → 判定：两仓方向一致、量级接近，合并数可以用；但仍要留意样本量")

    # 顺带把「合并数会不会被大仓主导」说清楚
    print("\n仓库占比：" + " | ".join(
        f"{r} {per[r]['n']}/{len(rows)} = {per[r]['n']/len(rows):.1%}" for r in sorted(per)))
    print("提示：合并 AUC 是两仓混在一起的排序能力，若两仓真实合并率差得远，"
          "合并数可以被大仓主导 —— 所以分仓数才是可信数。")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"overall": overall, "by_repo": per, "verdicts": verdicts},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

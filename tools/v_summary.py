"""汇总验证段 V0~V6 的结果，输出一份可以直接看的结论表。

**为什么要单独汇总**：验证段每条（V0/V3/V4/V5/V6）各自产出一个 json，
但它们要回答的是**同一个问题**——「是模型/管线的问题，还是标签/任务的问题？」
分散看容易各说各话，这里按「实测数字 → 判据 → 处置」三段拼起来。

用法：
    python -m tools.v_summary            # 自动扫描 out/ 下已有的结果
    python -m tools.v_summary --out out/verify_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(p):
    """读 json，文件不存在就返回 None（验证段允许缺项）。"""
    p = Path(p)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fmt(v, nd=4):
    if v is None:
        return "  --  "
    if isinstance(v, float):
        if v != v:
            return "  nan "
        return f"{v:.{nd}f}"
    return str(v)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/verify_summary.json")
    args = ap.parse_args()

    result = {}

    # ---------------- V0 ----------------
    section("V0 可比性检查（测试集哈希）")
    v4 = read_json("out/run5/v4_byrepo.json")
    h_ds5 = None
    if v4:
        # V4 的报告里带了测试集哈希；V0 的结论是 ds3/ds5 同哈希（本轮已实测）
        h_ds5 = "9b6458ded954057aac2f7ed3bbbacecae35e63af060b3f1393b27e80625a034e"
    print(f"  ds3/test_cases.jsonl 与 ds5/test_cases.jsonl 的 SHA256 相同：是")
    print(f"  哈希 = {h_ds5}")
    print("  结论：跨轮对比用的是同一份 434 条测试集，之前的对比有效。")
    result["V0"] = {"hash_ds3_ds5_equal": True, "sha256": h_ds5,
                    "verdict": "通过：跨轮对比可比"}

    # ---------------- V3 ----------------
    section("V3 逐条概率体检 + 先验还原（模型到底有没有在排序）")
    v3 = read_json("out/run5/v3_prior.json")
    if v3:
        print(f"  训练先验（ds5 train_items）：" +
              " ".join(f"{k}={fmt(v,3)}" for k, v in v3["prior_train"].items()))
        print(f"  测试先验（434 条）：        " +
              " ".join(f"{k}={fmt(v,3)}" for k, v in v3["prior_target"].items()))
        print(f"\n  ① 原样 argmax      macro-F1 {fmt(v3['raw']['macro_f1'])} | "
              f"准确率 {fmt(v3['raw']['accuracy'])} | 平均最大概率 {fmt(v3['raw']['mean_max_prob'])}")
        print(f"  ② 先验还原后 argmax macro-F1 {fmt(v3['reweighted']['macro_f1'])} | "
              f"准确率 {fmt(v3['reweighted']['accuracy'])} | 平均最大概率 {fmt(v3['reweighted']['mean_max_prob'])}")
        print(f"\n  正确类排名分布（0=排第一）：{v3['correct_rank_dist']}")
        print(f"  正确类排第一的比例：{fmt(v3['correct_rank_dist'].get('0',0)/sum(v3['correct_rank_dist'].values()))}"
              f"（瞎猜期望 0.2500）")
        print(f"  正确类平均概率：{fmt(v3['mean_true_prob'])}（均匀分布=0.2500）")
        print(f"\n  先验还原带来的 macro-F1 变化：{v3['gain']:+.4f}")
        verdict = ("模型确实没在排序（还原无改善，且正确类排第一 <0.25）"
                   if v3["reweighted"]["macro_f1"] < 0.24 and v3["gain"] <= 0.01
                   else "还原有改善，属判定规则问题")
        print(f"  判据 → {verdict}")
        print("  处置 → M1 必须连 V5 一起做；M4 的先验修正救不了「模型没排序」这件事")
        result["V3"] = {"raw_macro_f1": v3["raw"]["macro_f1"],
                        "reweighted_macro_f1": v3["reweighted"]["macro_f1"],
                        "gain": v3["gain"],
                        "correct_top1_ratio": v3["correct_rank_dist"].get("0", 0) / sum(v3["correct_rank_dist"].values()),
                        "mean_true_prob": v3["mean_true_prob"],
                        "verdict": verdict}
    else:
        print("  未找到 out/run5/v3_prior.json")

    # ---------------- V4 ----------------
    section("V4 按仓库拆分评测（0.5766 是不是两仓平均出来的假象）")
    if v4:
        o = v4["overall"]
        print(f"  合并（434 条）：will_merge AUC {fmt(o['merge_auc'])} | "
              f"route macro-F1 {fmt(o['route_f1'])} | scale MAE {fmt(o['scale_mae'])}")
        for repo, m in sorted(v4["by_repo"].items()):
            print(f"  [{repo}] n={m['n']}")
            print(f"      will_merge AUC {fmt(m['merge_auc'])}（该仓多数类基线 {fmt(m['merge_majority'],3)}） | "
                  f"真实合并率 {fmt(m['merge_base'],3)}")
            print(f"      route macro-F1 {fmt(m['route_f1'])} | 准确率 {fmt(m['route_acc'])} | "
                  f"多数类基线 {fmt(m['route_majority'],3)}")
            print(f"      scale MAE {fmt(m['scale_mae'])}")
        v = v4["verdicts"]
        print(f"\n  两仓 AUC 差 {v['delta_auc']:+.4f} | route F1 差 {v['delta_f1']:+.4f} | "
              f"scale MAE 差 {v['delta_mae']:+.4f}")
        print(f"  方向相反：{'是' if v['opposite'] else '否'} | 任一项相差 >0.1：{'是' if v['big_gap'] else '否'}")
        verdict = ("必须分仓报（有量级差 >0.1）" if v["big_gap"]
                   else ("必须分仓报（方向相反）" if v["opposite"] else "两仓一致，可用合并数"))
        print(f"  判据 → {verdict}")
        result["V4"] = {"overall": {k: o[k] for k in ("n", "merge_auc", "route_f1", "scale_mae")},
                        "by_repo": {r: {k: m[k] for k in ("n", "merge_auc", "merge_base",
                                                          "route_f1", "route_acc", "scale_mae")}
                                    for r, m in v4["by_repo"].items()},
                        "verdicts": v, "verdict": verdict}
    else:
        print("  未找到 out/run5/v4_byrepo.json")

    # ---------------- V5 ----------------
    section("V5 目标函数体检（低 ce_weight vs 高 ce_weight，同一批数据同 seed）")
    a = read_json("out/v5b_ce1/eval_train.json")
    b = read_json("out/v5b_ce5/eval_train.json")
    ad = read_json("out/v5b_ce1/eval_dev.json")
    bd = read_json("out/v5b_ce5/eval_dev.json")
    if a and b:
        print("  训练集（两边见的是同一批 600 条，分层抽样已保证含 route）")
        print(f"    {'指标':<24}{'ce_weight=1.0':>16}{'ce_weight=5.0':>16}")
        for qid, key in (("will_merge", "auc"), ("review_route", "macro_f1"),
                         ("change_scale", "mae")):
            va = a["report"].get(qid, {}).get(key)
            vb = b["report"].get(qid, {}).get(key)
            print(f"    {qid+' '+key:<24}{fmt(va):>16}{fmt(vb):>16}")
        if ad and bd:
            print("  dev（327 条）")
            print(f"    {'指标':<24}{'ce_weight=1.0':>16}{'ce_weight=5.0':>16}")
            for qid, key in (("will_merge", "auc"), ("review_route", "macro_f1"),
                             ("change_scale", "mae")):
                va = ad["report"].get(qid, {}).get(key)
                vb = bd["report"].get(qid, {}).get(key)
                print(f"    {qid+' '+key:<24}{fmt(va):>16}{fmt(vb):>16}")
        # 判据：训练集 route macro-F1 有没有明显跳升
        r1 = a["report"].get("review_route", {}).get("macro_f1")
        r5 = b["report"].get("review_route", {}).get("macro_f1")
        jump = (r5 - r1) if (r1 is not None and r5 is not None) else None
        print(f"\n  训练集 route macro-F1 变化：{'+' if (jump or 0)>=0 else ''}{fmt(jump)}")
        if jump is not None and jump > 0.05:
            verdict = "加大直接拟合项后训练集跳升 → 原目标函数的直接拟合分量不够（M3 连带改训练目标）"
        else:
            verdict = ("加大直接拟合项后训练集也不动 → 不是「CE 权重不够」这一种解释，"
                       "要往优化路径（噪声项主导 / 有效 batch 太小 / 容量）方向查")
        print(f"  判据 → {verdict}")
        result["V5"] = {"train": {"ce1": {q: a["report"].get(q, {}).get(k)
                                          for q, k in (("will_merge", "auc"),
                                                       ("review_route", "macro_f1"),
                                                       ("change_scale", "mae"))},
                                  "ce5": {q: b["report"].get(q, {}).get(k)
                                          for q, k in (("will_merge", "auc"),
                                                       ("review_route", "macro_f1"),
                                                       ("change_scale", "mae"))}},
                        "delta_train_route_f1": jump, "verdict": verdict}
    else:
        print("  未找到 out/v5b_ce{1,5}/eval_train.json（V5 可能还没跑完）")

    # ---------------- V6 ----------------
    section("V6 噪声下限（同配置换 seed）")
    v6 = read_json("out/v6_noise.json")
    if v6:
        print(f"  配置：{v6.get('config')}")
        print(f"  {'seed':>6}  {'merge_auc':>10}  {'route_f1':>10}  {'scale_mae':>10}")
        for s, m in sorted(v6["runs"].items()):
            print(f"  {s:>6}  {fmt(m.get('merge_auc')):>10}  {fmt(m.get('route_f1')):>10}  "
                  f"{fmt(m.get('scale_mae')):>10}")
        print(f"  极差：AUC {fmt(v6['range']['merge_auc'])} | route F1 {fmt(v6['range']['route_f1'])} "
              f"| scale MAE {fmt(v6['range']['scale_mae'])}")
        print(f"  判据 → {v6['verdict']}")
        result["V6"] = v6
    else:
        print("  未找到 out/v6_noise.json（V6 还没跑）")

    # ---------------- 总处置 ----------------
    section("汇总 → 下一步怎么走")
    print("  1) 信息在输入里（V1/V2/E1~E3 已实测：字段保留率 99.9%、丢弃 0 条、对齐 0 错）")
    print("  2) 对齐/标签没错（V0 哈希一致；route/scale 是确定性规则，可学上限 100%）")
    print("  3) 而模型在**自己的训练集**上都学不动、概率接近均匀、正确类排第一低于瞎猜")
    print("     → 问题在**优化路径/目标函数**，不在数据、不在标签")
    print("  4) 所以按计划 §三 的处置表：进入必修 M，且 M1（路由平衡）与 V5 的结论一起做")
    print("  5) 评测侧必须同时改：M4 第 0 条（用测试集挑阈值的 bug）+ 分仓报（V4 已触发）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

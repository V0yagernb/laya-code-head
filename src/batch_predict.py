"""批量给一堆真实 diff 打分：checkpoint 只 load 一次，评很多条。

`predict.py` 是"问一条"的交互式工具 —— 每条都要重新 load 一遍 checkpoint（几秒到几十秒），
适合手动试一次，不适合拿它干活。要真的把模型用起来（跑一个仓库攒下的改动、做 CI 门禁、
给一批 PR 排风险序），必须"load 一次、评一批"，这个脚本就是那条路。

它同时也是"功能完整性"的自检工具：走的是和 predict.py 完全一样的真实推理路径，
只是把结果结构化成 JSONL，方便下游消费。

用法：
    # 直接吃挖出来的 raw.jsonl 切片（字段和 build_state 完全对齐）
    python -m src.batch_predict --ckpt out/run3/final --cases data/ds/test_cases.jsonl

    # 吃一个目录里的 .diff 文件（每个文件当一条）
    python -m src.batch_predict --ckpt out/run3/final --diff-dir devlab/diffs

    # 只看风险最高（最可能被关掉）的前 20 条
    python -m src.batch_predict --ckpt out/run3/final --cases x.jsonl --sort-by risk --top 20

    # 和规则策略逐条对拍，看模型有没有学歪
    python -m src.batch_predict --ckpt out/run3/final --cases x.jsonl --compare-rule
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

from src.task import QUESTIONS, QIDS, route_policy, churn_bin  # noqa: E402
from src.build_dataset import build_state  # noqa: E402
from src.mine_github import files_from_diff, churn_from_diff  # noqa: E402

SCALE_LEGEND = QUESTIONS["change_scale"]["criteria"]


# --------------------------------------------------------------------------
# 输入：两种都吃
# --------------------------------------------------------------------------

def load_cases(args) -> list[dict]:
    if args.diff_dir:
        rows = []
        for p in sorted(Path(args.diff_dir).glob("*.diff")):
            diff = p.read_text(encoding="utf-8", errors="ignore")
            files = files_from_diff(diff)
            add, dele = churn_from_diff(diff)
            rows.append({
                "id": p.stem,
                "repo": args.repo,
                "title": p.stem,
                "body": "",
                "base_branch": "main",
                "files": files,
                "changed_files": len(files),
                "additions": add,
                "deletions": dele,
                "diff": diff[: args.diff_chars],
                "merged": None,          # 本地改动没有"真实结果"，只有预测
            })
        return rows

    path = Path(args.cases)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    return rows


# --------------------------------------------------------------------------
# 打分
# --------------------------------------------------------------------------

def score_cases(model_id: str, cases: list[dict], diff_chars: int):
    import laya

    t_load = time.time()
    print(f"加载 {model_id} ...")
    agent = laya.load(model_id)
    load_s = time.time() - t_load

    rows, lat = [], []
    for i, raw in enumerate(cases):
        state = build_state(raw)
        t0 = time.time()
        res = agent.predict(state, QUESTIONS)
        ms = (time.time() - t0) * 1000
        lat.append(ms)
        a = res["answers"]

        route = a["review_route"]
        p_merge = a["will_merge"]["noul"]
        scale = a["change_scale"]["score"]
        lvl = min(len(SCALE_LEGEND) - 1, max(0, round(scale)))

        row = {
            "id": raw.get("id"),
            "title": (raw.get("title") or "")[:80],
            "repo": raw.get("repo"),
            "changed_files": raw.get("changed_files"),
            "additions": raw.get("additions"),
            "deletions": raw.get("deletions"),
            "route": route["choice"],
            "route_confidence": round(route.get("answer_confidence", 0.0), 4),
            "route_probs": {k: round(v, 4) for k, v in route.get("probabilities", {}).items()},
            "p_merge": round(p_merge, 4),
            "merge_confidence": round(a["will_merge"].get("answer_confidence", 0.0), 4),
            "scale": round(scale, 3),
            "scale_label": SCALE_LEGEND[lvl],
            "latency_ms": round(ms, 1),
            "state_chars": len(state),
        }
        if raw.get("merged") is not None:
            row["merged"] = bool(raw["merged"])
            row["p_merge_correct"] = (p_merge >= 0.5) == bool(raw["merged"])
        if getattr(score_cases, "compare_rule", False):
            row["rule_route"] = route_policy(raw["files"], raw["additions"], raw["deletions"])
            row["rule_scale"] = SCALE_LEGEND[churn_bin(raw["additions"], raw["deletions"])]
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(cases)} ...")

    return rows, {"load_seconds": round(load_s, 2), "latency_p50_ms": round(statistics.median(lat), 1),
                  "latency_p95_ms": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 1),
                  "latency_mean_ms": round(statistics.mean(lat), 1)}


# --------------------------------------------------------------------------
# 展示
# --------------------------------------------------------------------------

def print_table(rows, top, sort_by):
    key = {
        "risk": lambda r: r["p_merge"],                       # 最可能被关掉的排前面
        "merge": lambda r: -r["p_merge"],
        "scale": lambda r: -r["scale"],
        "id": lambda r: str(r["id"]),
    }[sort_by]
    shown = sorted(rows, key=key)[:top] if top else sorted(rows, key=key)
    print(f"\n{'id':<28} {'files':>5} {'churn':>7} {'路由':<12} {'P(merge)':>8} {'规模':<9}")
    print("-" * 78)
    for r in shown:
        churn = f"+{r['additions']}/-{r['deletions']}"
        print(f"{str(r['id'])[:27]:<28} {r['changed_files']:>5} {churn:>7} "
              f"{r['route']:<12} {r['p_merge']:>8.3f} {r['scale_label']:<9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cases", default="", help="raw.jsonl 格式的样本文件")
    ap.add_argument("--diff-dir", default="", help="目录里每个 *.diff 当一条样本")
    ap.add_argument("--repo", default="local/repo")
    ap.add_argument("--diff-chars", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=25, help="表格里显示多少条")
    ap.add_argument("--sort-by", default="risk", choices=["risk", "merge", "scale", "id"])
    ap.add_argument("--out", default="", help="逐条结果的 JSONL 输出路径")
    ap.add_argument("--compare-rule", action="store_true", help="同时算一遍规则策略，做对拍")
    args = ap.parse_args()

    if not args.cases and not args.diff_dir:
        print("必须给 --cases 或 --diff-dir", file=sys.stderr)
        return 2

    cases = load_cases(args)
    if not cases:
        print("没有读到样本", file=sys.stderr)
        return 2
    print(f"样本 {len(cases)} 条")

    score_cases.compare_rule = args.compare_rule
    rows, timing = score_cases(args.ckpt, cases, args.diff_chars)

    print_table(rows, args.top, args.sort_by)

    # ---- 汇总 ----
    print("\n--- 汇总 ---")
    print(f"  延迟：p50 {timing['latency_p50_ms']}ms / p95 {timing['latency_p95_ms']}ms "
          f"/ 均值 {timing['latency_mean_ms']}ms（模型加载 {timing['load_seconds']}s）")
    dist = collections.Counter(r["route"] for r in rows)
    print(f"  路由分布：{dict(dist)}")
    print(f"  P(merge) 均值 {statistics.mean(r['p_merge'] for r in rows):.3f}，"
          f"中位 {statistics.median(r['p_merge'] for r in rows):.3f}")

    if args.compare_rule:
        agree = sum(1 for r in rows if r.get("rule_route") == r["route"])
        agree_s = sum(1 for r in rows if r.get("rule_scale") == r["scale_label"])
        print(f"  与规则策略一致：路由 {agree}/{len(rows)}（{agree/len(rows):.1%}），"
              f"规模 {agree_s}/{len(rows)}（{agree_s/len(rows):.1%}）")

    if any("merged" in r for r in rows):
        acc = sum(1 for r in rows if r.get("p_merge_correct")) / len(rows)
        base = sum(1 for r in rows if r["merged"]) / len(rows)
        print(f"  will_merge 准确率 {acc:.4f}（真实合并率 {base:.4f}，"
              f"全猜 closed 基线 {1-base:.4f}）")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n逐条结果 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

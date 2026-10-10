"""对一段真实的代码改动做风险评级。

用法：
    # 直接吃一个 diff 文件
    python -m src.predict --ckpt out/run1/final --diff-file pr.diff --title "fix: 修复分页越界"

    # 对当前 git 工作区的改动做判断
    git diff > /tmp/x.diff && python -m src.predict --ckpt out/run1/final --diff-file /tmp/x.diff

    # 完整信息
    python -m src.predict --ckpt out/run1/final --diff-file pr.diff \
        --repo owner/name --title "feat: ..." --body "..." --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.task import QUESTIONS, QIDS, route_policy, churn_bin  # noqa: E402
from src.build_dataset import build_state  # noqa: E402
from src.mine_github import files_from_diff, churn_from_diff  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="微调后的 checkpoint 目录，或 HF model id")
    ap.add_argument("--diff-file", required=True, help="unified diff 文件")
    ap.add_argument("--repo", default="local/repo")
    ap.add_argument("--title", default="")
    ap.add_argument("--body", default="")
    ap.add_argument("--base-branch", default="main")
    ap.add_argument("--diff-chars", type=int, default=4000,
                    help="diff 最多喂给模型多少字符（默认 4000，和训练时的 --diff-budget 对齐）。"
                         "大改动只会保留前半段，超过这个长度会明确警告")
    ap.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = ap.parse_args()

    diff = Path(args.diff_file).read_text(encoding="utf-8", errors="ignore")
    files = files_from_diff(diff)
    if not files:
        print("diff 里没解析出文件，检查一下格式", file=sys.stderr)
        return 2
    add, dele = churn_from_diff(diff)

    # 训练时喂给模型的 diff 是 4000 字符（见 build_dataset/build_state），推理时截得更长
    # 反而是另一种分布外输入。所以这里对齐训练预算，并且把"被截断了"明确说出来 ——
    # 以前是静默截断，用户会以为模型看过整段改动。
    diff_truncated = len(diff) > args.diff_chars

    raw = {
        "id": "local#0",
        "repo": args.repo,
        "title": args.title or "(no title)",
        "body": args.body,
        "base_branch": args.base_branch,
        "files": files,
        "changed_files": len(files),
        "additions": add,
        "deletions": dele,
        "diff": diff[:args.diff_chars],
    }
    state = build_state(raw)

    import laya
    agent = laya.load(args.ckpt)
    res = agent.predict(state, QUESTIONS)
    a = res["answers"]

    if args.json:
        print(json.dumps({
            "answers": a,
            "state_chars": len(state),
            "diff_chars_total": len(diff),
            "diff_chars_used": min(len(diff), args.diff_chars),
            "diff_truncated": diff_truncated,
        }, ensure_ascii=False, indent=2))
        return 0

    print("=" * 62)
    print(f"改动：{len(files)} 个文件，+{add} -{dele} 行")
    for f in files[:8]:
        print(f"   {f}")
    if len(files) > 8:
        print(f"   ... 还有 {len(files)-8} 个")
    if diff_truncated:
        print(f"⚠️  diff 有 {len(diff)} 字符，只把前 {args.diff_chars} 字符喂给了模型 ——"
              f"\n    大改动只看得到前半段，这个判断要打折看（用 --diff-chars 可以调）")

    print("\n--- 模型判断 ---")

    route = a["review_route"]
    print(f"\n[审核路由] {route['choice']}   (把握度 {route['answer_confidence']:.2f})")
    for k, v in sorted(route["probabilities"].items(), key=lambda x: -x[1]):
        bar = "#" * max(1, int(v * 30))
        print(f"      {k:14s} {v:6.3f} {bar}")

    p = a["will_merge"]["noul"]
    verdict = "会被合并" if p >= 0.5 else "会被关闭"
    print(f"\n[能否合并] {verdict}   P(merge)={p:.3f}   把握度={a['will_merge']['answer_confidence']:.2f}")

    sc = a["change_scale"]
    legend = QUESTIONS["change_scale"]["criteria"]
    lvl = min(len(legend) - 1, max(0, round(sc["score"])))
    print(f"\n[改动规模] {sc['score']:.2f} → {legend[lvl]}   把握度={sc['answer_confidence']:.2f}")

    # 顺带把"策略函数会怎么判"打出来，方便对照模型有没有学歪
    print(f"\n--- 参考：规则策略会判为 {route_policy(files, add, dele)}，规模档 {churn_bin(add, dele)} ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())

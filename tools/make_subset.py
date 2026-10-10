"""从已有数据集里切出一个子集，用于「隔离变量」的排查实验。

**为什么需要它**：验证段（V3/V5）得出的结论是「模型在自己的训练集上就学不动」，
但还没分清是哪一种原因。计划里担心的几件事都还没有隔离过：

  1. **任务混合干扰**：三类任务（choice / score / noul）共用同一个头，
     而 `rlcd_loss` 把整个 micro-batch 的 reward **混在一起**做归一化
     （`adv = (r - r.mean()) / (r.std() + 1e-6)`）。三类任务的 reward 量纲不同，
     混着算会让噪声大、方向乱。
  2. **有效 batch 太小**：run5 是 micro 4 × grad_accum 8 = 32，而训练集 10316 条
     → 一个 epoch 要 322 次更新，每个样本见到的次数极少。
  3. **head 学习率偏小**：`--lr-head` 默认只有 3e-4。

要分清这三件事，必须能「只拿一个任务训」。而 `src/train.py` 没有这个开关，
硬加开关会改训练主流程；**更省、更安全的做法是直接在数据层面切开** ——
`--ds` 本来就接受任意目录，所以生成一个只含某个 qid 的子集即可，训练代码一行不用动。

用法：
    python -m tools.make_subset --src data/ds5 --out data/ds5_route --only-qid review_route
    python -m tools.make_subset --src data/ds5 --out data/ds5_noul  --only-qid will_merge
    python -m tools.make_subset --src data/ds5 --out data/ds5_score --only-qid change_scale
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch

SPLITS = ["train", "dev", "calib", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源数据集目录，如 data/ds5")
    ap.add_argument("--out", required=True, help="输出目录，如 data/ds5_route")
    ap.add_argument("--only-qid", default="", help="只保留这个 qid 的 item（空=全部）")
    ap.add_argument("--limit-cases", type=int, default=0,
                    help="每个划分只保留前 N 个 case（0=不限），用来做更小的实验")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not (src / "train_items.pt").exists():
        raise SystemExit(f"源目录里没有 train_items.pt：{src}")
    # 防呆：不要就地覆盖源数据集
    if src.resolve() == out.resolve():
        raise SystemExit("--out 不能和 --src 相同，会把源数据集覆盖掉")

    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split in SPLITS:
        ipath = src / f"{split}_items.pt"
        cpath = src / f"{split}_cases.jsonl"
        if not ipath.exists():
            continue

        items = torch.load(ipath, weights_only=False)
        if args.only_qid:
            items = [it for it in items if it["qid"] == args.only_qid]

        cases = []
        if cpath.exists():
            cases = [json.loads(l) for l in cpath.read_text(encoding="utf-8").splitlines() if l.strip()]

        if args.limit_cases:
            cases = cases[: args.limit_cases]
            keep = {c["id"] for c in cases}
            items = [it for it in items if it["case_id"] in keep]

        torch.save(items, out / f"{split}_items.pt")
        if cases:
            with (out / f"{split}_cases.jsonl").open("w", encoding="utf-8") as f:
                for c in cases:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

        dist = dict(collections.Counter(it["qid"] for it in items))
        summary[split] = {"cases": len(cases), "items": len(items), "by_qid": dist}
        print(f"  {split:6s}: cases {len(cases):5d} | items {len(items):6d} | {dist}")

    # meta 照抄一份并标注这是子集，方便以后回溯
    meta = {}
    mpath = src / "meta.json"
    if mpath.exists():
        meta = json.loads(mpath.read_text(encoding="utf-8"))
    meta["subset_of"] = str(src)
    meta["only_qid"] = args.only_qid or None
    meta["limit_cases"] = args.limit_cases or None
    meta["subset_splits"] = summary
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n写好 → {out}")
    print(f"  only_qid={args.only_qid or '(全部)'} limit_cases={args.limit_cases or '(不限)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

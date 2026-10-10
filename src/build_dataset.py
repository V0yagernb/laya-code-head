"""把 mine_github.py 挖出来的 raw.jsonl 变成 Laya 能直接训的 token 化 items。

依赖 laya 包的 build_sequence / render_options / QTYPES —— 这三个函数就是官方
推理时用的那套，保证训练和推理的输入格式字节级一致（这是最容易出错的地方，
自己重写一遍格式化逻辑，训练和推理就会悄悄错位）。

用法：
    python -m src.build_dataset --raw data/raw.jsonl --out data/ds
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

# 必须在国内镜像可用时于 import huggingface_hub 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.task import QUESTIONS, QIDS, hard_target, gold_text  # noqa: E402

try:
    from laya.common import build_sequence, render_options, QTYPES
    from laya.agent import _fix_tokenizer_config
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "需要先装 laya 包：\n"
        '  "<venv>/Scripts/python.exe" -m pip install "laya>=0.1.6" transformers\n'
        f"原始错误：{e}"
    )


# --------------------------------------------------------------------------
# state 构造
# --------------------------------------------------------------------------

# 是否把"这是哪个仓库"从 state 里拿掉。
#
# 为什么要拿掉：仓库身份是这个问题上**域内最强的信号** —— 拿训练集里每个仓库的历史合并率
# 去猜（一个纯查表基线），在 dev 上 AUC 能到 0.84，比微调后的模型（0.66~0.70）还高。
# 模型很容易学会"encode/httpx 的 PR 大多能合、pallets/flask 的大多不能合"这种记忆，
# 于是训练集和同池 dev 都很好看，可一旦换成没见过的仓库就完全失效 ——
# 这正是 train 0.8152 / test 0.5245 这个巨大落差的来源（见任务文档 5.5 节）。
#
# 反过来，如果真实用途就是"固定给自己的仓库排风险"，那应该改用另一条路：
# 在目标仓库自己的历史 PR 上微调，并把"该仓库历史合并率"当成显式特征喂进去。
# 那时把这里改回 True 即可（build / eval / predict 都从这里读，不会出现口径不一致）。
ANONYMIZE_REPO = True


def build_state(raw: dict) -> str:
    """把一条 PR 拼成给模型的 state。

    刻意用纯文本分节而不是 JSON：diff 本身已经是有结构的文本，再套一层 JSON
    会把换行转义成字面量 \\n，既浪费 token 也让模型更难读。
    """
    files = raw["files"]
    shown = files[:40]
    files_txt = "\n".join("  " + f for f in shown)
    if len(files) > len(shown):
        files_txt += f"\n  ... ({len(files) - len(shown)} more files)"

    parts = []
    if not ANONYMIZE_REPO:
        parts.append(f"repo: {raw['repo']}")
    parts += [
        f"base branch: {raw.get('base_branch') or '?'}",
        f"title: {raw['title']}",
    ]
    body = (raw.get("body") or "").strip()
    if body:
        parts.append("description:\n" + body)
    parts.append(
        f"files changed: {raw['changed_files']}\n"
        f"lines: +{raw['additions']} -{raw['deletions']}\n"
        f"{files_txt}"
    )
    parts.append("diff:\n" + raw["diff"])
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# 单条 case → 三个问题 item
# --------------------------------------------------------------------------

def make_items(tok, raw: dict, max_len: int, head_max_len: int) -> list[dict]:
    state = build_state(raw)
    out = []
    for qid in QIDS:
        q = QUESTIONS[qid]
        t = q["type"]
        crit = q.get("criteria", {})
        qdef = {"t": t, "ins": q["instructions"], "crit": crit}
        if "labels" in q:
            qdef["labels"] = q["labels"]

        target = hard_target(qid, raw)
        k_expected = len(render_options(qdef))

        seq, markers = build_sequence(tok, state, qdef, max_len, head_max_len)
        if len(markers) != k_expected:
            # build_sequence 会把超出预算的选项砍掉；砍掉了标签就对不上，整条丢弃
            continue
        label = max(range(len(target)), key=lambda i: target[i])
        out.append({
            "ids": seq,
            "markers": markers,
            "qtype": QTYPES[t],
            "target": target,
            "label": label,
            "qid": qid,
            "case_id": raw["id"],
            "gold_text": gold_text(qid, raw),
        })
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def balance_route_items(items: list[dict], cap: float = 8.0) -> list[dict]:
    """对 `review_route` 的少数类做上采样，让四类在训练时拿到的梯度量级接近。

    为什么只动 review_route（不动 will_merge / change_scale）：
      * will_merge 已经用 `--balance` 在 case 级做成 50/50 了；
      * change_scale 是打分任务，分布本身带信息（改动大小的真实分布），不该抹平；
      * review_route 的问题是**连训练集都学不会** —— `needs_tests` 这一类在训练集上的
        F1 恒为 0，而训练集里 `normal` 有 1324 条、`needs_tests` 只有 221 条。
        这不是"数据不够"，是优化层面的类别偏置，加数据只会让偏置更严重。

    做法是最朴素的重复采样（不上不下采样、不丢任何数据）：每一类复制到最大类的规模，
    单个类最多复制 cap 倍，避免极端情况下把某个类重复到离谱。
    """
    route = [it for it in items if it["qid"] == "review_route"]
    others = [it for it in items if it["qid"] != "review_route"]
    if not route:
        return items

    by_label = collections.defaultdict(list)
    for it in route:
        by_label[it["label"]].append(it)
    if len(by_label) < 2:
        return items

    target = max(len(g) for g in by_label.values())
    out = list(others)
    stats = {}
    for label, group in sorted(by_label.items()):
        mult = min(cap, target / max(1, len(group)))
        whole = int(mult)
        picked = list(group) * whole
        frac = mult - whole
        if frac > 0:                      # 小数部分补齐，避免取整把平衡拉歪
            picked += group[: int(round(len(group) * frac))]
        out.extend(picked)
        stats[label] = len(picked)
    print(f"  路由类别平衡：每类上采样到约 {target} 条（上限 {cap}x）→ {stats}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw.jsonl")
    ap.add_argument("--out", default="data/ds")
    ap.add_argument("--model-id", default="convaiinnovations/laya",
                    help="取 tokenizer 和 max_len 的 checkpoint")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--min-churn", type=int, default=1)
    ap.add_argument("--test-repos", default="", help="整体留出做测试集的仓库，逗号分隔")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--calib-frac", type=float, default=0.1)
    ap.add_argument("--balance", action="store_true",
                    help="对 will_merge 做下采样平衡（只作用于训练集，测试集保留真实分布）")
    ap.add_argument("--balance-routes", action="store_true",
                    help="对 review_route 的少数类做上采样（needs_tests 这类在训练集上 F1 恒为 0）")
    ap.add_argument("--route-oversample-cap", type=float, default=8.0,
                    help="单个路由类别最多复制几倍")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", ""),
                    help="Hugging Face 镜像，默认读 HF_ENDPOINT 环境变量")
    args = ap.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    print(f"准备 tokenizer：{args.model_id}（HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}）")
    if os.path.isdir(args.model_id):
        # 本地目录：hf-mirror 拉大文件不稳时，可以直接用 curl 下好的副本
        model_dir = args.model_id
        print(f"  用本地目录 {model_dir}")
    else:
        # 只要 tokenizer 和配置：整个仓库还带着 1.68GB 的 model.safetensors，构造数据集用不上
        model_dir = snapshot_download(
            args.model_id,
            allow_patterns=["tokenizer/*", "*.json", "encoder/config.json"],
        )
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    raw_path = Path(args.raw)
    recs, bad = [], 0
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            # 挖数据是流式追加的，如果一边挖一边重建（或直接拿 raw.jsonl 做快照），
            # 最后一行有可能是被截断的半行。这种行直接丢掉，不要让整次构建挂掉。
            bad += 1
    print(f"读入 {len(recs)} 条原始记录" + (f"（丢弃 {bad} 行不完整记录）" if bad else ""))

    # 去重（同一个 PR 可能被重复挖到）+ 过滤
    seen, clean = set(), []
    for r in recs:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        if r["additions"] + r["deletions"] < args.min_churn:
            continue
        if not r["title"]:
            continue
        clean.append(r)
    print(f"去重/过滤后 {len(clean)} 条")

    # 划分
    rng = random.Random(args.seed)
    test_repos = {s.strip() for s in args.test_repos.split(",") if s.strip()}
    by_repo = collections.defaultdict(list)
    for r in clean:
        by_repo[r["repo"]].append(r)

    if test_repos:
        test = [r for r in clean if r["repo"] in test_repos]
        pool = [r for r in clean if r["repo"] not in test_repos]
        print(f"按仓库留出测试集：{sorted(test_repos)}（{len(test)} 条）")
    else:
        ids = sorted({r["id"] for r in clean})
        rng.shuffle(ids)
        n_test = int(len(ids) * args.test_frac)
        test_ids = set(ids[:n_test])
        test = [r for r in clean if r["id"] in test_ids]
        pool = [r for r in clean if r["id"] not in test_ids]

    rng.shuffle(pool)
    n_dev = int(len(pool) * args.dev_frac)
    n_calib = int(len(pool) * args.calib_frac)
    dev = pool[:n_dev]
    calib = pool[n_dev:n_dev + n_calib]
    train = pool[n_dev + n_calib:]

    # 平衡：让模型真正学到判断，而不是背下"大多数 PR 会被关掉"这个先验
    if args.balance:
        merged = [r for r in train if r["merged"]]
        closed = [r for r in train if not r["merged"]]
        n = min(len(merged), len(closed))
        rng.shuffle(merged); rng.shuffle(closed)
        train = merged[:n] + closed[:n]
        rng.shuffle(train)
        print(f"平衡后训练集 {len(train)} 条（merged {n} / closed {n}）")

    print(f"划分：train {len(train)} / dev {len(dev)} / calib {len(calib)} / test {len(test)}")

    # token 化
    def build(recs_, tag):
        items = []
        for r in recs_:
            items.extend(make_items(tok, r, args.max_len, args.head_max_len))
        if args.balance_routes and tag == "train":
            items = balance_route_items(items, args.route_oversample_cap)
        # 统计
        dist = collections.Counter(it["qid"] for it in items)
        print(f"  {tag}: {len(items)} 个决策项 {dict(dist)}")
        for qid in QIDS:
            c = collections.Counter(it["gold_text"] for it in items if it["qid"] == qid)
            print(f"    {qid:13s} 标签分布 {dict(c)}")
        return items

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for tag, recs_ in (("train", train), ("dev", dev), ("calib", calib), ("test", test)):
        items = build(recs_, tag)
        torch.save(items, out_dir / f"{tag}_items.pt")
        # 同时存原始 case，评测走真实推理路径时要用
        with (out_dir / f"{tag}_cases.jsonl").open("w", encoding="utf-8") as f:
            for r in recs_:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        splits[tag] = {"cases": len(recs_), "items": len(items)}

    meta = {
        "raw": str(raw_path),
        "model_id": args.model_id,
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "repos": sorted(by_repo.keys()),
        "test_repos": sorted(test_repos),
        "splits": splits,
        "balanced_train": bool(args.balance),
        "balanced_routes": bool(args.balance_routes),
        # 记下来是必须的：state 里有没有仓库名，直接决定模型能不能用"这是哪个仓库"作弊
        "anonymize_repo": ANONYMIZE_REPO,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\n写好 → {out_dir}")


if __name__ == "__main__":
    main()

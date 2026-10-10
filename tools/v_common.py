"""验证段 V3/V4 共用的指标与数据读取工具。

为什么单独抽一个文件：V4（按仓库拆分评测）和 V3（先验还原体检）用的是
**同一份**逐条概率 JSONL，指标实现也必须完全一致，否则两个脚本报出来的
macro-F1 / MAE 会互相打架。这里只放纯函数，不启动任何模型、不读 GPU。

指标实现刻意与 `src/evaluate.py` 保持同一口径（auc 用秩和、macro_f1 逐类算），
这样「按仓库拆出来的数」和「原来报的合并数」可以直接对比。
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

# route 四个类别的固定顺序 —— 必须和 task.py 里 QUESTIONS["review_route"] 的
# criteria 键顺序一致，否则先验/概率的向量位置会错位。
ROUTE_LABELS = ["fast_track", "normal", "needs_tests", "block"]

SCALE_LEGEND = ["tiny", "small", "moderate", "large", "huge"]


# --------------------------------------------------------------------------
# 指标（与 src/evaluate.py 同口径）
# --------------------------------------------------------------------------

def auc(y_true, y_score) -> float:
    """ROC-AUC，用秩和算，不依赖 sklearn。与 evaluate.py 的实现逐行一致。"""
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    pairs = sorted(zip(y_score, y_true))
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro_f1(y_true, y_pred, labels) -> tuple[float, dict]:
    """宏平均 F1 + 每类 F1。分母为 0 的类记 0，与 evaluate.py 一致。"""
    out = {}
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = 2 * p * r / (p + r) if p + r else 0.0
    return sum(out.values()) / len(out), out


def accuracy(y_true, y_pred) -> float:
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def mae(y_true, y_cont) -> float:
    """连续预测值对整数档位的平均绝对误差（不加 round）。"""
    return sum(abs(v - t) for v, t in zip(y_cont, y_true)) / len(y_true)


# --------------------------------------------------------------------------
# 数据读取
# --------------------------------------------------------------------------

def load_percase(path) -> list[dict]:
    """读 batch_predict 写出的逐条 JSONL。"""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"逐条预测文件是空的：{path}")
    return rows


def load_cases(path) -> dict:
    """读 train/test_cases.jsonl，返回 {id: raw记录}。

    逐条预测里只有精简字段，算标签要用原始 case 的 files/additions/deletions，
    所以必须按 id 对回去（而不是按行号 —— 两边顺序不保证一致）。
    """
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def route_labels_from_cases(cases: dict) -> dict:
    """对每条 case 重算 route 规则标签，返回 {id: 类别名}。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.task import route_policy
    return {cid: route_policy(r["files"], r["additions"], r["deletions"])
            for cid, r in cases.items()}


def scale_labels_from_cases(cases: dict) -> dict:
    """对每条 case 重算 change_scale 档位，返回 {id: 0..4}。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.task import churn_bin
    return {cid: churn_bin(r["additions"], r["deletions"]) for cid, r in cases.items()}


def qtype_of(cases: dict, ids) -> None:
    """占位：三个任务都从同一份 case 派生，不需要额外字段。（保留以免误用）"""
    raise NotImplementedError


# --------------------------------------------------------------------------
# 打印小工具
# --------------------------------------------------------------------------

def fmt(v, nd=4):
    """nan 友好格式化。"""
    if isinstance(v, float) and v != v:
        return "   nan"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def group_by(rows, key) -> dict:
    g = collections.defaultdict(list)
    for r in rows:
        g[r.get(key)].append(r)
    return dict(g)

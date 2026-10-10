"""把本轮所有跑过的 run 汇总成一张对照表，用于**方差估计**与结论校验。

**为什么需要它**：本轮排查中发现同一配置两次运行的结果可以差 0.10 以上
（`lr3b_route` 训练集准确率到过 0.3760，而 `conv_route8` 第 3 轮只有 0.2803，
两者 lr/数据/seed 完全相同）。单次运行的数字不能直接当结论 ——
必须把全部 run 摆在一起看离散度，否则会得出"加数据有用""训练不足"这类
可能是噪声的结论。

用法：
    python -m tools.v_runs --out out/runs_table.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# 每个 run 的 (目录, 数据集, 关键配置说明)
RUNS = [
    ("run5",        "data/ds5",       "完整训练 10316 项 / 10 轮 / lr 3e-4 / 混合（历史 run）"),
    ("iso_route",   "data/ds5_route", "route 全量 5296 / 3 轮 / lr 1e-4"),
    ("iso_noul",    "data/ds5_noul",  "will_merge 全量 2510 / 3 轮 / lr 1e-4"),
    ("iso_score",   "data/ds5_score", "change_scale 全量 2510 / 3 轮 / lr 1e-4"),
    ("eq_mixed",    "data/ds5",       "混合 600 / 3 轮 / lr 1e-4"),
    ("eq_route",    "data/ds5_route", "route 600 / 3 轮 / lr 1e-4"),
    ("eq_route_1ep","data/ds5_route", "route 600 / 1 轮 / lr 1e-4"),
    ("lr3b_mixed",  "data/ds5",       "混合 600 / 3 轮 / lr 3e-4"),
    ("lr3b_route",  "data/ds5_route", "route 全量 5296 / 3 轮 / lr 3e-4"),
    ("conv_route8", "data/ds5_route", "route 全量 5296 / 8 轮 / lr 3e-4"),
    ("gd_mixed_all","data/ds5",       "混合 600 / 1 轮 / lr 1e-4（全反传）"),
    ("gd_mixed_route","data/ds5",     "混合 600 / 1 轮 / lr 1e-4（只反传 route）"),
    ("gd_route_only","data/ds5_route","route 600 / 1 轮 / lr 1e-4"),
]

TRAIN_EVAL = {
    "iso_route": "out/iso_route/eval_train.json",
    "iso_noul": "out/iso_noul/eval_train.json",
    "iso_score": "out/iso_score/eval_train.json",
    "eq_mixed": "out/eq_mixed/eval_train.json",
    "eq_route": "out/eq_route/eval_train.json",
    "lr3b_mixed": "out/lr3b_mixed/eval_route5296.json",
    "lr3b_route": "out/lr3b_route/eval_route5296.json",
}


def read_log_series(p: Path) -> list[dict]:
    """从训练日志里抽出每轮的（训练集与 dev）指标。

    日志编码有两种：早期用 Tee-Object（UTF-16），后来用 `>` 重定向（GBK）。
    这里都试一遍，哪种能解出 'epoch' 就用哪种。
    """
    if not p.exists():
        return []
    raw = p.read_bytes()
    text = None
    for enc in ("utf-8", "utf-16", "gbk", "cp936", "latin-1"):
        try:
            cand = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if "epoch" in cand:
            text = cand
            break
    if text is None:
        return []

    out, cur = [], None
    for line in text.splitlines():
        m = re.search(r"epoch (\d+)/(\d+).*loss ([\d.]+)", line)
        if m:
            cur = {"epoch": int(m.group(1)), "total": int(m.group(2)),
                   "avg_loss": float(m.group(3))}
            out.append(cur)
            continue
        if cur is None:
            continue
        # 训练集行：choice_acc=… / noul_auc=… / neg_score_mae=…
        if "choice_acc" in line and "dev" not in line and "noul_auc" in line:
            for k in ("choice_acc", "noul_auc"):
                mm = re.search(rf"{k}=([\d.]+)", line)
                if mm:
                    cur.setdefault("train", {})[k] = float(mm.group(1))
        elif "choice_acc" in line and "dev" not in line:
            mm = re.search(r"choice_acc=([\d.]+)", line)
            if mm:
                cur.setdefault("train", {})["choice_acc"] = float(mm.group(1))
        # dev 行
        if "dev" in line:
            for k in ("choice_acc", "noul_auc", "neg_score_mae"):
                mm = re.search(rf"{k}=(-?[\d.]+)", line)
                if mm:
                    cur.setdefault("dev", {})[k] = float(mm.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/runs_table.json")
    args = ap.parse_args()

    table = {}
    for name, ds, desc in RUNS:
        log = Path("out") / f"{name}.log"
        series = read_log_series(log)
        ev = None
        p = TRAIN_EVAL.get(name)
        if p and Path(p).exists():
            ev = json.loads(Path(p).read_text(encoding="utf-8")).get("report", {})
        table[name] = {"desc": desc, "dataset": ds, "series": series, "eval": ev}

    # ---- 打印 ----
    print("=" * 100)
    print("本轮全部 run 对照（训练集指标优先，dev 作参考）")
    print("=" * 100)
    print(f"{'run':<18}{'配置':<44}{'末轮训练集acc':>14}{'末轮dev acc':>12}{'loss':>8}")
    vals = []
    for name, rec in table.items():
        s = rec["series"]
        last = s[-1] if s else {}
        tacc = last.get("train", {}).get("choice_acc")
        dacc = last.get("dev", {}).get("choice_acc")
        loss = last.get("avg_loss")
        if tacc is not None:
            vals.append(tacc)
        print(f"{name:<18}{rec['desc']:<44}"
              f"{(f'{tacc:.4f}' if tacc is not None else '   --  '):>14}"
              f"{(f'{dacc:.4f}' if dacc is not None else '   --  '):>12}"
              f"{(f'{loss:.4f}' if loss is not None else '   --  '):>8}")

    # ---- 方差估计：只看"route 单任务全量、3 轮"这一类 ----
    print("\n" + "=" * 100)
    print("同类配置的离散度（判断单次运行能不能当结论）")
    print("=" * 100)
    same_class = []
    for name, rec in table.items():
        s = rec["series"]
        if not s:
            continue
        t = s[-1].get("train", {}).get("choice_acc")
        # "route 全量 3 轮"这一类
        if t is not None and "5296" in rec["desc"] and "3 轮" in rec["desc"]:
            same_class.append((name, t))
    if same_class:
        for n, v in same_class:
            print(f"  {n:<18} 训练集 acc {v:.4f}")
        if len(same_class) > 1:
            vs = [v for _, v in same_class]
            print(f"  极差 {max(vs) - min(vs):.4f}（{min(vs):.4f} ~ {max(vs):.4f}）")
            print("  → 极差 >0.05 说明单次运行不可信，所有对照都必须重复或看趋势")
    else:
        print("  没有足够的同类配置样本")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

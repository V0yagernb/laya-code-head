"""单卡训练 Laya 决策头（为 8GB 显存调过）。

对齐官方 notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb 的配方，改动只有三处，
都是为单卡 8GB 做的：

  1. 去掉 DDP（单卡不需要），micro batch 从 8 降到 4，靠累积维持有效 batch。
  2. 打开 encoder 的 gradient checkpointing 和 head 的 checkpointing。
  3. 加了 `--freeze-encoder`：只训决策头（约 40M 参数），forward 时用 detach_encoder
     切断梯度，显存降到 2~3GB，速度也快很多。第一版建议先用这个跑通。

损失函数一字未改，还是官方那套 RLCD：
    策略梯度项（对加噪 logits 采样 GROUP_SIZE 次，用严格恰当评分规则给奖励）
  + 软交叉熵项（对同一个目标分布）
我们用的是硬标签，所以 target 是 one-hot —— 官方用 teacher 软分布，这一项退化成普通交叉熵，
其余完全一致。

用法：
    # 先验证管线能跑通（2 步就停）
    python -m src.train --ds data/ds --smoke

    # 正式训练（冻结 encoder，最省显存）
    python -m src.train --ds data/ds --out out/run1 --freeze-encoder --epochs 4

    # 全量微调（吃满 8GB）
    python -m src.train --ds data/ds --out out/run2 --epochs 4
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from laya.common import build_model, proper_reward, QTYPES
    from laya.agent import _fix_tokenizer_config
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"需要先装 laya 包：pip install 'laya>=0.1.6' transformers\n原始错误：{e}")

# 复用评测里的 AUC 实现，避免"训练时说的 AUC"和"评测时报的 AUC"是两套口径
from src.evaluate import auc  # noqa: E402


# --------------------------------------------------------------------------
# 批次拼装（照抄官方，保证 marker / target 的对齐方式一致）
# --------------------------------------------------------------------------

def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


# --------------------------------------------------------------------------
# 温度校准（照抄官方 fit_one_temp）
# --------------------------------------------------------------------------

# laya 0.3.21 的 Agent 加载时只接受这个区间的温度，超出会警告并强行钳回边界
TEMP_MIN, TEMP_MAX = 0.5, 5.0


def fit_one_temp(sel):
    """在 log 温度上拟合一个标量温度，最小化 NLL。

    clamp 用的是 [0.5, 5] 而不是官方 notebook 的 [0.1, 10]：laya 0.3.21 的 Agent 在加载时
    会校验温度，超出 [0.5, 5] 就打印 RuntimeWarning 并强行钳回边界
    （实测 T=10 被改写成 5）。在比加载器更宽的范围里拟合，只会得到一个加载时被悄悄改掉的
    数字 —— 所以这里直接对齐加载器的可接受区间。
    """
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    # .detach()：log_t 带 requires_grad，直接 float() 会触发 UserWarning（无害但吵）
    return float(torch.clamp(log_t.detach().exp(), TEMP_MIN, TEMP_MAX))


# --------------------------------------------------------------------------
# 前向：抽出来是为了统一处理 detach_encoder 和 OOM 重试
# --------------------------------------------------------------------------

def forward_batch(model, batch, device, freeze_encoder):
    return model(
        batch["input_ids"].to(device, non_blocking=True),
        batch["attention_mask"].to(device, non_blocking=True),
        batch["marker_pos"].to(device, non_blocking=True),
        batch["marker_mask"].to(device, non_blocking=True),
        batch["qtype"].to(device, non_blocking=True),
        detach_encoder=freeze_encoder,
    )


def rlcd_loss(logits, batch, device, group_size, sigma, ce_weight=1.0, loss_qtype=None,
              adv_norm="mixed"):
    """官方 RLCD 损失：策略梯度项 + 软交叉熵项。

    ce_weight 是「额外一路普通交叉熵」的权重，默认 1.0 = 官方原样（数值一字未改）。

    loss_qtype 是排查用的开关：只让**指定 qtype 的样本**产生损失（其余在前向里照常
    参与，但不贡献梯度）。默认 None = 全部样本都算，行为与官方一致。

    **为什么需要它**：`DecisionModel` 的三个任务共用同一个 `scorer`（输出 1 维标量），
    任务类型只靠加性 `type_emb` 区分，所以三类任务的梯度会打在同一组共享参数上。
    隔离实验显示「单任务训练 route 的 macro-F1 0.397，而混合训练只有 0.188」；
    用这个开关就能判定：到底是**梯度互相干扰**（只反传 route 就能恢复），
    还是**批次里混着别的任务**本身就有害（只反传也恢复不了）。
    """
    logits = logits.float()
    mask = batch["marker_mask"].to(device)
    k = mask.sum(-1, keepdim=True).float()
    target = batch["target"].to(device)
    qtype = batch["qtype"].to(device)

    # 1. 采样 G 组带零均值噪声的 logits 投影
    eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    # 2. 严格恰当评分规则给奖励，组均值当 baseline
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
        if adv_norm == "grouped":
            # 按任务分组各自中心化 + 标准化（默认关闭：实测两种方式符号一致率 0.9833，
            # 差别极小，所以不改变官方行为，只作为可选项保留）
            r0 = r.mean(0)                                  # [N]
            adv0 = torch.zeros_like(r0)
            for qt in torch.unique(qtype):
                sel = qtype == qt
                g = r0[sel]
                adv0[sel] = (g - g.mean()) / (g.std() + 1e-6)
            adv = adv0.unsqueeze(0).expand_as(r)             # 广播回 [G, N]，与 mixed 分支同形

    # 3. 策略梯度 + 软交叉熵
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()

    if loss_qtype is not None:
        sel = (qtype == loss_qtype).float()
        n_sel = sel.sum().clamp(min=1.0)
        # 逐样本算好再按 qtype 掩码取平均；分母用真实选中数，避免被 batch 组成稀释
        per_rl = -((adv * logp).sum(-1) * sel).sum() / n_sel
        per_ce = -((target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1))
                   .sum(-1) * sel).sum() / n_sel
        return per_rl + ce_weight * per_ce, r.mean()

    return loss_rl + ce_weight * loss_ce, r.mean()


# --------------------------------------------------------------------------
# dev 早停
# --------------------------------------------------------------------------

# 早停判据：值越大越好。默认用 noul_auc —— 它阈值无关，而且 will_merge 是三个任务里
# 唯一有真实硬标签的（review_route 来自规则、change_scale 来自阈值分箱）。
EARLY_STOP_METRICS = {
    "noul_auc": "dev 上 will_merge 的 AUC",
    "choice_acc": "dev 上 review_route 的准确率",
    "neg_score_mae": "dev 上 change_scale 的 -MAE",
}


def save_ckpt_dir(path, model, tok, cfg, meta: dict):
    """把一个完整的、能被 `laya.load()` 直接加载的 checkpoint 写到目录里。

    必须包含 `rl_agent_config.json`：没有它官方加载器会直接报文件不存在 —— 而
    "训到一半先停下、以后接着用"恰恰最需要这个目录能直接加载。
    温度填未校准的默认值，真正的校准只在全部轮次跑完后的 `final/` 里做。
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(path / "model.safetensors"))
    model.encoder.config.save_pretrained(str(path / "encoder"))
    tok.save_pretrained(str(path / "tokenizer"))
    ckpt_cfg = copy.deepcopy(cfg)
    ckpt_cfg["fine_tuned"] = True
    ckpt_cfg["temperature"] = [1.2, 1.2, 1.2]
    ckpt_cfg["calibrated"] = False
    ckpt_cfg.pop("temperature_by_options", None)
    (path / "rl_agent_config.json").write_text(
        json.dumps(ckpt_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "checkpoint_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def eval_items(model, items, tok, device, micro: int = 8) -> dict:
    """在给定 items 上量三个任务的分指标（用训练时的批前向，不走 laya.load）。

    为什么不走 `laya.load()`：那要重新加载一个 842MB 的 checkpoint，每轮都做太贵。
    这里复用训练用的 collate/forward，口径和 evaluate.py 一致（同一个 AUC 实现）。
    """
    if not items:
        return {}
    model.eval()
    buckets = collections.defaultdict(list)
    with torch.no_grad():
        for i in range(0, len(items), micro):
            chunk = items[i:i + micro]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, _ = forward_batch(model, batch, device, freeze_encoder=True)
            logits = logits.float().cpu()
            for r, it in enumerate(chunk):
                buckets[it["qtype"]].append((logits[r, : len(it["markers"])].tolist(), it["label"]))
    model.train()

    out = {}
    if buckets[2]:                                     # noul → will_merge
        y = [1 if lab == 1 else 0 for _, lab in buckets[2]]
        p = [float(torch.softmax(torch.tensor(z), -1)[1]) for z, _ in buckets[2]]
        out["noul_auc"] = auc(y, p)
    if buckets[0]:                                     # choice → review_route
        ok = sum(int(int(torch.argmax(torch.softmax(torch.tensor(z), -1))) == lab)
                 for z, lab in buckets[0])
        out["choice_acc"] = ok / len(buckets[0])
    if buckets[1]:                                     # score → change_scale 的期望值
        err = 0.0
        for z, lab in buckets[1]:
            e = torch.softmax(torch.tensor(z), -1)
            err += abs(float(sum(i * float(v) for i, v in enumerate(e))) - lab)
        out["neg_score_mae"] = -(err / len(buckets[1]))
    return out


def stratified_limit(items, n, rng):
    """按 qid 分层取前 n 个 item，保证三个任务都被覆盖。

    **为什么必须这样取**：`build_dataset` 写出来的 items 是「按 case 铺开」的，
    而 route item 只占其中一部分、且**不是均匀分布**。实测 ds5 的 train_items.pt：
    第 0~5019 个位置只有 will_merge / change_scale 交替出现，**第一个 review_route
    在第 5020 个位置**。于是 `--limit-items 200` 取到的 200 条里一条 route 都没有 ——
    route 的准确率会永远停在「什么都没训」的常数上，用它去做 V5 的目标函数体检，
    对 route 完全没有检验力（这是计划 §V5 里一个没被发现的坑）。

    做法：按各 qid 的实际占比分配名额，再各自用同 seed 的 rng 抽样 —— 这样
    ①三个任务都在；②同一组 (items, n, seed) 每次抽到同一批，实验可复现。
    """
    if n <= 0 or n >= len(items):
        return list(items)
    by_q = collections.defaultdict(list)
    for it in items:
        by_q[it["qid"]].append(it)
    out = []
    for qid, group in sorted(by_q.items()):
        k = max(1, int(round(n * len(group) / len(items))))
        idx = list(range(len(group)))
        rng.shuffle(idx)
        out.extend(group[i] for i in sorted(idx[:min(k, len(group))]))
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/ds")
    ap.add_argument("--out", default="out/run")
    ap.add_argument("--model-id", default="convaiinnovations/laya")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="只训决策头，显存 2~3GB，推荐第一次跑用这个")
    ap.add_argument("--no-checkpointing", action="store_true",
                    help="关掉梯度检查点（更快，但更吃显存）")
    ap.add_argument("--smoke", action="store_true", help="只跑 2 个 step 验证管线")
    ap.add_argument("--early-stop-metric", default="noul_auc", choices=sorted(EARLY_STOP_METRICS),
                    help="用 dev 上哪个指标挑最优轮次：" +
                         "；".join(f"{k}={v}" for k, v in sorted(EARLY_STOP_METRICS.items())))
    ap.add_argument("--no-early-stop", action="store_true",
                    help="关掉 dev 早停，跑满 --epochs（不建议：固定轮数容易挑到过拟合的那一轮）")
    ap.add_argument("--resume", default="", metavar="CKPT_DIR",
                    help="从某个滚动 checkpoint 继续训练。--epochs 表示总轮数：已存第 5 轮、"
                         "想跑满 10 轮就传 --epochs 10 --resume out/run3/checkpoint_latest。"
                         "注意优化器状态没有保存，这是带权重重新起步")
    ap.add_argument("--limit-items", type=int, default=0, help="只取前 N 个 item（调试）")
    ap.add_argument("--ce-weight", type=float, default=1.0,
                    help="额外一路普通交叉熵的权重。1.0 = 官方 RLCD 原样（默认，行为不变）；"
                         "V5 目标函数体检时把它调大（比如 5.0）看训练集指标会不会跳升")
    ap.add_argument("--loss-qid", default="", choices=["", "will_merge", "review_route", "change_scale"],
                    help="排查用：只让指定任务的样本产生梯度（其余照常前向）。"
                         "空=全部任务都算（默认、官方行为）。用来判定混合训练差是不是梯度互相干扰")
    ap.add_argument("--adv-norm", default="mixed", choices=["mixed", "grouped"],
                    help="优势归一化的方式。mixed=官方原样（整个 micro-batch 全平铺算均值/标准差）；"
                         "grouped=按 qtype 分组归一化。默认 mixed，行为与官方一致")
    ap.add_argument("--eval-train-items", type=int, default=1024,
                    help="每轮额外在训练集的一个分层子集上量指标（0=不量）。"
                         "不量的话只能看到 dev/早停指标，看不出「训练集到底拟合到什么程度」——"
                         "而这恰恰是判断欠拟合 vs 过拟合的关键（计划 M3 第 1 条要的也是这个）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", ""))
    args = ap.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --loss-qid 是排查用的：把 qid 翻成 qtype 数字（QTYPES: choice=0, score=1, noul=2）
    loss_qtype = None
    if args.loss_qid:
        from src.task import QUESTIONS as _Q
        loss_qtype = QTYPES[_Q[args.loss_qid]["type"]]
        print(f"[!] 只反传 {args.loss_qid}（qtype={loss_qtype}）的损失，其余任务不产生梯度")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"显存 {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    else:
        print("警告：没检测到 CUDA，将在 CPU 上训练（会非常慢）")

    ds_dir = Path(args.ds)
    train_items = torch.load(ds_dir / "train_items.pt", weights_only=False)
    calib_items = torch.load(ds_dir / "calib_items.pt", weights_only=False)
    # dev 切片：用来做早停判据。老数据集可能没有这个文件，缺了就直接关掉早停。
    dev_items = []
    dev_path = ds_dir / "dev_items.pt"
    if dev_path.exists():
        dev_items = torch.load(dev_path, weights_only=False)
    if args.limit_items:
        # 分层抽样：老写法 `train_items[:N]` 在 ds5 上取不到任何 route item
        # （第一个 route 在第 5020 个位置），详见 stratified_limit 的注释。
        # 用固定 seed 的独立 rng，保证同一组参数抽到同一批，实验可复现。
        lim_rng = random.Random(args.seed)
        train_items = stratified_limit(train_items, args.limit_items, lim_rng)
        calib_items = stratified_limit(calib_items, max(16, args.limit_items // 3), lim_rng)
        dev_items = stratified_limit(dev_items, max(16, args.limit_items // 3), lim_rng)
        dist = collections.Counter(it["qid"] for it in train_items)
        print(f"  [limit-items 分层抽样] 训练项三类分布 {dict(sorted(dist.items()))}")
    print(f"训练项 {len(train_items)} | 校准项 {len(calib_items)} | dev 项 {len(dev_items)}")

    # 训练集评测子集：和 dev 一起每轮量一次。
    # 为什么要它：只看 dev 分不清「模型没拟合训练集」（欠拟合）和「过拟合了」——
    # 而这一轮的核心结论恰恰是「模型在自己的训练集上都学不动」，必须能直接量到。
    eval_train_items = []
    if args.eval_train_items:
        eval_train_items = stratified_limit(
            train_items, args.eval_train_items, random.Random(args.seed + 999))
        print(f"  另外每轮在训练集分层子集 {len(eval_train_items)} 项上量指标")
    if args.no_early_stop or not dev_items:
        print("  [!] 早停已关闭：会一直跑满 --epochs 轮。"
              "上一轮实验就是固定跑满导致测试集变差的（见任务文档 5.5）")

    # ---- 模型 ----
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # 续训：从某个滚动 checkpoint 接着跑。--epochs 在这里表示"总共要跑到第几轮"，
    # 比如已经存到第 5 轮、想跑满 10 轮，就用 --epochs 10 --resume out/run3/checkpoint_latest。
    #
    # ⚠️ 诚实说明：优化器状态（AdamW 的动量/方差）和调度器进度**没有**被保存过，
    #    所以这是"带着权重重新起步"，不是逐比特还原的训练状态。对决策头这种小参数量
    #    的微调影响不大，但严格来说不等价于"从没停过"。
    resume_dir = Path(args.resume) if args.resume else None
    start_epoch = 0
    weights_dir = None
    if resume_dir:
        if not (resume_dir / "model.safetensors").exists():
            raise SystemExit(f"--resume 指向的目录里没有 model.safetensors：{resume_dir}")
        meta_path = resume_dir / "checkpoint_meta.json"
        if meta_path.exists():
            start_epoch = int(json.loads(meta_path.read_text(encoding="utf-8")).get("epoch", 0))
        if start_epoch >= args.epochs:
            raise SystemExit(f"该 checkpoint 已完成 {start_epoch} 轮，"
                             f"--epochs={args.epochs} 没有剩下可跑的轮次")
        weights_dir = resume_dir
        print(f"续训：权重取自 {resume_dir}（已完成 {start_epoch} 轮，"
              f"本次跑到第 {args.epochs} 轮）")

    print(f"准备 checkpoint：{args.model_id}")
    if os.path.isdir(args.model_id):
        model_dir = args.model_id
        print(f"  用本地目录 {model_dir}")
    else:
        model_dir = snapshot_download(args.model_id)
    _fix_tokenizer_config(model_dir)

    # tokenizer / 配置优先用续训目录里的（那是训练时真正写出来的那份）
    tok_dir = model_dir
    cfg_dir = model_dir
    enc_dir = os.path.join(model_dir, "encoder")
    if resume_dir:
        if (resume_dir / "tokenizer").exists():
            tok_dir = str(resume_dir)
        if (resume_dir / "rl_agent_config.json").exists():
            cfg_dir = str(resume_dir)
        if (resume_dir / "encoder").exists():
            enc_dir = str(resume_dir / "encoder")

    tok = AutoTokenizer.from_pretrained(os.path.join(tok_dir, "tokenizer"))
    with open(os.path.join(cfg_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    cfg["gradient_checkpointing"] = not args.no_checkpointing
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = args.head_max_len
    cfg["head_max_len_train"] = args.head_max_len

    model = build_model(cfg, encoder_dir=enc_dir)
    weights = load_file(os.path.join(weights_dir or model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    if not args.no_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
        print("已开启梯度检查点")

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"冻结 encoder，只训 {n_train/1e6:.1f}M 参数")

    model.to(device)
    model.train()

    enc_params = [p for n, p in model.named_parameters()
                  if "encoder." in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if "encoder." not in n and p.requires_grad]
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_encoder})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    print(f"可训参数组：encoder {len(enc_params)} 个张量 / head {len(head_params)} 个张量")

    steps_per_epoch = max(1, len(train_items) // (args.micro_batch * args.grad_accum))
    # 续训时余弦调度只覆盖"还剩下的轮次"，否则学习率会按早已跑完的总步数一路衰减到底
    remaining_epochs = max(1, args.epochs - start_epoch)
    total_updates = steps_per_epoch * remaining_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    SIGMA_START, SIGMA_END = 0.4, 0.1
    micro = args.micro_batch
    t0 = time.time()
    global_step = 0
    stopped = False
    best_score, best_epoch = float("-inf"), 0

    for epoch in range(start_epoch, args.epochs):
        if stopped:
            break
        random.seed(args.seed + epoch)
        random.shuffle(train_items)
        progress = epoch / max(1, args.epochs - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        epoch_loss, n_batches, accum = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        i = 0
        while i < len(train_items) and not stopped:
            chunk = train_items[i:i + micro]
            i += micro
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)
            try:
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    logits, act = forward_batch(model, batch, device, args.freeze_encoder)
                loss, reward = rlcd_loss(logits, batch, device, args.group_size, sigma,
                                         ce_weight=args.ce_weight,
                                         loss_qtype=loss_qtype,
                                         adv_norm=args.adv_norm)
                loss = loss / args.grad_accum + 0.0 * act.sum()
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                # 显存不够就当场减半重试，而不是整个 run 挂掉
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro <= 1:
                    print("micro batch 已经降到 1 还是 OOM，请加 --freeze-encoder 或降 --max-len")
                    raise
                micro = max(1, micro // 2)
                print(f"  [!] OOM，micro batch 降到 {micro} 并重试")
                i -= len(chunk)
                continue

            accum += 1
            if accum % args.grad_accum == 0 or i >= len(train_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 20 == 0:
                    vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                    print(f"  ep{epoch+1} step{global_step} loss={loss.item()*args.grad_accum:.4f} "
                          f"reward={reward.item():.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                          f"peakVRAM={vram:.2f}GB")

            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            if args.smoke and n_batches >= 2:
                print("  [smoke] 跑满 2 个 step，停止训练循环")
                stopped = True
                break

        print(f"=== epoch {epoch+1}/{args.epochs} 用时 {time.time()-t0:.0f}s "
              f"平均 loss {epoch_loss/max(1,n_batches):.4f} ===")

        # 每轮存一个滚动 checkpoint，崩了不至于全丢
        save_ckpt_dir(out_dir / "checkpoint_latest", model, tok, cfg, {
            "epoch": epoch + 1, "total_epochs": args.epochs,
            "avg_loss": epoch_loss / max(1, n_batches),
            "calibrated": False,
            "note": "中途滚动存档：温度未校准，confidence 不可用于门禁",
        })

        # ---- dev 早停：固定跑满 N 轮是不对的 ----
        # 上一轮实验（任务文档 5.5）就是这样踩的坑：第 10 轮在训练集上 AUC 0.8152、
        # 但跨仓库测试集反而比第 5 轮差。所以每轮都量一次 dev，只保留 dev 最好的那份。
        if dev_items and not args.no_early_stop:
            m = eval_items(model, dev_items, tok, device)
            shown = " ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()))
            score = m.get(args.early_stop_metric)
            # 同一轮也在训练集子集上量一次：这是区分「欠拟合」和「过拟合」的唯一直接证据
            if eval_train_items:
                mt = eval_items(model, eval_train_items, tok, device)
                print("  训练集: " + " ".join(f"{k}={v:.4f}" for k, v in sorted(mt.items())))
            if score is None:
                print(f"  dev 指标 {shown}（缺 {args.early_stop_metric}，跳过早停判据）")
            else:
                flag = ""
                if score > best_score:
                    best_score, best_epoch = score, epoch + 1
                    save_ckpt_dir(out_dir / "best", model, tok, cfg, {
                        "epoch": epoch + 1, "total_epochs": args.epochs,
                        "avg_loss": epoch_loss / max(1, n_batches),
                        "dev_metrics": m,
                        "early_stop_metric": args.early_stop_metric,
                        "early_stop_score": score,
                        "calibrated": False,
                        "note": "dev 最优存档：温度未校准",
                    })
                    flag = f"  ↑ 新的 dev 最优，已存 best/"
                print(f"  dev: {shown} | 早停判据 {args.early_stop_metric}={score:.4f}"
                      f"（历史最优 {best_score:.4f} @ 第{best_epoch}轮）{flag}")
        print(f"  已存滚动 checkpoint → {out_dir / 'checkpoint_latest'}")

    # ---- 用 dev 最优的那一轮覆盖当前权重 ----
    # 这样 final/ 存下来的、以及后面拟合温度的，都是"dev 最好"的那份，而不是"最后那一轮"。
    # 上一轮实验的教训就在这里：最后一轮在训练集上更好，跨仓库测试集反而更差。
    best_ckpt = out_dir / "best" / "model.safetensors"
    if best_epoch and best_epoch != args.epochs and best_ckpt.exists():
        print(f"\n用 dev 最优的第 {best_epoch} 轮权重覆盖当前权重"
              f"（{args.early_stop_metric}={best_score:.4f}；最后一轮是第 {args.epochs} 轮）")
        model.load_state_dict(load_file(str(best_ckpt)), strict=True)
        model.to(device)
    elif best_epoch:
        print(f"\ndev 最优就是最后一轮（第 {best_epoch} 轮，"
              f"{args.early_stop_metric}={best_score:.4f}），无需回退")

    # ---- 温度校准 ----
    print("\n拟合校准温度（在训练前就留出的 calib 切片上）...")
    model.eval()
    calib_preds = []
    if calib_items:
        with torch.no_grad():
            for c in range(0, len(calib_items), 16):
                cb = collate_train_batch(calib_items[c:c + 16], tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    l_sub, _ = forward_batch(model, cb, device, args.freeze_encoder)
                l_np = l_sub.float().cpu().numpy()
                for r, it in enumerate(calib_items[c:c + 16]):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r, :k], it["target"]))

    fitted = [1.2, 1.2, 1.2]
    try:
        for qt in range(3):
            sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
            if sel:
                fitted[qt] = fit_one_temp(sel)
        print("校准温度 (choice, score, noul):", [round(t, 3) for t in fitted])
    except Exception as e:
        print("温度拟合失败，用默认 1.2：", e)

    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(final / "model.safetensors"))
    model.encoder.config.save_pretrained(str(final / "encoder"))
    tok.save_pretrained(str(final / "tokenizer"))

    saved_cfg = copy.deepcopy(cfg)
    saved_cfg["fine_tuned"] = True
    saved_cfg["temperature"] = fitted
    # 这一条是官方文档特别点名的坑：继承来的 bucket 温度会在推理时覆盖新拟合的值
    saved_cfg.pop("temperature_by_options", None)
    (final / "rl_agent_config.json").write_text(
        json.dumps(saved_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # 记一下训练元信息，方便复现
    (final / "train_meta.json").write_text(json.dumps({
        "base": args.model_id,
        "epochs": args.epochs,
        "resumed_from": str(resume_dir) if resume_dir else None,
        "resumed_at_epoch": start_epoch,
        "epochs_run_this_time": args.epochs - start_epoch,
        "micro_batch_final": micro,
        "grad_accum": args.grad_accum,
        "group_size": args.group_size,
        "ce_weight": args.ce_weight,
        "freeze_encoder": args.freeze_encoder,
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "train_items": len(train_items),
        "calib_items": len(calib_items),
        "dev_items": len(dev_items),
        "early_stop_metric": args.early_stop_metric,
        "best_epoch": best_epoch,
        "best_score": (round(best_score, 4) if best_epoch else None),
        "wall_time_sec": round(time.time() - t0, 1),
        "temperatures": fitted,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n训练完成 → {final}")
    print(f"用法：laya.load('{final.as_posix()}'） 或 python -m src.predict --ckpt {final.as_posix()}")


if __name__ == "__main__":
    main()

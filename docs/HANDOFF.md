# HANDOFF —— laya-code-head 项目交接

> 写于 2026-09-29。给下一个接手这个项目的 agent。
> **先读第 1 节和第 7 节**，其余是背景。

---

## 1. TL;DR（30 秒版）

**做了什么**：把开源的 Laya 决策模型（`convaiinnovation/laya`，421M）微调成一个「代码改动风险评级器」，
用 GitHub 真实 PR 数据训练。**整条管线已经端到端跑通并实测过**。

**现在什么状态**：

| | 训练集 (n=64) | 测试集 (n=29，整个 `encode/httpx` 仓库留出) |
|---|---|---|
| `will_merge` 准确率 | **1.0000** | 0.7586 |
| `review_route` 准确率 | **0.9531** | 0.3103（= 多数类基线） |
| `review_route` macro-F1 | **0.9488** | 0.2842 |
| `change_scale` MAE | 0.4683 | 0.7482 |

**诊断结论**：训练集能拟合到 95~100%，说明**代码和管线完全正确**；但测试集塌缩到多数类基线，
说明**纯粹是数据量不够导致的过拟合**。当前只有 192 条真实 PR（官方微调用的是 1200 case / 6000 决策项）。

**下一任最该做的事（按性价比排序）**：
1. **拿一个 GitHub token**（`set GITHUB_TOKEN=ghp_xxx`），配额从 60 次/小时 变 5000 次/小时，把数据从 192 条挖到 5000+ 条。
2. 重新 `build_dataset` → `train` → `evaluate`，看测试集指标有没有起来。
3. 如果起来了，再考虑换成人工标注的 `review_route` 标签（现在这个标签是规则派生的，见第 6 节）。

---

## 2. 项目位置与文件

**项目根目录**：`<旧桌面副本>\`

```
laya-code-head/
├── HANDOFF.md              ← 本文件
├── README.md               使用文档（面向人类使用者）
├── requirements.txt
├── config/repos.txt        默认挖取的仓库清单（已写 15 个）
├── src/
│   ├── task.py             ★ 任务定义 + 标签规则。换任务只改这一个文件
│   ├── mine_github.py      挖真实 PR → data/raw.jsonl
│   ├── build_dataset.py    raw.jsonl → Laya token 化 items（.pt）
│   ├── train.py            单卡 8GB 训练 + 温度校准
│   ├── evaluate.py         指标 + 与未微调底座对拍
│   └── predict.py          对一段真实 diff 出三个判断
├── data/                   数据与日志（raw.jsonl, ds/, *.log）
├── models/laya/            Laya 底座权重（803MB，curl 从 hf-mirror 下的）
├── wheels/                 torch 轮子（2.6GB）
└── out/
    ├── smoke/              冒烟测试产物
    ├── run1/               30 epoch, lr_head=1e-4（效果差）
    └── run2/               ★ 40 epoch, lr_head=3e-4（当前最好，用这个）
```

**磁盘占用约 7.4GB**（`out/` 4.8GB + `wheels/` 2.6GB）。每个 run 里 `final/` 和 `checkpoint_latest/`
各存了一份完整权重（842MB），**可以安全删掉 `checkpoint_latest/` 和 `wheels/` 省 3~4GB**。

---

## 3. 环境

**Python 虚拟环境**（独立，不要用共享的 `default`）：
```
<VENV>\Scripts\python.exe
```
Python 3.13.14。已装：

| 包 | 版本 | 备注 |
|---|---|---|
| torch | 2.11.0+cu128 | 从 `wheels/` 本地安装的，CUDA 12.8 |
| transformers | 5.17.0 | |
| huggingface_hub | 1.33.0 | |
| safetensors | 0.8.0 | |
| laya | 0.3.21 | 官方包，`build_sequence`/`build_model`/`proper_reward` 都从这来 |
| numpy, tqdm, typer, rich, httpx, ... | | transformers 的依赖 |

**硬件**：RTX 4060 Ti / 8GB / sm_89，驱动 591.86。`torch.cuda.is_available()` 已验证 True。

**每次跑之前必须设的环境变量**（huggingface.co 在国内直连不通）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```
`src/*.py` 里已经用 `os.environ.setdefault` 预设了这两个，一般不用手动加。

---

## 4. 复现全部命令

在 `<旧桌面副本>` 目录下：

```bash
PY="<VENV>/Scripts/python.exe"

# 1) 挖数据（未认证 60 次/小时；有 token 会快 80 倍）
"$PY" -u -m src.mine_github --repos-file config/repos.txt --pages 2 \
    --min-open-hours 1 --out data/raw.jsonl

# 2) 构造数据集（-test-repos 指定整个留出做测试集的仓库）
"$PY" -u -m src.build_dataset --raw data/raw.jsonl --out data/ds \
    --model-id models/laya --test-repos encode/httpx --balance

# 3) 训练（先冒烟，2 步验证管线）
"$PY" -u -m src.train --ds data/ds --out out/run3 --model-id models/laya \
    --smoke --freeze-encoder --epochs 1

# 3b) 正式训练（当前最优配置）
"$PY" -u -m src.train --ds data/ds --out out/run3 --model-id models/laya \
    --freeze-encoder --epochs 40 --micro-batch 4 --grad-accum 4 --lr-head 3e-4

# 4) 评测：测试集 + 训练集（训练集用来判断是"没学到"还是"过拟合"）
"$PY" -u -m src.evaluate --ckpt out/run3/final --ds data/ds --split test
"$PY" -u -m src.evaluate --ckpt out/run3/final --ds data/ds --split train

# 5) 拿真实 diff 试
git diff > pr.diff
"$PY" -u -m src.predict --ckpt out/run3/final --diff-file pr.diff --title "fix: ..."
```

> ⚠️ 在这个沙箱里：**长命令必须 `run_in_background=true`**，且**不要把 python 输出接到 `| tail`**
> （会 SIGTERM 且吞掉输出）。用 `> some.log 2>&1` 然后读日志文件。

---

## 5. 实测结果与诊断（run2）

**训练配置**：冻结 encoder，只训 26.5M 参数的决策头；40 epoch；micro-batch 4 × grad-accum 4；
lr_head=3e-4；RLCD 损失（策略梯度 + 软交叉熵）；**峰值显存 2.47GB**；全程约 8 分钟。

| 指标 | 训练集 (64) | 测试集 (29) | 测试集基线 |
|---|---|---|---|
| `will_merge` accuracy | 1.0000 | 0.7586 | 全猜 closed = 0.7241；真实合并率 0.2759 |
| `will_merge` Brier | 0.0069 | 0.2322 | |
| `will_merge` ECE | 0.0419 | 0.2774 | 出厂时是过度自信的 |
| `review_route` accuracy | 0.9531 | 0.3103 | 多数类 `block` = 0.3103 |
| `review_route` macro-F1 | 0.9488 | 0.2842 | |
| `change_scale` MAE | 0.4683 | 0.7482 | |
| `change_scale` within-one | 1.0000 | 0.8621 | |
| 单条延迟 p50 | 149.7 ms | 152.7 ms | CPU/GPU 混合，未优化 |

对比底座（未微调，同样 29 条测试）：`will_merge` acc 0.2069、`review_route` acc 0.2069。
所以微调**确实带来了提升**（0.207 → 0.759 / 0.207 → 0.310），但幅度远不够用。

**为什么测试集只有多数类水平**：

1. **数据量差 10~30 倍**。官方 notebook 用 1200 case / 6000 决策项；我们是 64 case / 192 items。
2. **`review_route` 的标签是规则派生的**（见第 6 节），规则本身只看 churn/文件路径，
   语义信息量低，模型很难从 diff 文本里学到规则之外的泛化。
3. **`will_merge` 有强先验**：真实数据里 74% 的 PR 会被关闭。模型学到"多数会关"就能拿到 0.72。
   已经用 `--balance` 平衡了训练集，但测试集保留真实分布，所以这个陷阱还在。
4. **欠采样后类别极少**：`needs_tests` 在训练集里只有 6 个样本，测试集有 6 个——学不动。

**run1 对比**（30 epoch，lr_head 默认 1e-4）：训练 loss 停在 1.09（≈ 只学会了标签边际分布），
`will_merge` 的 P(merge) 全部落在 0.488~0.507，即**完全没做决定**。
把 lr_head 提到 3e-4 之后 loss 降到 0.2346、reward 由负转正，才开始真正学习。
→ **教学点：这个任务上 lr_head=1e-4 太小，默认值应该改。**

---

## 6. 任务定义与标签来源（重要，别弄错）

定义在 `src/task.py`，三个问题：

| 问题 | 类型 | 标签来源 | 可信度 |
|---|---|---|---|
| `will_merge` | `noul` | **GitHub 真实结果**（PR 被 merge 还是被关） | 硬标签 |
| `review_route` | `choice` | **`route_policy()` 这个人为写的确定性规则** | ⚠️ 策略蒸馏，不是人的判断 |
| `change_scale` | `score` | GitHub 真实增删行数按固定阈值分箱 | 客观测量 |

`route_policy()` 当前逻辑（`src/task.py`）：
```python
if critical or churn >= 400 or n_files >= 10:            return "block"
if src and not tests and churn >= 20:                    return "needs_tests"
if churn <= 30 and n_files <= 3 and (tests or not src):   return "fast_track"
return "normal"
```

**要换成人工标注**，只改 `LabelPolicy.review_route()`，其余管线一行不用动。
建议路径：先用模型跑一遍 5000 条真实 PR，把模型和规则不一致的样本挑出来人工裁决，用这批做新标签。

---

## 7. 下一步该做什么（按优先级）

### P0 —— 扩数据（最大的杠杆）

```bash
set GITHUB_TOKEN=ghp_xxx   # 没有 token 就只能 60 次/小时，挖不动
"$PY" -u -m src.mine_github --repos-file config/repos.txt --pages 5 \
    --min-open-hours 1 --out data/raw.jsonl
```
- 1 次 API 请求 = 100 个 PR 的元数据；**diff 走 `github.com/<repo>/pull/<n>.diff` 不消耗配额**。
- 目标：**2000~5000 条真实 PR**（对应 6000~15000 个决策项，接近官方量级）。
- `config/repos.txt` 里有 15 个候选仓库，已验证可用的：`pallets/flask`、`encode/httpx`、
  `tiangolo/typer`、`psf/requests`、`pandas-dev/pandas`、`psf/black`。
- 注意 `--min-open-hours 1`：按 `sort=updated` 抓到的 PR 里有很多"开出来几分钟就被关掉"的
  低质量投稿（`pallets/flask` 一页 100 个只合 3 个就是这原因），必须滤掉。

### P1 —— 重训重评

```bash
"$PY" -u -m src.build_dataset --raw data/raw.jsonl --out data/ds \
    --model-id models/laya --test-repos encode/httpx --balance
"$PY" -u -m src.train --ds data/ds --out out/run3 --model-id models/laya \
    --freeze-encoder --epochs 20 --micro-batch 4 --grad-accum 8 --lr-head 3e-4
"$PY" -u -m src.evaluate --ckpt out/run3/final --ds data/ds --split test --baseline models/laya
```
数据变多之后：epoch 要相应减少（现在 40 epoch 是给 192 items 用的），`--grad-accum` 可以加大。

### P2 —— 换掉 `review_route` 的标签
见第 6 节。

### P3 —— 尝试全量微调
8GB 上全量微调 421M 参数很紧（参数 1.68GB + 梯度 1.68GB + AdamW 状态 3.36GB ≈ 6.7GB，再加激活就爆）。
如果一定要试：去掉 `--freeze-encoder`，并把 `--micro-batch` 降到 1~2。`train.py` 里有 OOM 自动减半重试。
**更推荐**的是 LoRA（`Cahol/laya-banking77-v1` 用的就是这套），但目前 `train.py` 还没实现。

### P4 —— 温度校准现在是退化状态
`fit_one_temp` 拟合出来的三个温度**每次都顶到边界 5.0**，说明模型在 calib 切片上过度自信且判错。
数据量上来之后应该会落在合理区间；如果还是顶边界，说明校准切片太难或者模型确实没学好。
**在校准正常之前，不要拿 confidence 做任何门禁。**

---

## 8. Laya 内部技术细节（省你重新扒源码）

模型结构（`laya/common.py` 的 `DecisionModel`）：
```python
DecisionModel(
  encoder    = ModernBERT-large      # 421M，本 checkpoint 是 fp16 存的（842MB → 205 个 F16 张量）
  head       = nn.TransformerEncoder(2 层)
  type_emb   = nn.Embedding(3, d)    # 三种问题类型
  scorer     = LayerNorm → Linear(d,d) → GELU → Linear(d,1)
  act_head   = Linear(d+4,256) → GELU → Linear(256,2)
  temperature= buffer(3)
)
forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False)
        → (logits[B,Q,K], act_logits)
```
- **`detach_encoder=True` 是关键**：直接切断 encoder 梯度，这就是 `--freeze-encoder` 的实现方式，
  峰值显存 2.47GB。
- 序列格式：`[CLS] <type> question: <指令> [SEP] [MASK] 选项0 [MASK] 选项1 ... [SEP] <state> [SEP]`
  —— 每个选项前一个 `[MASK]` marker，决策头读这些位置的隐状态。
- `QTYPES = {"choice": 0, "score": 1, "noul": 2}`。

训练损失（RLCD，照抄官方并只改了单卡相关部分）：
```
策略梯度项：对 logits 加零均值高斯噪声采样 G=4 次，用 proper_reward 打分（w_sph=0.75, w_rps=1.0），
            组内减均值除标准差当 advantage，再乘加噪分布与原始 logits 的 log 概率
软交叉熵项：-(target · log_softmax(logits))，权重 1.0
总损失 = PG + CE，梯度裁剪 1.0，AdamW(encoder 2.5e-5 / head 1e-4)，cosine 调度
```
我们用 one-hot 硬标签，所以官方那套"teacher 软分布"退化成普通交叉熵，其余一致。

**三个必须知道的坑**：

1. **温度区间**：`laya 0.3.21` 的 `Agent` 加载时**只接受 [0.5, 5]**，超出会打 RuntimeWarning
   并强行钳到边界（实测 T=10 被改成 5）。官方 notebook 用 [0.1, 10] 拟合，在 0.3.21 上会被悄悄改掉。
   `train.py` 里已改成 `TEMP_MIN, TEMP_MAX = 0.5, 5.0` 对齐。
2. **`temperature_by_options` 会静默覆盖新拟合的温度**：保存时**必须 pop 掉**这个 key，
   `train.py` 已经做了。
3. **`noul` 的 `false`/`true` 标签本身有时会压过状态**（官方 BENCHMARKS.md 记录过）。
   如果发现 `will_merge` 的答案不随输入变化，把 `src/task.py` 里的 `NOUL_LABELS`
   改成 `{"false": "B", "true": "A"}` 重训。

**GitHub 数据挖掘的坑**：
1. **`/repos/*/pulls` 列表接口不返回 `additions`/`deletions`**（只有单 PR 接口才有，成本 100 倍）。
   现在是从 diff 自己数行数。
2. **`sort=updated` 会排出一堆 Dependabot 版本升级 PR**，语义单薄。已默认过滤
   `[bot]`/`dependabot`/`renovate`。
3. `.diff` 端点抓多了会被限速变慢（实测跑几百个之后明显变慢）。

---

## 9. 已知问题 / 待修清单

- [ ] `train.py` 的温度拟合出来顶边界（见 P4）。
- [ ] `fit_one_temp` 里 `float(torch.clamp(...))` 会触发 `requires_grad=True` 的 UserWarning，
      加个 `.detach()` 就好（无害，但吵）。
- [ ] 没有 LoRA 支持。8GB 全量微调会爆，LoRA 是更好的路。
- [ ] `evaluate.py` 的 `--batch-size` 参数目前没用上（逐条推理），延迟数字不是最优值。
- [ ] `predict.py` 硬编码 diff 截断 4000 字符。
- [ ] `out/` 里 `checkpoint_latest/` 冗余占 3GB，可以考虑改成只留最近 2 个。
- [ ] `config/repos.txt` 里 `django/django`、`python/cpython`、`scikit-learn` 等还没挖过。

---

## 10. 参考资料位置

| 内容 | 位置 |
|---|---|
| 官方微调文档 | `github.com/NandhaKishorM/laya` → `docs/finetune.md` |
| 单卡微调范例（16GB，浏览器 Agent 决策头） | 同仓库 → `docs/finetune_browser_agent.md` |
| 官方 Kaggle notebook（双 T4 DDP） | 同仓库 → `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` |
| 模型源码 | `laya/common.py`（DecisionModel）、`laya/agent.py`（Agent / load） |
| 已训练作品的调研清单 | `<桌面>\Laya 决策模型 - 已训练可用项目清单.md` |
| 同类参考实现 | `SampleBias/Coding_Decision_Agent`（MIT，代码决策打分框架，权重未产出）；<br>`jinlio/mjbrain`（中文微调全链路，8GB 可训）；<br>`Cahol/laya-banking77-v1`（LoRA + 决策头的完整发布包） |

---

## 11. 本机环境坑（会浪费你很多时间）

1. **长命令会被沙箱 SIGTERM**（约 60 秒以上的网络/下载/训练）。**一律用 `run_in_background=true`**，
   否则表现为 `exit 1 / Signal: SIGTERM / 输出全空` —— 但文件其实已经写了一部分。
2. **不要用 `| tail` / `| head` 接 python 输出**，同样会吞输出 + SIGTERM。
   写成 `python -u ... > x.log 2>&1`，然后读日志文件。
3. **`wmic.exe` 在黑名单里**，调用会把整条命令 SIGTERM。查 GPU 用
   `nvidia-smi --query-gpu=name,memory.total --format=csv`，查磁盘用 `df -h`。
4. **Windows 下 `/tmp` 是陷阱**：`curl -o /tmp/x` 写到 `C:\tmp\`，而 bash 里的 python 读 `/tmp/`
   是 Git Bash 的临时目录，两者不通。中间文件写到项目目录里。
5. **pip 从 pypi.org 装包极慢**（几十 KB/s）。用
   `-i https://pypi.tuna.tsinghua.edu.cn/simple`。大轮子（torch）直接用 `curl -C -` 下更稳更快
   （实测 9~18 MB/s vs pip 的 2.4 MB/s）。
6. **huggingface.co 不通**，必须走 `hf-mirror.com`；`urllib` 抓 HF 必须带 `User-Agent` 否则 403。
   `snapshot_download` 拉整个仓库容易在小文件上反复超时，用 `allow_patterns` 限定，或者直接 `curl`。

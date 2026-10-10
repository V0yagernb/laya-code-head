# laya-code-head —— 代码改动风险评级

把开源的 [Laya](https://github.com/NandhaKishorM/laya) 决策模型微调成一个**代码审查判断器**：
给它一段 diff，它在一次前向传播里直接吐出三个结构化判断，不生成任何文本。

```
diff  ──►  [审核路由]  fast_track / normal / needs_tests / block
            [能否合并]  P(merge) = 0.72
            [改动规模]  2.34 → moderate
```

## 为什么做这个

调研过 Hugging Face + GitHub 上所有 Laya 生态后（见 `Laya 决策模型 - 已训练可用项目清单.md`），
代码方向**真训练过的成品只有 3 个**，而且都在边缘位置：

| 已有 | 做什么 |
|---|---|
| `tindang/laya-code` | 代码相关性 reranker，配合 Claude Code 用 |
| `wxsys/code-oracle-laya-421m` | 代码验证 APPROVED/REJECTED |
| `leehack/laya-command-head` | 8 类命令意图 |

**"改动风险评级 / review 路由"这个位置是空的。** `SampleBias/Coding_Decision_Agent`（MIT）
专门为此写了训练框架，但作者明确说权重还没产出来。这个项目就是补上这一格。

## 硬件要求

实测开发机：**RTX 4060 Ti / 8GB**，够用。

| 模式 | 显存 | 说明 |
|---|---|---|
| `--freeze-encoder`（默认推荐） | ~2–3 GB | 只训决策头，几十分钟 |
| 全量微调 | ~7–8 GB | 靠梯度检查点 + micro batch 4 |

没显卡也能跑推理（慢），或者把训练丢到 Kaggle 免费 2×T4。

## 快速开始

```bash
# 0. 环境（Python 3.10+，独立 venv）
PY="<PYTHON_BASE>/python.exe"
"$PY" -m venv "<VENV>"
PIP="<VENV>/Scripts/python.exe"

# 4060 Ti 是 sm_89，cu128 轮子直接支持
"$PIP" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"$PIP" -m pip install "laya>=0.1.6" "transformers>=4.48" huggingface_hub safetensors

# 国内必须挂镜像（huggingface.co 直连不通）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# 1. 挖数据（未认证 60 次/小时，一次跑 2~4 个仓库）
python -m src.mine_github --repos-file config/repos.txt --pages 2 --out data/raw.jsonl
#    有 token 的话先 set GITHUB_TOKEN=ghp_xxx，配额变 5000 次/小时，可以全跑

# 2. 构造数据集
python -m src.build_dataset --raw data/raw.jsonl --out data/ds --balance

# 3. 训练（先 smoke 验证管线）
python -m src.train --ds data/ds --out out/run1 --smoke --freeze-encoder
python -m src.train --ds data/ds --out out/run1 --freeze-encoder --epochs 4

# 4. 评测，并且和没微调的底座对拍
python -m src.evaluate --ckpt out/run1/final --ds data/ds \
    --baseline convaiinnovations/laya

# 5. 拿真实 diff 试
git diff > pr.diff
python -m src.predict --ckpt out/run1/final --diff-file pr.diff --title "fix: ..."
```

## 三个问题和它们的标签怎么来的

**这一节请务必读完，它决定了你能不能用这些数字做决策。**

| 问题 | 类型 | 标签来源 | 可信度 |
|---|---|---|---|
| `will_merge` | `noul` | **GitHub 真实结果**：PR 被 merge 了 / 被关掉了 | 硬标签，可信 |
| `review_route` | `choice` | **`src/task.py` 里 `route_policy()` 这个确定性规则函数** | ⚠️ 是策略蒸馏，不是人的判断 |
| `change_scale` | `score` | **GitHub 真实增删行数**按固定阈值分箱 | 客观测量 |

### 关于 `review_route` 的坦白

第一个版本的模型学的是"把一条人为写的规则语义化"，不是学人的 review 判断。
规则在 `src/task.py`：

```python
if critical or churn >= 400 or n_files >= 10:      return "block"
if src and not tests and churn >= 20:              return "needs_tests"
if churn <= 30 and n_files <= 3 and (tests or not src): return "fast_track"
return "normal"
```

这样做有两个实际好处：标签免费且 100% 可复现；模型在学会规则之后，会把它**泛化到语义层面**
（比如"这个改动看起来危险"），而这正是后面换成人工标注时的起点。

**要换成人的判断**，只需要改 `src/task.py` 里的 `LabelPolicy.review_route()`，
其余管线（挖掘 / 构造 / 训练 / 评测）一行都不用动。

### `will_merge` 有个先验陷阱

真实数据里大部分 PR 是**被关闭**的（`psf/requests` 上约 60~70%）。
所以 `--balance` 会对训练集做下采样平衡；**测试集保留真实分布**，同时报告基线先验。
看准确率之前先看 `base_rate` —— 如果模型只是学会"大部分 PR 会被关"，
它的准确率会贴近基线却毫无判断力。

## 数据是怎么挖的（省配额的关键）

| 数据 | 来源 | 成本 |
|---|---|---|
| PR 元数据（merged_at、作者、文件等） | `api.github.com/repos/*/pulls` 列表接口 | 1 次请求 = 100 个 PR |
| **diff** | `github.com/<repo>/pull/<n>.diff` 网页端点 | **0 次配额** |
| 增删行数 | 从 diff 自己数 | 0 次配额 |

两个踩过的坑：

1. **`/pulls` 列表接口不返回 `additions`/`deletions`**（只有单 PR 接口才有）。
   一开始全挖成 0，导致 `change_scale` 全落在一个箱里。现在改成自己从 diff 数。
2. **`sort=updated` 排出来全是 Dependabot 的版本升级 PR**，语义单薄、会污染训练集。
   现在默认过滤 `[bot]` / `dependabot` / `renovate` 等账号。

## 训练的显存调优

对齐官方 notebook 的 RLCD 配方（策略梯度 + 软交叉熵，`proper_reward` 用严格恰当评分规则），
为单卡 8GB 改了三处：

- 去掉 DDP；micro batch 4（官方 8），靠梯度累积维持有效 batch 32
- 打开 encoder 梯度检查点 + head 检查点
- 新增 `--freeze-encoder`：只训决策头，forward 时 `detach_encoder=True` 切断梯度

**OOM 不会让整个 run 挂掉** —— 捕获 `torch.cuda.OutOfMemoryError` 后会当场把 micro batch 减半重试。

温度校准按官方做法：在**训练前就留出的 calib 切片**上，按问题类型各拟合一个温度，
用 LBFGS 在 log 温度上优化，钳制到 `[0.1, 10]`，写进 `rl_agent_config.json` 的 `temperature`，
并**主动删掉继承来的 `temperature_by_options`** —— 官方文档点名了这个坑：不删的话
它会静默覆盖新拟合的值，等于白校准。

## 评测指标

`src/evaluate.py` 走 `laya.load()` + `agent.predict()` 的**真实推理路径**（不是复用训练的前向），
顺带验证 checkpoint 能被官方加载器加载、温度真的生效。

- `will_merge`：accuracy / Brier / ECE / 基线先验
- `review_route`：accuracy / macro-F1 / 逐类 F1 / ECE / 多数类基线
- `change_scale`：MAE / argmax 准确率 / 相邻一级内比例 / ECE
- 每条 case 的中位延迟

加 `--baseline convaiinnovations/laya` 会同时跑一遍没微调的底座 —— **这才是你真正要看的对照**。

## 实测结果（2026-09-29，192 条真实 PR）

配置：冻结 encoder 只训 26.5M 参数的决策头，40 epoch，lr_head=3e-4，**峰值显存 2.47GB**，约 8 分钟。
测试集是整个留出的 `encode/httpx` 仓库（29 条，与训练仓无重叠）。

| 指标 | 训练集 (64) | 测试集 (29) | 测试集基线 |
|---|---|---|---|
| `will_merge` accuracy | 1.0000 | 0.7586 | 全猜 closed = 0.7241 |
| `review_route` accuracy | 0.9531 | 0.3103 | 多数类 = 0.3103 |
| `review_route` macro-F1 | 0.9488 | 0.2842 | |
| `change_scale` MAE | 0.4683 | 0.7482 | |
| 单条延迟 p50 | 149.7 ms | 152.7 ms | |

对比未微调底座：`will_merge` 0.2069、`review_route` 0.2069。所以微调**确实有用**，但还不够。

**结论：管线正确，数据量不够。** 训练集能到 95~100%，证明训练/校准/保存/加载/推理整条链路都对；
测试集塌缩到多数类基线，是典型的过拟合。官方微调用 1200 case / 6000 决策项，这里是 64 case / 192 items。

另一个教学点：`lr_head` 用默认的 1e-4 时，训练 loss 停在 1.09（≈ 只学会了标签的边际分布），
`will_merge` 的 P(merge) 全部落在 0.488~0.507，模型完全没做决定。**提到 3e-4 才开始真正学习。**

下一步该做什么，见 `HANDOFF.md`。

## 目录结构

```
laya-code-head/
├── config/repos.txt        默认挖取的仓库清单
├── src/
│   ├── task.py             ★ 任务定义 + 标签规则（改这里就能换任务）
│   ├── mine_github.py      挖真实 PR → data/raw.jsonl
│   ├── build_dataset.py    raw.jsonl → Laya 训练 items
│   ├── train.py            单卡训练 + 温度校准
│   ├── evaluate.py         指标 + 与底座对拍
│   └── predict.py          对一段真实 diff 做判断
├── data/                   raw.jsonl / ds/（生成物）
└── out/                    checkpoint 与评测报告（生成物）
```

## 已知局限

- **`review_route` 是策略蒸馏**，不是人的判断。它的天花板就是那条规则。
- **`will_merge` 有分布偏移**：训练仓和测试仓不同时，效果会掉。建议用 `--test-repos`
  留出整个仓库做跨仓测试，比随机切分更诚实。
- **state 会被截断**。`max_len=1024`、`head_max_len=256`，留给 diff 的大概只有 700 token
  （约 2800 字符）。大改动会被砍掉后半段，长 diff 上的判断不可靠。
- **`noul` 的标签敏感问题**：Laya 官方文档记录过 `false`/`true` 这对标签本身有时会压过状态。
  如果发现 `will_merge` 的答案不随输入变化，把 `src/task.py` 里的 `NOUL_LABELS`
  改成 `{"false": "B", "true": "A"}` 重训一次。
- 社区那批镜像里 `mys/*-GGUF` 是 `ggmlc` 产物，**llama.cpp 加载不了**，别搞混。

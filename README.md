# Laya 代码审查决策模型

把开源的 [Laya](https://github.com/NandhaKishorM/laya) 决策模型（421M）微调成一个**代码改动风险评级器**：给它一段 diff，它在一次前向传播里直接吐出三个结构化判断，不生成任何文本。

```
diff ──►  [审核路由]  fast_track / normal / needs_tests / block
          [能否合并]  P(merge) = 0.72
          [改动规模]  2.34 → moderate
```

> ## ⚠️ 现状：这个模型**还不能用**（2026-10-10）
>
> 三个任务在 434 条「整仓留出」的真实 PR 测试集上，**没有一个赢过"不动脑子"的参照物**：
>
> | 任务 | 本模型 | 参照物 |
> |---|---|---|
> | 会不会被合并 `will_merge` | AUC **0.5766**（0.5 阈值准确率 0.4977） | 一律猜多数类 **0.6060**；只用"仓库历史合并率"的查表基线 **AUC 0.8376** |
> | 审核路由 `review_route` | macro-F1 **0.1852**，正确类排第一 **20.0%** | 瞎猜 25%；一律答 `normal` 准确率 49.5% |
> | 改动规模 `change_scale` | MAE **0.8185** | 一律答 `moderate` 0.86；直接数 diff 行数 = 0 |
>
> **所以这个仓库的价值不在模型，而在另外两件事**：一套可直接复用的工程管线（真实 PR 挖掘 → 8GB 单卡训练 → 能防止自欺的评测），以及 `docs/` 里那份把**四个错误假设逐个证伪**的完整复盘。如果你要的是"一个能工作的 PR 风险判断器"，请先读 [§ 为什么说不能用](#为什么说不能用) 和 [§ 我们试过但没走通的路](#我们试过但没走通的路)。

---

## 目录

- [三个判断](#三个判断)
- [为什么说不能用](#为什么说不能用)
- [这个仓库能拿来干什么](#这个仓库能拿来干什么)
- [快速开始](#快速开始)
- [仓库结构](#仓库结构)
- [数据是怎么来的](#数据是怎么来的)
- [评测纪律：这个项目最值钱的部分](#评测纪律这个项目最值钱的部分)
- [我们试过但没走通的路](#我们试过但没走通的路)
- [已知局限](#已知局限)
- [版本与许可](#版本与许可)

---

## 三个判断

三个任务刻意覆盖了 Laya 的三种决策原语（`noul` 二选一 / `choice` 多选 / `score` 打分），任务定义集中在 [`src/task.py`](src/task.py) 一个文件——换任务只改这一个文件。

| 问题 | 原语 | 标签来源 | 可信度 |
|---|---|---|---|
| `will_merge` 会不会被合并 | `noul` | **GitHub 真实结果**：PR 被 merge 了还是被关掉了 | ✅ 硬标签 |
| `review_route` 审核路由 | `choice` | **`src/task.py` 里 `route_policy()` 这个确定性规则函数** | ⚠️ 是策略蒸馏，不是人的判断 |
| `change_scale` 改动规模 | `score` | GitHub 真实增删行数按固定阈值分箱 | ✅ 客观测量 |

`review_route` 的规则长这样（这一版模型学的是"把这条规则语义化"，**不是**学人的 review 判断）：

```python
if critical or churn >= 400 or n_files >= 10:            return "block"
if src and not tests and churn >= 20:                    return "needs_tests"
if churn <= 30 and n_files <= 3 and (tests or not src):  return "fast_track"
return "normal"
```

**这里有一个很重要的推论**：`review_route` 和 `change_scale` 的标签都是这条规则的确定性输出，而规则的输入（文件列表、增删行数）全都印在文本里。也就是说**这两个任务用规则直接算就是 100% 正确、零延迟、完全可解释**——模型在这两格上唯一可能的价值是"泛化到规则没覆盖的语义"（比如"这个改动看起来危险"），而它现在连规则本身都没学会。真正非模型不可的只有 `will_merge`，因为它依赖 GitHub 的真实结果，规则算不出来。

---

## 为什么说不能用

### 1) 三个任务都没赢过对照物

测试集固定为 `encode/httpx` + `python-poetry/poetry` 两个**整仓留出**的真实 PR，共 434 条，训练时从未见过这两个仓库：

| 模型 | `will_merge` AUC | `review_route` macro-F1 | `change_scale` MAE |
|---|---|---|---|
| 原始底座（对齐输入预算） | 0.4488 | 0.1670 | 0.8936 |
| run3（10 轮，state 里带仓库名） | 0.5245 | **0.2634** | **0.6952** |
| **run5（去掉仓库名 + 路由平衡 + dev 早停）** | **0.5766** | 0.1852 | 0.8185 |
| 参照：多数类 / 常数 | 准确率 0.6060 | 准确率 0.4954 | 0.86 |
| 参照：只用「仓库历史合并率」查表 | **0.8376**（同池 dev） | — | — |

底座 AUC 0.4488 **低于随机**，微调确实把它拉到了随机之上（0.5766），但**仍然低于"一律猜多数类"**，更低于一个五分钟能写出来的查表基线。放大到错误的训练目标上不解决问题。

### 2) 模型根本没在排序，不是"阈值没选好"

V3 实验（[`tools/v_prior.py`](tools/v_prior.py)）做了先验还原，把训练时的类别先验搬回测试集的真实先验：

- macro-F1 从 0.1852 → **0.1656**（不升反降）
- **正确类排第一的比例只有 20.0%**（瞎猜是 25%）
- 正确类的平均概率 **0.2501**，四选一的均匀分布理论值就是 0.25

概率分布几乎是平的。这排除了"只是判定规则被先验压住"的可能。

### 3) 真正动过的杠杆只有一个：训练量

| 配置 | 训练集 `review_route` macro-F1 |
|---|---|
| 混合数据 600 项 × 3 轮 | 0.1820 |
| route 单任务 600 项 × 3 轮 | 0.1870 |
| **route 单任务全量 5296 项 × 3 轮** | **0.3966** |

数据量差 8.8 倍，指标差 2 倍。但这条路也不好走：**同一配置两次运行的训练集准确率可以差 0.10**（0.3760 vs 0.2803，数据、学习率、seed 全同），所以任何小于 0.10 的差异都不能当结论——而评测口径的噪声下限只有 0.004，两者是完全不同的量。详见 [`docs/超参对照汇总.md`](docs/超参对照汇总.md)。

---

## 这个仓库能拿来干什么

**现在就能用：**

1. **当管线模板复用**。任何"输入一段文本、输出一个结构化判断"的任务，改 `src/task.py` 换任务即可，挖掘 / 构建 / 训练 / 评测 / 预测五个环节不用动。Laya 的三种原语在这里都跑通了。
2. **用它的规则函数做真实的审核路由与规模判定**。`src/task.py` 的 `route_policy()` / `churn_bin()` 零延迟、可解释、与训练标签 100% 一致。模型的延迟是 ~150ms 且更不准。
3. **拿数据和划分方式做别的事**。9951 条真实 PR（49 个仓库）的挖掘管线、按整仓留出的切分方式、434 条固定测试集（`data/ds5/test_cases.jsonl`，可直接用于评测）。
4. **读复盘**。`docs/` 里记录了四个被实验证伪的假设、两个评测口径的坑、以及一次"以为自己成功了"的完整过程。负面结果比正面结果稀缺。

**不能用于：**

- ❌ 跨仓库通用风险评分（AUC 0.58 ≈ 抛硬币）
- ❌ CI 门禁：自动拦 PR、自动指派路由
- ❌ 任何依赖 `confidence` 的自动决策（概率几乎均匀）
- ❌ 对外汇报"最优阈值下的准确率"——评测脚本里那个数字是在被评测的那份数据上挑的阈值，偏乐观（见下）

---

## 快速开始

### 1) 环境

```bash
python -m venv <VENV>
<VENV>/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128  # 4060 Ti(sm_89)/cu128；CPU 机器换 /cpu
<VENV>/Scripts/python.exe -m pip install -r requirements.txt
```

实测开发机：**RTX 4060 Ti / 8GB**。冻结 encoder 只训决策头时**峰值显存 2.47GB**，很宽裕；全量微调会 OOM。

### 2) 不配 GitHub token 也能跑通全流程

仓库里带了一份 100 条真实 PR 的样本（`data/sample/raw_sample_100.jsonl`）：

```bash
python -m src.build_dataset --raw data/sample/raw_sample_100.jsonl --out data/demo \
    --model-id convaiinnovations/laya --balance
python -m src.train --ds data/demo --out out/demo --model-id models/laya --smoke --freeze-encoder
```

### 3) 正式跑一遍（需要自己准备底座权重）

**本仓库不包含模型权重**（803MB 的底座、每个 800MB+ 的 checkpoint 都没入库）。
底座可以用 `huggingface-cli download`，或者：

```bash
git lfs install && git clone https://huggingface.co/convaiinnovations/laya models/laya
```

然后：

```bash
# 构建数据集（按仓库留出测试集）
python -m src.build_dataset --raw data/raw_snapshot.jsonl --out data/ds5 \
    --model-id models/laya --test-repos encode/httpx,python-poetry/poetry --balance

# 训练（冻结 encoder，10 轮约 76 分钟）
python -m src.train --ds data/ds5 --out out/run3 --model-id models/laya \
    --freeze-encoder --epochs 10 --micro-batch 4 --grad-accum 8 --lr-head 3e-4

# 评测（走 laya.load 的真实推理路径，并和底座对拍）
python -m src.evaluate --ckpt out/run3/final --ds data/ds5 --split test --baseline models/laya

# 对一段真实 diff 打分
git diff > pr.diff
python -m src.predict --ckpt out/run3/final --diff-file pr.diff --title "fix: ..."

# 批量打分并按风险排序（真正能干活的形式：只加载一次模型）
python -m src.batch_predict --ckpt out/run3/final --cases data/ds5/test_cases.jsonl \
    --compare-rule --sort-by risk --top 20
```

> 想要完整复现实验报告：`out/` 里原来的评测 JSON 已精选到 [`reports/`](reports/)（40 份，276KB），可以直接对照 `docs/` 里的结论。

---

## 仓库结构

```
.
├── src/
│   ├── task.py            ★ 任务定义 + 标签规则（换任务只改这里）
│   ├── mine_github.py     挖真实 PR → raw.jsonl
│   ├── build_dataset.py   raw.jsonl → Laya 的 token 化 items
│   ├── train.py           单卡 8GB 训练 + 温度校准 + dev 早停 + 续训
│   ├── evaluate.py        指标 + 与底座对拍
│   ├── predict.py         对一段真实 diff 出三个判断
│   └── batch_predict.py   加载一次、评一批、可排序、可与规则对拍
├── tools/                 实验与验证脚本（V0~V6、隔离实验、诊断）
├── devlab/                "换到我自己的代码上准不准"的对拍夹具
│   ├── miniurl/           一个迷你 URL 短链项目（基线）
│   ├── diffs/             10 个覆盖规则每条边界的场景 diff + index.json
│   └── logs/              对拍结果（各模型在 10 个场景上的判定）
├── data/
│   ├── sample/            100 条真实 PR 样本（不用 token 就能跑）
│   └── ds5/               434 条固定测试集 + 数据集口径 meta.json
├── reports/               精选评测证据（每个结论对应的原始 JSON）
├── docs/                  ★ 任务文档 / 训练计划 / 排查结论 / 超参对照 / 交接文档
├── config/repos.txt       挖掘用的仓库清单（62 个，按吞吐性价比排序）
└── requirements.txt
```

---

## 数据是怎么来的

| 数据 | 来源 | 成本 |
|---|---|---|
| PR 元数据（merged_at、作者、文件数…） | `api.github.com/repos/*/pulls` 列表接口 | 1 次请求 = 100 条 PR |
| **diff** | `github.com/<repo>/pull/<n>.diff` 网页端点 | **0 次配额** |
| 增删行数 | 从 diff 自己数 | 0 次配额 |

两个关键优化（把吞吐从 **5 条/分钟提到 ~160 条/分钟**）：

1. **"元数据贵、diff 免费"**——少请求元数据、多并发抓 diff；单条 diff 下载限 200KB，整页大面积失败时退避重试。
2. **hybrid 取 diff**：先走免费的网页端点，失败才回落到 API（有 token 时配额只花在失败的那部分）。

三个踩过的坑：

1. `/pulls` **列表**接口不返回 `additions`/`deletions`（只有单 PR 接口才有，成本 100 倍）→ 改成从 diff 自己数行。
2. `sort=updated` 排出来全是 Dependabot 的版本升级 PR，语义单薄、污染训练集 → 默认过滤 `[bot]` / `dependabot` / `renovate`。
3. 同一批里还有大量"开出来几分钟就被关掉"的低质量投稿，它们的 `merged=0` 标签没错，但反映的是"没人认真看过" → `--min-open-hours 1` 剔掉。

---

## 评测纪律：这个项目最值钱的部分

训一个小模型不难，**难的是不把自己的结论骗过去**。这个仓库里有六条现成的检查，建议任何做决策头微调的人都抄一遍：

| 检查 | 问题 | 本项目的实测结果 |
|---|---|---|
| **V0 测试集哈希** | 跨轮对比的是不是同一份测试集？ | `ds3` 与 `ds5` 的 `test_cases.jsonl` SHA256 相同 → 之前对比有效 |
| **V3 先验还原** | "不会分类"还是"判定规则被先验压住"？ | 还原后反而更低（0.1852→0.1656）→ 模型确实没在排序 |
| **V4 按仓库拆分** | 合并指标是不是被大仓平均出来的？ | 两仓 AUC 一致（0.584/0.585），但 **`change_scale` MAE 差 0.157** → 必须分仓报 |
| **V5 目标函数体检** | 损失函数真的在优化我们要的东西吗？ | 普通交叉熵权重 1.0→5.0，指标几乎不动 → 不是权重问题 |
| **V6 噪声下限** | 多大的提升才算"有效"？ | 同配置换 seed，**评测**口径极差仅 0.004；但**训练**的运行间方差高达 0.10 |
| **配对显著性** | 两个模型的差距是真的吗？ | run3→run5 的 +0.052 AUC，自助法一侧 p≈0.025，95% 置信区间下界贴着 0 → "勉强显著" |

外加两条容易忽略的**评测口径 bug**（本仓库保留原样，未修）：

- `src/evaluate.py` 的 `best_threshold_accuracy` 是在**被评测的那个划分上**挑最优阈值——用 `--split test` 跑就等于拿测试集调参。**这个数字不要引用。**
- 训练集做了 50/50 平衡而测试集保留真实分布（60.6% 合并率），所以 0.5 阈值天然偏，**看 `will_merge` 请只看 AUC**。

---

## 我们试过但没走通的路

省下你重走的四小时（详细证据见 [`docs/排查结论_v1.md`](docs/排查结论_v1.md) 与 [`docs/超参对照汇总.md`](docs/超参对照汇总.md)）：

| 假设 | 怎么验的 | 结果 |
|---|---|---|
| **仓库身份泄漏**是过拟合病根 | 从 state 里删掉 `repo: owner/name` 重训 | ✅ **真杠杆**：训练集-测试集缺口 0.2907 → 0.0096 |
| 优势归一化把三个任务混在一起算带偏了 | 全平铺 vs 按任务分组归一化 | ❌ 符号一致率 0.9833、相关 0.9812 |
| 任务混合训练互相干扰 | 严格等数据量对照（600 项） | ❌ 混合 0.1820 vs 单任务 0.1870 |
| 梯度来源冲突（该只反传 route） | `--loss-qid` 只反传 route | ❌ 权重指纹差 0.002%、预测 100% 相同 |
| 目标函数权重不对 | `--ce-weight` 1.0 → 5.0 | ❌ 训练集指标几乎不动 |
| 微调把底座的信号压平了 | 对比 logits 极差 | ⚠️ 现象真实，但换全量训练后 logits 恢复 → 是"训得不够"，不是结构损坏 |
| **数据 / 轮数**是杠杆 | 192→3705 条、5→10 轮 | ⚠️ 只在单任务全量 route 上成立；跨仓库指标不动 |

**最后一条最重要**：`0.2907 → 0.0096` 这个缺口消失，一开始被解读成"过拟合治好了"，后来发现是**训练集那头塌了**（训练集 AUC 0.8152 → 0.5862）——模型不再背仓库名了，但也没真正学会判断。**同一份数据可以支持两个相反的解读，这就是为什么要固定口径、先跑噪声下限再下结论。**

---

## 已知局限

1. **`review_route` 是策略蒸馏**，天花板就是 `task.py` 里那条规则，不是人的判断。要换成真实标签需要抓 GitHub 的 review 事件（`--with-reviews` 已有，但语料里目前**没有** `merged_at`、评论数、review 时间这些字段，得重挖）。
2. **`state` 会被截断**：`max_len=1024` / `head_max_len=256`，实测 **64.1%** 的样本触到上限，而截掉的是排在最后、也是最能说明问题的 `diff`（元数据保留率 99.9%，不受影响）。
3. **推理配置必须和训练一致**：数据集与训练用 `max_len=1024`，而底座自带 512。对拍底座时要用对齐预算的副本，否则对底座不公平（两种预算下 AUC 差 0.03~0.04，而 434 条样本的 AUC 标准误差约 0.038——**这个差距在噪声范围内**）。
4. **`noul` 的标签敏感**：官方文档记录过 `false`/`true` 这对标签本身有时会压过状态。若发现答案不随输入变化，把 `src/task.py` 的 `NOUL_LABELS` 改成 `{"false": "B", "true": "A"}` 重训。
5. **没有 LoRA 支持**：8GB 上全量微调会爆，目前只训决策头（26.5M 参数）。`--freeze-encoder` 用 `detach_encoder=True` 切断梯度。
6. **单卡训练的 OOM 保护**：捕获 `torch.cuda.OutOfMemoryError` 后当场把 micro batch 减半重试，不会让整个 run 挂掉。
7. **温度校准的坑已修**：laya 0.3.21 的加载器只接受温度 `[0.5, 5]`，超出会被静默钳回边界；保存时必须 `pop` 掉 `temperature_by_options`，否则它会静默覆盖新拟合的温度。

---

## 版本与许可

- 版本：见 [`VERSION`](VERSION)，更新记录见 [`CHANGELOG.md`](CHANGELOG.md)。0.x 表示尚未稳定。
- 本仓库代码：MIT（见 [`LICENSE`](LICENSE)）。
- 底座模型 [Laya](https://github.com/NandhaKishorM/laya) 与训练所用的真实 PR 数据，版权与许可归各自来源；本仓库只包含代码、文档、少量数据样本与评测报告，**不包含任何模型权重**。

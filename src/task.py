"""任务定义：代码改动风险评级（code-review decisions）

三个问题，覆盖 Laya 的全部三种原语。标签来源分两类，都在这里写明：

  will_merge    (noul)   —— 硬标签。直接来自 GitHub 的真实结果：PR 被 merge 了还是被关掉了。
  review_route  (choice) —— 由下面的 route_policy() 这个**公开、确定性的策略函数**派生。
                            也就是说第一版模型学的是"把这个策略语义化"，不是学人的判断。
                            等你攒够人工标注，把 label_route 换成人的标注即可，其余管线不用改。
  change_scale  (score)  —— 客观测量。来自 GitHub 真实的 additions + deletions，按固定阈值分箱。

改任务只需要改这个文件，mine / build / train / eval 都不用动。
"""

# ---------------------------------------------------------------------------
# 1. 问题定义（直接喂给 Laya 的 questions 结构）
# ---------------------------------------------------------------------------

# noul 的两个槽位模型看到什么文本。留着 None 就是默认的 `false:` / `true:`。
# 主仓库 BENCHMARKS.md 记录过：英文 checkpoint 上 `false`/`true` 这对标签本身有时会压过状态。
# 如果发现 will_merge 的答案不随输入变化，把这里改成 {"false": "B", "true": "A"} 再来一遍。
# 注意：无论这里写什么，返回值永远是 P(true)。
NOUL_LABELS = None

QUESTIONS = {
    "will_merge": {
        "type": "noul",
        "instructions": "Will this change be merged as it stands?",
        "criteria": {
            "true": "the change looks good enough to be merged",
            "false": "the change will be closed without being merged",
        },
    },
    "review_route": {
        "type": "choice",
        "instructions": "How should a reviewer route this change?",
        "criteria": {
            "fast_track": "small, low blast radius, already covered by tests: a skim is enough",
            "normal": "an ordinary change that needs a normal read-through",
            "needs_tests": "it changes behavior but ships no test for that behavior",
            "block": "large or high blast radius: needs a deep review before it can merge",
        },
    },
    "change_scale": {
        "type": "score",
        "instructions": "How large is this change?",
        "criteria": ["tiny", "small", "moderate", "large", "huge"],
    },
}
if NOUL_LABELS:
    QUESTIONS["will_merge"]["labels"] = NOUL_LABELS


# ---------------------------------------------------------------------------
# 2. 标签派生规则
# ---------------------------------------------------------------------------

CODE_EXT = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs",
    ".java", ".kt", ".kts", ".scala", ".rb", ".php", ".cs", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".swift", ".m", ".mm", ".lua", ".sh", ".bash", ".sql",
}

# 碰到这些路径就认为"爆炸半径大"，优先级最高
CRITICAL_PATTERNS = (
    "migration", "migrations/", "/schema", "dockerfile", "docker-compose",
    ".github/workflows", "pyproject.toml", "setup.py", "package.json",
    "requirements", "poetry.lock", "package-lock", "yarn.lock", "cargo.toml",
    "go.mod", "pom.xml", "build.gradle", "/auth", "/security", "/crypto",
    "/billing", "/payment", "terraform", "kubernetes", "helm/",
)

TEST_PATTERNS = ("test_", "_test.", "/tests/", "/test/", "spec.", ".spec.",
                 "conftest.py", "tests/", "__tests__")


def is_test_file(path: str) -> bool:
    p = path.lower()
    return any(pat in p for pat in TEST_PATTERNS)


def is_code_file(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(ext) for ext in CODE_EXT)


def is_critical_file(path: str) -> bool:
    p = path.lower()
    return any(pat in p for pat in CRITICAL_PATTERNS)


def churn_bin(additions: int, deletions: int) -> int:
    """change_scale：按总改动行数分箱。固定阈值，可复现。"""
    c = additions + deletions
    if c <= 5:
        return 0        # tiny
    if c <= 30:
        return 1        # small
    if c <= 120:
        return 2        # moderate
    if c <= 500:
        return 3        # large
    return 4            # huge


def route_policy(files, additions: int, deletions: int) -> str:
    """review_route 的确定性策略（第一版的 teacher）。

    顺序即优先级。这是一个人为写的规则，不是人的判断 —— 它的价值是让第一版模型
    先学会"按语义泛化这套规则"，之后再用人工标注替换掉。
    """
    churn = additions + deletions
    n_files = len(files)
    tests = any(is_test_file(f) for f in files)
    src = any(is_code_file(f) for f in files)
    critical = any(is_critical_file(f) for f in files)

    if critical or churn >= 400 or n_files >= 10:
        return "block"
    if src and not tests and churn >= 20:
        return "needs_tests"
    if churn <= 30 and n_files <= 3 and (tests or not src):
        return "fast_track"
    return "normal"


class LabelPolicy:
    """把一条挖到的 PR 记录变成一个训练样本的标签。

    raw 里需要的字段见 mine_github.py 写出的 JSONL schema。
    """

    def will_merge(self, raw) -> int:
        # 硬标签：GitHub 的真实结果
        return 1 if raw["merged"] else 0

    def review_route(self, raw) -> str:
        return route_policy(raw["files"], raw["additions"], raw["deletions"])

    def change_scale(self, raw) -> int:
        return churn_bin(raw["additions"], raw["deletions"])


# ---------------------------------------------------------------------------
# 3. 每个问题的目标分布（硬标签 → one-hot）
# ---------------------------------------------------------------------------

QIDS = list(QUESTIONS.keys())


def hard_target(qid: str, raw) -> list:
    """返回该问题的 one-hot 目标分布，顺序必须和 render_options 的输出一致。"""
    pol = LabelPolicy()
    q = QUESTIONS[qid]
    if qid == "will_merge":
        # render_options 对 noul 恒为 [false, true]
        v = pol.will_merge(raw)
        return [1.0 - v, float(v)]
    if qid == "review_route":
        keys = list(q["criteria"].keys())
        v = pol.review_route(raw)
        return [1.0 if k == v else 0.0 for k in keys]
    if qid == "change_scale":
        n = len(q["criteria"])
        v = pol.change_scale(raw)
        return [1.0 if i == v else 0.0 for i in range(n)]
    raise KeyError(qid)


def gold_text(qid: str, raw) -> str:
    """人类可读的标签，只用于日志和人工核查。"""
    pol = LabelPolicy()
    if qid == "will_merge":
        return "merged" if pol.will_merge(raw) else "closed_unmerged"
    if qid == "review_route":
        return pol.review_route(raw)
    return str(pol.change_scale(raw))

"""生成一套"像真开发"的 diff 场景，用来检验训练好的模型在本地改动上到底能不能用。

为什么不用真仓库：真实 PR 已经有专门的留出测试集（`data/ds/test_cases.jsonl`），
但那是"别人的仓库"。这里要回答的是另一个问题 —— **换到我自己的代码上，它还准不准**。
所以造一个语法风格、目录结构、改动形态都正常的迷你项目，然后沿着
`src/task.py:route_policy()` 的每一条判定边界各造一个改动。

10 个场景覆盖了规则策略的每个分支：

    +---- 场景 --------------------- 规则会判成 ---- 边界在哪
    |  01 改注释错字                  fast_track     churn 最小、非代码也放行
    |  02 只改 README                 fast_track     not src and not tests
    |  03 只加测试                    fast_track     tests 存在
    |  04 新功能但不带测试            needs_tests    src and not tests and churn>=20
    |  05 修 bug 且带回归测试         normal         有测试 → 不落 needs_tests
    |  06 跨 10 个文件重命名          block          n_files >= 10
    |  07 改 CI / pyproject           block          critical patterns
    |  08 500+ 行大特性               block          churn >= 400
    |  09 小改动但没测试              normal         churn<20 → 掉出 needs_tests 分支
    |  10 依赖版本升级                block          critical: requirements

用法：
    python devlab/make_diffs.py                 # 生成 devlab/diffs/*.diff + index.json
"""

from __future__ import annotations

import difflib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "miniurl"          # 基线项目
DIFFS = ROOT / "diffs"
sys.path.insert(0, str(ROOT.parent))
from src.task import route_policy, churn_bin  # noqa: E402


# --------------------------------------------------------------------------
# 基线迷你项目：一个 URL 短链服务（有 src、有 tests、有 pyproject、有 CI）
# --------------------------------------------------------------------------

def base_files() -> dict[str, str]:
    return {
        "pyproject.toml": """[project]
name = "miniurl"
version = "0.3.1"
requires-python = ">=3.10"
dependencies = ["httpx>=0.27"]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.5"]
""",
        "README.md": """# miniurl

A tiny URL shortener used as a playground for review-risk scoring.

## Usage

```python
from miniurl import shorten
shorten("https://example.com/a/very/long/path")
```
""",
        ".github/workflows/ci.yml": """name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: pytest -q
""",
        "src/miniurl/__init__.py": """from .shorten import shorten, expand
from .store import MemoryStore

__all__ = ["shorten", "expand", "MemoryStore"]
""",
        "src/miniurl/shorten.py": '''"""base62 短码生成与还原。"""

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE = len(ALPHABET)


def encode(num: int) -> str:
    """把自增 id 变成短码。"""
    if num < 0:
        raise ValueError("id must be non-negative")
    if num == 0:
        return ALPHABET[0]
    out = []
    while num > 0:
        num, rem = divmod(num, BASE)
        out.append(ALPHABET[rem])
    return "".join(reversed(out))


def decode(code: str) -> int:
    """短码还原成 id。"""
    num = 0
    for ch in code:
        num = num * BASE + ALPHABET.index(ch)
    return num


def shorten(url: str, store) -> str:
    """存下长链接，返回短码。"""
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http(s) urls are supported")
    item_id = store.add(url)
    return encode(item_id)


def expand(code: str, store) -> str:
    """短码还原成长链接。"""
    return store.get(decode(code))
''',
        "src/miniurl/store.py": '''"""存储层：内存实现 + 一个抽象基类。

刻意做成两层：线上换成 Redis/SQLite 时只替换 Store 的实现，业务代码不动。
"""


class Store:
    def add(self, value: str) -> int:
        raise NotImplementedError

    def get(self, item_id: int) -> str:
        raise NotImplementedError


class MemoryStore(Store):
    def __init__(self):
        self._rows: list[str] = []
        self._index: dict[str, int] = {}

    def add(self, value: str) -> int:
        if value in self._index:
            return self._index[value]
        self._rows.append(value)
        item_id = len(self._rows) - 1
        self._index[value] = item_id
        return item_id

    def get(self, item_id: int) -> str:
        if item_id < 0 or item_id >= len(self._rows):
            raise KeyError(item_id)
        return self._rows[item_id]

    def __len__(self) -> int:
        return len(self._rows)
''',
        "src/miniurl/api.py": '''"""极简 HTTP 层（只为演示，不追求生产可用）。"""

import json
from http.server import BaseHTTPRequestHandler

from .shorten import expand, shorten


class Handler(BaseHTTPRequestHandler):
    store = None  # 由 create_server 注入

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        code = shorten(payload["url"], self.store)
        self._send(200, {"code": code})

    def do_GET(self):
        code = self.path.strip("/")
        try:
            url = expand(code, self.store)
        except KeyError:
            self._send(404, {"error": "unknown code"})
            return
        self._send(302, {"url": url})

    def _send(self, status: int, body: dict):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
''',
        "tests/test_shorten.py": '''import pytest

from miniurl.shorten import decode, encode
from miniurl.store import MemoryStore


def test_encode_zero():
    assert encode(0) == "0"


def test_encode_decode_roundtrip():
    for n in (1, 61, 62, 3843, 123456):
        assert decode(encode(n)) == n


def test_encode_rejects_negative():
    with pytest.raises(ValueError):
        encode(-1)


def test_store_dedupes():
    store = MemoryStore()
    assert store.add("https://a.example") == store.add("https://a.example")
    assert len(store) == 1
''',
        "tests/test_store.py": '''import pytest

from miniurl.store import MemoryStore


def test_get_out_of_range():
    store = MemoryStore()
    store.add("https://a.example")
    with pytest.raises(KeyError):
        store.get(7)


def test_get_returns_inserted_value():
    store = MemoryStore()
    item_id = store.add("https://b.example")
    assert store.get(item_id) == "https://b.example"
''',
    }


# --------------------------------------------------------------------------
# 场景：每个场景 = 对基线做一组操作
# --------------------------------------------------------------------------

def _pad_feature(n_lines: int) -> str:
    """造一段"看起来像真代码"的大改动，用来触发 churn >= 400。"""
    body = ['"""分析统计：把访问记录聚合成报表。"""', "", "from collections import Counter", "", ""]
    for i in range(n_lines):
        body += [
            f"def metric_{i}(rows):",
            f'    """第 {i} 个统计口径。"""',
            "    counter = Counter(",
            f'        r.get("field_{i}") for r in rows if r.get(f"enabled_{i}", True)',
            "    )",
            "    total = sum(counter.values()) or 1",
            "    return {k: v / total for k, v in counter.items()}",
            "",
        ]
    return "\n".join(body)


def scenarios() -> dict:
    base = base_files()
    s: dict[str, dict] = {}

    # 01 改注释错字：最小改动，非行为变化
    files = dict(base)
    files["src/miniurl/shorten.py"] = base["src/miniurl/shorten.py"].replace(
        "把自增 id 变成短码。", "把自增 id 变成短码（base62）。")
    s["01_typo_in_docstring"] = {"files": files, "note": "只改了一句 docstring，无行为变化"}

    # 02 只改 README
    files = dict(base)
    files["README.md"] = base["README.md"].replace(
        "A tiny URL shortener used as a playground for review-risk scoring.",
        "A tiny URL shortener. Used as a playground for automated review-risk scoring.")
    s["02_readme_only"] = {"files": files, "note": "文档改动，非代码文件"}

    # 03 只加测试（给已有行为补覆盖）
    files = dict(base)
    files["tests/test_shorten.py"] = base["tests/test_shorten.py"] + '''

def test_decode_rejects_unknown_char():
    with pytest.raises(ValueError):
        decode("!!")
'''
    s["03_tests_only"] = {"files": files, "note": "只加测试，没碰业务代码"}

    # 04 新功能但不带测试：src 改动 + 无测试，churn 超过 20
    files = dict(base)
    files["src/miniurl/ratelimit.py"] = '''"""按客户端限流（滑动窗口）。"""

import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int = 60, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        hits = self._hits[client]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True


def client_key(handler) -> str:
    return handler.client_address[0]
'''
    files["src/miniurl/api.py"] = base["src/miniurl/api.py"].replace(
        "from .shorten import expand, shorten",
        "from .ratelimit import RateLimiter, client_key\nfrom .shorten import expand, shorten",
    ).replace(
        "    store = None  # 由 create_server 注入",
        "    store = None  # 由 create_server 注入\n    limiter = RateLimiter()",
    ).replace(
        "    def do_POST(self):\n        length",
        "    def do_POST(self):\n"
        "        if not self.limiter.allow(client_key(self)):\n"
        '            self._send(429, {"error": "rate limited"})\n'
        "            return\n"
        "        length",
    )
    s["04_new_feature_no_tests"] = {"files": files, "note": "新增限流模块并接进 HTTP 层，一行测试都没加"}

    # 05 修 bug 且带回归测试：base62 解码对非法字符没报错
    files = dict(base)
    files["src/miniurl/shorten.py"] = base["src/miniurl/shorten.py"].replace(
        '''def decode(code: str) -> int:
    """短码还原成 id。"""
    num = 0
    for ch in code:
        num = num * BASE + ALPHABET.index(ch)
    return num''',
        '''def decode(code: str) -> int:
    """短码还原成 id。非法字符要报错，而不是抛一个看不懂的 ValueError。"""
    if not code:
        raise ValueError("empty code")
    num = 0
    for ch in code:
        if ch not in ALPHABET:
            raise ValueError(f"illegal character in code: {ch!r}")
        num = num * BASE + ALPHABET.index(ch)
    return num''')
    files["tests/test_shorten.py"] = base["tests/test_shorten.py"] + '''

def test_decode_rejects_illegal_character():
    with pytest.raises(ValueError, match="illegal character"):
        decode("0!")


def test_decode_rejects_empty():
    with pytest.raises(ValueError, match="empty code"):
        decode("")
'''
    s["05_bugfix_with_test"] = {"files": files, "note": "修 bug 且补了回归测试"}

    # 06 跨 10 个文件的重命名：触发 n_files >= 10
    files = dict(base)
    files["src/miniurl/store.py"] = base["src/miniurl/store.py"].replace("Store", "Repository")
    files["src/miniurl/__init__.py"] = base["src/miniurl/__init__.py"].replace("MemoryStore", "MemoryRepository")
    files["src/miniurl/api.py"] = base["src/miniurl/api.py"].replace(
        "from .shorten import expand, shorten",
        "from .store import Repository  # Store 已改名\nfrom .shorten import expand, shorten")
    for i in range(7):
        files[f"src/miniurl/plugins/p{i}.py"] = f'''"""插件 {i}：把 Store 换成 Repository 之后同步改名。"""

from ..store import Repository


class Plugin{i}(Repository):
    def add(self, value: str) -> int:
        return super().add(value.strip())
'''
    s["06_rename_across_10_files"] = {"files": files, "note": "跨 10 个文件重命名 Store → Repository"}

    # 07 改 CI 与构建配置：命中 critical patterns
    files = dict(base)
    files[".github/workflows/ci.yml"] = base[".github/workflows/ci.yml"].replace(
        "      - run: pytest -q",
        "      - run: pip install -e \".[dev]\"\n      - run: ruff check src tests\n      - run: pytest -q --cov=miniurl",
    )
    files["pyproject.toml"] = base["pyproject.toml"].replace(
        'version = "0.3.1"',
        'version = "0.4.0"',
    )
    s["07_ci_and_pyproject"] = {"files": files, "note": "改 CI 流程和 pyproject（爆炸半径大）"}

    # 08 500+ 行的大特性
    files = dict(base)
    files["src/miniurl/metrics.py"] = _pad_feature(55)
    files["src/miniurl/api.py"] = base["src/miniurl/api.py"].replace(
        "from .shorten import expand, shorten",
        "from .metrics import metric_0\nfrom .shorten import expand, shorten",
    )
    s["08_huge_feature"] = {"files": files, "note": "一次性加 500+ 行统计模块"}

    # 09 小改动但没测试：churn 小于 20，掉出 needs_tests 分支
    files = dict(base)
    files["src/miniurl/shorten.py"] = base["src/miniurl/shorten.py"].replace(
        '    if not url.startswith(("http://", "https://")):',
        '    if not url.startswith(("http://", "https://", "ftp://")):')
    s["09_small_src_no_test"] = {"files": files, "note": "放宽协议白名单，10 行以内，没加测试"}

    # 10 依赖版本升级
    files = dict(base)
    files["pyproject.toml"] = base["pyproject.toml"].replace('"httpx>=0.27"', '"httpx>=0.28"')
    files["requirements.txt"] = "httpx==0.28.1\npytest==8.3.4\n"
    s["10_dependency_bump"] = {"files": files, "note": "依赖版本升级（requirements 属于 critical 路径）"}
    return s


# --------------------------------------------------------------------------
# 生成 unified diff
# --------------------------------------------------------------------------

def unified_diff(path: str, old: str | None, new: str | None, context: int = 3) -> str:
    """把一对文件内容渲染成 git 风格 unified diff。

    自己渲染而不是调 git：一是免依赖，二是 `git diff --no-index` 对新增/删除文件的
    表现随版本变，自己写反而更可控。格式必须和 src/mine_github.py 的
    `^diff --git a/(.+?) b/(.+?)$` 对得上。
    """
    a_path = f"a/{path}" if old is not None else "/dev/null"
    b_path = f"b/{path}" if new is not None else "/dev/null"
    old_lines = (old or "").splitlines(keepends=True)
    new_lines = (new or "").splitlines(keepends=True)
    body = list(difflib.unified_diff(old_lines, new_lines, fromfile=a_path, tofile=b_path, n=context))
    if not body:
        return ""
    head = f"diff --git a/{path} b/{path}\n"
    if old is None:
        head += "new file mode 100644\n"
    elif new is None:
        head += "deleted file mode 100644\n"
    return head + "".join(body)


def main():
    if DIFFS.exists():
        shutil.rmtree(DIFFS)
    DIFFS.mkdir(parents=True)
    (ROOT / "miniurl").mkdir(exist_ok=True)
    base = base_files()

    # 把基线项目落盘，方便人肉翻看
    for path, content in base.items():
        p = BASE / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    index = []
    for name, sc in scenarios().items():
        files = sc["files"]
        chunks = []
        for path in sorted(set(base) | set(files)):
            d = unified_diff(path, base.get(path), files.get(path))
            if d:
                chunks.append(d)
        text = "".join(chunks)
        (DIFFS / f"{name}.diff").write_text(text, encoding="utf-8")

        paths = [p for p in sorted(set(base) | set(files)) if base.get(p) != files.get(p)]
        add = sum(1 for l in text.splitlines() if l.startswith("+") and not l.startswith("+++"))
        dele = sum(1 for l in text.splitlines() if l.startswith("-") and not l.startswith("---"))
        rule = route_policy(paths, add, dele)
        index.append({
            "scenario": name,
            "note": sc["note"],
            "files": paths,
            "changed_files": len(paths),
            "additions": add,
            "deletions": dele,
            "rule_route": rule,
            "rule_scale": ["tiny", "small", "moderate", "large", "huge"][churn_bin(add, dele)],
        })
        print(f"{name:<26} files={len(paths):>2} +{add:<4} -{dele:<4} 规则判定={rule}")

    (DIFFS / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"\n基线项目 → {BASE}")
    print(f"场景 diff → {DIFFS}（{len(index)} 个）")


if __name__ == "__main__":
    main()

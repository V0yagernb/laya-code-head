"""从 GitHub 挖真实的已关闭 PR，转成 Laya 训练用的一条条 case。

设计要点（重要，决定了能挖多少数据）：

1. **PR 元数据走 api.github.com**，一个请求返回 100 条，含 merged_at / additions /
   deletions / changed_files —— 这些就是 our 硬标签的来源。未认证限额 60 次/小时。
2. **diff 走 github.com/<repo>/pull/<n>.diff**，这是网页端点，**不计入 API 配额**。
   所以"元数据贵、diff 免费"，策略是：少请求元数据，多抓 diff。
3. 默认**不抓 reviews**（那要每个 PR 一次 API 请求）。如果你有 token，加
   `--with-reviews` 就能拿到 CHANGES_REQUESTED，把 review_route 换成真实标签。

用法：
    # 未认证：约能挖 2~4 个仓库 × 100 个 PR
    python -m src.mine_github --repos psf/requests,pallets/flask

    # 有 token（5000 次/小时）：想挖多少挖多少
    set GITHUB_TOKEN=ghp_xxx
    python -m src.mine_github --repos-file config/repos.txt --pages 5

输出：data/raw.jsonl，一行一个 PR。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

UA = {"User-Agent": "laya-code-head-miner/0.1", "Accept": "application/vnd.github+json"}
DIFF_UA = {"User-Agent": "Mozilla/5.0"}

API = "https://api.github.com"
DIFF_TMPL = "https://github.com/{repo}/pull/{num}.diff"

# diff 里文件路径的解析
FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.M)

# 机器人账号：Dependabot / Renovate 的版本升级 PR 量大、语义单薄，会污染训练集
BOT_HINTS = ("[bot]", "dependabot", "renovate", "github-actions", "copilot-swe")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


class RateLimited(RuntimeError):
    """GitHub 限流。

    wait_seconds 取 `X-RateLimit-Reset` 的倒计时 —— 长跑时睡这么久就能自动续上，
    比"整个 run 直接退出、等下一轮重跑"省得多（重跑会把已经抓过的分页再请求一遍）。
    它是 RuntimeError 的子类，所以老的 except RuntimeError 分支行为不变。
    """

    def __init__(self, code: int, wait_seconds: int):
        super().__init__(f"HTTP {code}（多半是限流），配额约 {wait_seconds} 秒后重置")
        self.code = code
        self.wait_seconds = wait_seconds


def api_get(url: str, retries: int = 3):
    headers = dict(UA)
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 429):
                reset = e.headers.get("X-RateLimit-Reset")
                wait_s = 0
                if reset:
                    wait_s = max(0, int(reset) - int(time.time())) + 5
                raise RateLimited(e.code, wait_s) from e
            if e.code == 404:
                return None
            if e.code >= 500 and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:  # 网络抖动
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(str(last))


def api_get_wait(url: str, wait_on_limit: bool, retries: int = 3):
    """api_get 的无人值守版：撞限流就睡到配额重置再重试同一个请求。"""
    while True:
        try:
            return api_get(url, retries=retries)
        except RateLimited as e:
            if not wait_on_limit:
                raise
            nap = e.wait_seconds or 60
            print(f"    [限流] 睡 {nap}s 等配额重置，然后继续...", file=sys.stderr)
            time.sleep(nap)


def _read_capped(r, max_bytes: int) -> tuple[bytes, bool]:
    """按上限读响应体。返回 (bytes, 是否顶到上限)。"""
    raw = r.read(max_bytes) if max_bytes else r.read()
    capped = bool(max_bytes) and len(raw) >= max_bytes
    return raw, capped


def fetch_diff_web(repo: str, num: int, timeout: int = 12, max_bytes: int = 0,
                   retries: int = 1) -> tuple[str | None, bool]:
    """走网页端点 github.com/<repo>/pull/<n>.diff —— **不吃 API 配额**。

    ⚠️ 实测坑：这条链路在有些网络下很不稳 —— TLS 握手只要 0.2s，但 HTTP 请求会
    整条挂住直到超时（20~36s 才返回 0 字节）。所以：
      * timeout 默认压到 12s（原来是 45s，一个挂住的请求会白占一个线程 45 秒）
      * 失败重试 retries 次（挂住是随机的，重试一次往往就过了）

    max_bytes > 0 时只读前这么多字节就断开：单条 diff 能到几 MB（python/cpython、
    kubernetes 的大 PR），而喂给模型的 state 只保留 4000 字符，全拉下来纯属浪费。
    截断只会让增删行数被低估，但触发上限的 diff 早已远超 churn>=400（huge/block），
    标签不会翻转。
    """
    url = DIFF_TMPL.format(repo=repo, num=num)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=DIFF_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw, capped = _read_capped(r, max_bytes)
            if raw:
                return raw.decode("utf-8", "ignore"), capped
        except Exception:
            pass
        if attempt < retries:
            time.sleep(0.3 * (attempt + 1))
    return None, False


def fetch_diff_api(repo: str, num: int, token: str | None, timeout: int = 30,
                   max_bytes: int = 0) -> tuple[str | None, bool]:
    """走 API 端点取 diff —— **每个 PR 消耗 1 次配额**，但链路可靠得多。

    api.github.com 在本机是通的（网页端点不稳），所以有 token 时这条路明显更快：
    5000 次/小时 ≈ 每秒 1.4 个 PR。请求头 Accept: application/vnd.github.v3.diff
    会让 GitHub 直接返回 unified diff 文本，和网页端点内容一致。
    """
    url = f"{API}/repos/{repo}/pulls/{num}"
    headers = dict(UA)
    headers["Accept"] = "application/vnd.github.v3.diff"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, capped = _read_capped(r, max_bytes)
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            reset = e.headers.get("X-RateLimit-Reset")
            wait_s = max(0, int(reset) - int(time.time())) + 5 if reset else 0
            raise RateLimited(e.code, wait_s) from e
        return None, False
    except Exception:
        return None, False
    if not raw:
        return None, False
    return raw.decode("utf-8", "ignore"), capped


def fetch_diff(repo: str, num: int, timeout: int = 12, max_bytes: int = 0,
               retries: int = 1, mode: str = "web",
               token: str | None = None) -> tuple[str | None, bool]:
    """取一条 diff，返回 (文本, 是否被下载上限截断)。

    三种模式：
      web     只用网页端点（不花配额，但随机挂住）
      api     只用 API（稳定，1 次配额/条）
      hybrid  **先用 web，失败才回落到 api** —— 有 token 时这是最优解：
              网页端点成功的那 70% 完全免费，配额只花在剩下 30% 上。
              实测 5000/小时的配额能覆盖 5000 条里失败的那部分，绰绰有余。
    """
    if mode == "api":
        return fetch_diff_api(repo, num, token, timeout, max_bytes)
    if mode == "hybrid":
        text, capped = fetch_diff_web(repo, num, timeout, max_bytes, retries)
        if text is None:
            text, capped = fetch_diff_api(repo, num, token, timeout, max_bytes)
        return text, capped
    return fetch_diff_web(repo, num, timeout, max_bytes, retries)


def rate_limit() -> dict:
    try:
        return api_get(f"{API}/rate_limit")["resources"]["core"]
    except Exception:
        return {}


def fetch_diffs(repo: str, cands: list, workers: int, sleep: float, max_bytes: int = 0,
                timeout: int = 12, retries: int = 1, mode: str = "web",
                token: str | None = None, wait_on_limit: bool = False):
    """并发抓 diff，边完成边产出 (key, pr, diff, 是否被下载截断)。

    只在飞 workers 个请求（滑动窗口），而不是把整页一次性提交 —— 有些 diff 有几百 KB，
    全攒在内存里没必要。失败是常态（见 fetch_diff_web 的坑），所以失败的直接跳过，
    不让一条坏数据拖死整轮。
    """
    def one(key, pr):
        return fetch_diff(repo, pr["number"], timeout, max_bytes, retries, mode, token)

    if workers <= 1:
        for key, pr in cands:
            try:
                diff, capped = one(key, pr)
            except RateLimited:
                raise
            yield key, pr, diff, capped
            if sleep:
                time.sleep(sleep)
        return

    it = iter(cands)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        inflight = {}
        while True:
            while len(inflight) < workers:
                try:
                    key, pr = next(it)
                except StopIteration:
                    break
                inflight[ex.submit(one, key, pr)] = (key, pr)
                if sleep:
                    time.sleep(sleep / workers)   # 把总节奏控制在和顺序抓取相当
            if not inflight:
                return
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                key, pr = inflight.pop(fut)
                try:
                    diff, capped = fut.result()
                except RateLimited as e:
                    if not wait_on_limit:
                        raise
                    nap = e.wait_seconds or 60
                    print(f"    [限流] 睡 {nap}s 等配额重置（这一条放弃）...", file=sys.stderr)
                    time.sleep(nap)
                    diff, capped = None, False
                except Exception:
                    diff, capped = None, False
                yield key, pr, diff, capped



# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------

def files_from_diff(diff: str) -> list[str]:
    out = []
    for a, b in FILE_RE.findall(diff):
        out.append(b if b != "/dev/null" else a)
    return out


def is_bot(pr: dict) -> bool:
    login = ((pr.get("user") or {}).get("login") or "").lower()
    return any(h in login for h in BOT_HINTS)


def churn_from_diff(diff: str) -> tuple[int, int]:
    """从 diff 自己数增删行。

    注意：GitHub 的 /pulls **列表**接口不返回 additions/deletions，只有单 PR 接口才有。
    单 PR 接口要每个 PR 一次请求，成本是 100 倍，所以这里直接自己数 —— 免费且够准。
    """
    add = dele = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            add += 1
        elif line.startswith("-"):
            dele += 1
    return add, dele


def open_hours(pr: dict) -> float | None:
    """PR 从创建到关闭挂了多久。

    按 updated 排序取到的"最近关闭"的 PR 里，有大量开出来几分钟就被关掉的
    低质量投稿（pallets/flask 上一页 100 个能合 3 个就是这原因）。这些样本的
    `merged=0` 标签虽然没错，但它反映的是"没人认真看过"，不是"代码有问题"，
    灌进训练集会教坏模型。用 --min-open-hours 把它们剔掉。
    """
    c, cl = pr.get("created_at"), pr.get("closed_at")
    if not c or not cl:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        import datetime as _dt
        d = (_dt.datetime.strptime(cl, fmt) - _dt.datetime.strptime(c, fmt))
        return d.total_seconds() / 3600.0
    except Exception:
        return None



def truncate_diff(diff: str, budget: int) -> tuple[str, bool]:
    """按字符预算截断 diff，但尽量保留完整 hunk 边界。"""
    if len(diff) <= budget:
        return diff, False
    cut = diff[:budget]
    nl = cut.rfind("\n")
    if nl > budget * 0.7:
        cut = cut[:nl]
    return cut + "\n... [diff truncated]", True


def build_raw(repo: str, pr: dict, diff: str, budget: int, with_reviews: bool,
              diff_capped: bool = False) -> dict | None:
    files = files_from_diff(diff)
    if not files:
        return None

    # 先在全量 diff 上数增删，再截断，否则长 diff 的规模会被低估
    additions, deletions = churn_from_diff(diff)
    if additions == 0 and deletions == 0:
        return None  # 二进制文件 / 纯重命名，没有可判断的内容

    diff_text, truncated = truncate_diff(diff, budget)
    body = (pr.get("body") or "").strip()
    if len(body) > 1500:
        body = body[:1500] + " ... [truncated]"

    rec = {
        "id": f"{repo}#{pr['number']}",
        "repo": repo,
        "number": pr["number"],
        "author": ((pr.get("user") or {}).get("login") or ""),
        "title": (pr.get("title") or "").strip(),
        "body": body,
        "base_branch": (pr.get("base") or {}).get("ref", ""),
        "files": files,
        "changed_files": len(files),
        "additions": additions,
        "deletions": deletions,
        "commits": pr.get("commits", 0),
        "review_comments": pr.get("review_comments", 0),
        "comments": pr.get("comments", 0),
        "labels": [l["name"] for l in pr.get("labels", [])],
        "created_at": pr.get("created_at"),
        "closed_at": pr.get("closed_at"),
        "merged": pr.get("merged_at") is not None,
        "open_hours": (round(_oh, 2) if (_oh := open_hours(pr)) is not None else None),
        "diff": diff_text,
        "diff_truncated": truncated,
        # 下载阶段就被上限截了：增删行数是下界。触发上限的 diff 一定已经远超 huge 档，
        # 所以标签不会翻转，但评测时可以把这批单独标出来看。
        "diff_download_capped": diff_capped,
    }
    if with_reviews:
        reviews = pr.get("_reviews") or []
        rec["review_states"] = [r.get("state") for r in reviews]
        rec["changes_requested"] = any(r.get("state") == "CHANGES_REQUESTED" for r in reviews)
    return rec


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def mine_repo(repo: str, pages: int, budget: int, with_reviews: bool,
              seen: set, out_f, sleep: float, min_files: int, max_per_repo: int,
              include_bots: bool = False, min_churn: int = 1,
              min_open_hours: float = 0.0, workers: int = 16,
              wait_on_limit: bool = False, max_diff_bytes: int = 200_000,
              diff_timeout: int = 12, diff_retries: int = 1,
              diff_mode: str = "web", page_retries: int = 2,
              retry_backoff: int = 45) -> int:
    added = 0
    for page in range(1, pages + 1):
        url = (f"{API}/repos/{repo}/pulls?state=closed&per_page=100&page={page}"
               f"&sort=updated&direction=desc")
        try:
            prs = api_get_wait(url, wait_on_limit)
        except Exception as e:
            print(f"  [!] {repo} page {page} 拉取失败：{e}", file=sys.stderr)
            return added
        if not prs:
            break

        # 先做一遍免费过滤：draft / 机器人 / 秒关 / 已经挖过。这些都不花 API 配额，
        # 放在抓 diff 之前能把并发批次缩小好几倍。
        cands = []
        for pr in prs:
            if pr.get("draft"):
                continue
            if not include_bots and is_bot(pr):
                continue
            if min_open_hours > 0:
                oh = open_hours(pr)
                if oh is not None and oh < min_open_hours:
                    continue
            key = f"{repo}#{pr['number']}"
            if key in seen:
                continue
            cands.append((key, pr))

        # 一页里失败太多，几乎一定是被 github.com 限速了（最坏情况整页 0 成功，实测见过
        # 60 个候选一个 diff 都拿不到）。这种时候要"退避 + 重试同一页"：PR 列表已经在手上，
        # 重试这批不花任何 API 配额，只是再走一遍网络。不这么做的话，一次限速就等于
        # 白白烧掉一整页的配额。
        pending = list(cands)
        n_ok = n_written = 0
        for attempt in range(page_retries + 1):
            if not pending:
                break
            failed = []
            for key, pr, diff, capped in fetch_diffs(repo, pending, workers, sleep, max_diff_bytes,
                                                     diff_timeout, diff_retries, diff_mode,
                                                     _token(), wait_on_limit):
                if not diff:
                    failed.append((key, pr))
                    continue
                n_ok += 1

                files = files_from_diff(diff)
                if len(files) < min_files:
                    continue

                if with_reviews:
                    try:
                        revs = api_get_wait(f"{API}/repos/{repo}/pulls/{pr['number']}/reviews",
                                            wait_on_limit)
                        pr["_reviews"] = revs or []
                    except Exception as e:
                        print(f"    reviews {key} 失败：{e}", file=sys.stderr)

                rec = build_raw(repo, pr, diff, budget, with_reviews, capped)
                if not rec:
                    continue
                if rec["additions"] + rec["deletions"] < min_churn:
                    continue
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                seen.add(key)
                added += 1
                n_written += 1

                if added % 25 == 0:
                    print(f"    ... 已写 {added} 条")
                if max_per_repo and added >= max_per_repo:
                    print(f"  {repo}: 达到 --max-per-repo={max_per_repo}，停止")
                    return added

            pending = failed
            # 只有"大面积失败"才值得退避重试；零星几条失败多半是 PR 本身没 diff
            if failed and attempt < page_retries and len(failed) >= max(3, int(0.5 * len(cands))):
                nap = retry_backoff * (attempt + 1)
                print(f"  [!] {repo} page {page}: {len(failed)}/{len(cands)} 条没拿到 diff"
                      f"（疑似被限速），睡 {nap}s 后重试这批（不花配额）", file=sys.stderr)
                time.sleep(nap)

        # 这行是用来判断"数据挖得慢"到底慢在哪一步的：候选多但 diff 拿不到 = 网络/限流，
        # diff 拿到了但写出少 = 过滤规则太紧
        print(f"  {repo} page {page}: 候选 {len(cands)} → 拿到 diff {n_ok} → 写出 {n_written}"
              f"（累计新增 {added}）")
        if len(prs) < 100:
            break
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="", help="逗号分隔，如 psf/requests,pallets/flask")
    ap.add_argument("--repos-file", default="", help="每行一个 owner/name")
    ap.add_argument("--pages", type=int, default=2, help="每仓库翻多少页（每页 100 个 PR）")
    ap.add_argument("--out", default="data/raw.jsonl")
    ap.add_argument("--diff-budget", type=int, default=4000, help="单个 diff 保留多少字符")
    ap.add_argument("--min-files", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.25, help="两次抓 diff 之间歇多久")
    ap.add_argument("--with-reviews", action="store_true",
                    help="额外抓 reviews（每 PR 1 次 API 请求，未认证时慎用）")
    ap.add_argument("--max-per-repo", type=int, default=0, help="每仓库最多写多少条（0=不限）")
    ap.add_argument("--min-churn", type=int, default=1, help="低于这个增删行数直接丢弃")
    ap.add_argument("--include-bots", action="store_true", help="不排除 dependabot/renovate 等机器人 PR")
    ap.add_argument("--min-open-hours", type=float, default=0.0,
                    help="丢弃关闭得太快的 PR（小时）。建议 1~6，能滤掉秒关的低质量投稿")
    ap.add_argument("--workers", type=int, default=16,
                    help="并发抓 diff 的线程数。web 模式不吃 API 配额，只有网络往返成本，"
                         "而且这条链路的连接经常整条挂住，所以要靠并发 + 短超时把它们摊掉")
    ap.add_argument("--wait-on-limit", action="store_true",
                    help="撞 API 限流时睡到配额重置再继续，而不是整轮退出（长跑必备）")
    ap.add_argument("--max-diff-bytes", type=int, default=200_000,
                    help="单条 diff 最多下载多少字节（0=不限）。只影响增删行数的精度下界，"
                         "不影响喂给模型的 4000 字符；不限制的话 python/cpython 这类仓库"
                         "的大 PR 会把吞吐拖到不可用")
    ap.add_argument("--diff-mode", default="hybrid", choices=["web", "api", "hybrid"],
                    help="web=github.com/<repo>/pull/<n>.diff（不吃配额，但本机实测经常挂住）；"
                         "api=api.github.com 取 diff（每 PR 花 1 次配额，链路稳）；"
                         "hybrid=先 web，失败才回落 api（有 token 时最优：配额只花在失败的那部分）")
    ap.add_argument("--diff-timeout", type=int, default=12,
                    help="单条 diff 的超时秒数。挂住的连接会白占一个线程，所以要短")
    ap.add_argument("--diff-retries", type=int, default=1,
                    help="单条 diff 失败后重试几次（挂住是随机的，重试一次往往就过了）")
    ap.add_argument("--page-retries", type=int, default=2,
                    help="整页大面积抓失败（被限速）时，退避后重试这一页几次。"
                         "PR 列表已经在手，重试不花 API 配额")
    ap.add_argument("--retry-backoff", type=int, default=45,
                    help="整页重试前的退避秒数（按重试次数线性递增）")
    ap.add_argument("--shuffle", action="store_true",
                    help="打乱仓库顺序。配额不够整轮跑完时，避免永远只挖到清单开头那几个")
    args = ap.parse_args()

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    if args.repos_file:
        # utf-8-sig：清单是人手写的，很容易带 BOM；带了 BOM 的话第一行的 "#" 不会被认成注释，
        # 会被当成一个仓库名去请求（白白浪费一次配额换一个 404）
        for line in Path(args.repos_file).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                repos.append(line)
    if not repos:
        print("必须给 --repos 或 --repos-file", file=sys.stderr)
        return 2
    if args.shuffle:
        random.shuffle(repos)

    rl = rate_limit()
    tok = _token()
    if rl:
        print(f"GitHub API 配额：{rl.get('remaining')}/{rl.get('limit')}"
              f"{'（已认证）' if tok else '（未认证，限 60/小时）'}")
    print(f"仓库 {len(repos)} 个 × {args.pages} 页，抓 diff 并发 {args.workers}"
          f"（{args.diff_mode} 模式，超时 {args.diff_timeout}s）"
          f"{'，限流自动等待' if args.wait_on_limit else ''}")
    if args.diff_mode in ("api", "hybrid") and not tok:
        print("  [!] 没有 GITHUB_TOKEN：api 回落会被 60 次/小时的匿名额度卡住，"
              "建议加 --diff-mode web 或先配 token", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["id"])
            except Exception:
                pass
        print(f"已存在 {len(seen)} 条，本次续写")

    total = 0
    with out_path.open("a", encoding="utf-8") as f:
        for repo in repos:
            print(f"\n== {repo} ==")
            try:
                total += mine_repo(repo, args.pages, args.diff_budget, args.with_reviews,
                                   seen, f, args.sleep, args.min_files, args.max_per_repo,
                                   args.include_bots, args.min_churn, args.min_open_hours,
                                   args.workers, args.wait_on_limit, args.max_diff_bytes,
                                   args.diff_timeout, args.diff_retries, args.diff_mode,
                                   args.page_retries, args.retry_backoff)
            except RuntimeError as e:
                print(f"  [!] 配额/网络中断，先停下：{e}", file=sys.stderr)
                print(f"  已挖到的数据在 {out_path}，等配额重置后重跑同一命令会续写。")
                break

    print(f"\n完成：本次新增 {total} 条，累计 {len(seen)} 条 → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

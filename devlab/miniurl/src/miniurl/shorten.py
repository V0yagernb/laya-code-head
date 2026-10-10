"""base62 短码生成与还原。"""

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

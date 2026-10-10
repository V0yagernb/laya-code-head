"""存储层：内存实现 + 一个抽象基类。

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

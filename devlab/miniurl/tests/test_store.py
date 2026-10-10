import pytest

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

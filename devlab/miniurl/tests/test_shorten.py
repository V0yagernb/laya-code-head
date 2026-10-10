import pytest

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

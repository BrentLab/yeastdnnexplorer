import time

import pytest

from yeastdnnexplorer.interface.Cache import Cache


def test_cache_set_and_get():
    cache = Cache()
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("key2", "default_value") == "default_value"


def test_cache_list():
    cache = Cache()
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    keys = cache.list()
    assert "key1" in keys
    assert "key2" in keys


def test_cache_delete():
    cache = Cache()
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.delete("key1")
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"


def test_cache_ttl():
    cache = Cache(ttl=1)  # TTL set to 1 second
    cache.set("key1", "value1")
    time.sleep(1.5)  # Wait for TTL to expire
    assert cache.get("key1") is None  # Should be None after TTL expiry


def test_cache_lru():
    cache = Cache(maxsize=2)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")  # This should evict "key1" if LRU works
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_separate_cache_instances():
    cache1 = Cache()
    cache2 = Cache()

    cache1.set("key1", "value1")
    cache2.set("key2", "value2")

    # Ensure they don't share state
    assert cache1.get("key1") == "value1"
    assert cache1.get("key2") is None

    assert cache2.get("key2") == "value2"
    assert cache2.get("key1") is None


if __name__ == "__main__":
    pytest.main()

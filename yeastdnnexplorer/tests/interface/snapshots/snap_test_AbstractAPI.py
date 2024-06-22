# snapshottest: v1 - https://goo.gl/zC4yUc

from snapshottest import Snapshot

snapshots = Snapshot()

snapshots["test_cache_operations cache_get_after_delete"] = None

snapshots["test_cache_operations cache_get_after_set"] = "test_value"

snapshots["test_cache_operations cache_list"] = ["test_key"]

snapshots["test_pop_params pop_params_after_all_removed"] = {}

snapshots["test_pop_params pop_params_after_one_removed"] = {"param2": "value2"}

snapshots["test_push_params push_params"] = {"param1": "value1", "param2": "value2"}

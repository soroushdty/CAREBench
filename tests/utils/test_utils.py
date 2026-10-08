"""Tests for shared/utils/* modules."""
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.utils.text_utils import normalize_for_matching, trim_item_value
from shared.utils.path_utils import is_remote_location
from shared.utils.array_utils import as_float32_array
from shared.utils.file_utils import (
    ensure_parent_dir,
    write_csv_file,
    remove_file_if_exists,
    write_text_file,
)
from shared.utils.json_utils import load_json_from_source, load_json_from_cfg, save_json_file
from shared.utils.mapping_utils import (
    is_valid_grouped_mapping_obj,
    _load_grouped_mapping,
    _build_reverse_lookup,
)


# ---------------------------------------------------------------------------
# text_utils
# ---------------------------------------------------------------------------

class TestNormalizeForMatching:
    def test_casefolds(self):
        assert normalize_for_matching("Hello") == "hello"

    def test_strips_whitespace_by_default(self):
        assert normalize_for_matching("  hello  ") == "hello"

    def test_no_strip_when_false(self):
        result = normalize_for_matching("  hello  ", strip=False)
        assert result == "  hello  "

    def test_nan_passthrough(self):
        assert math.isnan(normalize_for_matching(float("nan")))

    def test_pd_na_passthrough(self):
        result = normalize_for_matching(pd.NA)
        assert pd.isna(result)

    def test_unicode_casefold(self):
        # German ß → ss via casefold
        assert normalize_for_matching("STRAßE") == "strasse"

    def test_non_string_coerced(self):
        assert normalize_for_matching(42) == "42"


class TestTrimItemValue:
    def test_strips_leading_trailing_whitespace(self):
        assert trim_item_value("  foo  ") == "foo"

    def test_no_change_when_already_trimmed(self):
        assert trim_item_value("foo") == "foo"

    def test_nan_passthrough(self):
        assert math.isnan(trim_item_value(float("nan")))

    def test_pd_na_passthrough(self):
        result = trim_item_value(pd.NA)
        assert pd.isna(result)

    def test_non_string_coerced(self):
        assert trim_item_value(123) == "123"


# ---------------------------------------------------------------------------
# path_utils
# ---------------------------------------------------------------------------

class TestIsRemoteLocation:
    def test_http_is_remote(self):
        assert is_remote_location("http://example.com/data.json") is True

    def test_https_is_remote(self):
        assert is_remote_location("https://example.com/data.json") is True

    def test_local_path_is_not_remote(self):
        assert is_remote_location("/home/user/data.json") is False

    def test_relative_path_is_not_remote(self):
        assert is_remote_location("examples/synthetic/mapping.json") is False

    def test_empty_string_is_not_remote(self):
        assert is_remote_location("") is False

    def test_ftp_is_not_remote(self):
        assert is_remote_location("ftp://example.com/file") is False


# ---------------------------------------------------------------------------
# array_utils
# ---------------------------------------------------------------------------

class TestAsFloat32Array:
    def test_returns_float32(self):
        arr = as_float32_array([1, 2, 3], name="test")
        assert arr.dtype == np.float32

    def test_nan_filled_with_zero(self):
        arr = as_float32_array([1.0, float("nan"), 3.0], name="test")
        assert arr[1] == 0.0
        assert not np.isnan(arr).any()

    def test_nan_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            as_float32_array([float("nan")], name="myarray")
        assert "myarray" in caplog.text
        assert "NaN" in caplog.text

    def test_object_dtype_coerced(self):
        arr = as_float32_array(np.array(["1.0", "2.5"], dtype=object), name="test")
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, [1.0, 2.5])

    def test_already_float32_unchanged(self):
        original = np.array([1.0, 2.0], dtype=np.float32)
        arr = as_float32_array(original, name="test")
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, original)

    def test_shape_preserved(self):
        original = np.ones((4, 3), dtype=np.float64)
        arr = as_float32_array(original, name="test")
        assert arr.shape == (4, 3)


# ---------------------------------------------------------------------------
# file_utils
# ---------------------------------------------------------------------------

class TestEnsureParentDir:
    def test_creates_nested_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "file.txt"
        result = ensure_parent_dir(target)
        assert (tmp_path / "a" / "b" / "c").is_dir()
        assert result == target

    def test_returns_path_object(self, tmp_path):
        target = tmp_path / "sub" / "file.txt"
        result = ensure_parent_dir(target)
        assert isinstance(result, Path)

    def test_idempotent(self, tmp_path):
        target = tmp_path / "sub" / "file.txt"
        ensure_parent_dir(target)
        ensure_parent_dir(target)  # second call must not raise
        assert (tmp_path / "sub").is_dir()

    def test_accepts_string_path(self, tmp_path):
        target = str(tmp_path / "str_sub" / "file.txt")
        result = ensure_parent_dir(target)
        assert (tmp_path / "str_sub").is_dir()
        assert isinstance(result, Path)


class TestWriteCsvFile:
    def test_writes_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "out.csv"
        write_csv_file(df, path)
        df_read = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, df_read)

    def test_no_index_by_default(self, tmp_path):
        df = pd.DataFrame({"x": [10]})
        path = tmp_path / "out.csv"
        write_csv_file(df, path)
        with open(path) as f:
            header = f.readline().strip()
        assert header == "x"

    def test_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame({"v": [1]})
        path = tmp_path / "nested" / "dir" / "out.csv"
        write_csv_file(df, path)
        assert path.exists()


class TestRemoveFileIfExists:
    def test_removes_existing_file_returns_true(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert remove_file_if_exists(f) is True
        assert not f.exists()

    def test_missing_file_returns_false(self, tmp_path):
        f = tmp_path / "nonexistent.txt"
        assert remove_file_if_exists(f) is False

    def test_directory_returns_false(self, tmp_path):
        # Directories should not be deleted
        assert remove_file_if_exists(tmp_path) is False


class TestWriteTextFile:
    def test_writes_content(self, tmp_path):
        path = tmp_path / "out.txt"
        write_text_file("hello world", path)
        assert path.read_text(encoding="utf-8") == "hello world"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "out.txt"
        write_text_file("content", path)
        assert path.exists()

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "out.txt"
        write_text_file("first", path)
        write_text_file("second", path)
        assert path.read_text(encoding="utf-8") == "second"


# ---------------------------------------------------------------------------
# json_utils
# ---------------------------------------------------------------------------

class TestLoadJsonFromSource:
    def test_loads_from_local_file(self, tmp_path):
        data = {"key": "value", "num": 42}
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = load_json_from_source(str(f))
        assert result == data

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json_from_source(str(tmp_path / "missing.json"))

    def test_inline_payload_accepted_when_allowed(self):
        payload = '{"a": 1}'
        result = load_json_from_source(payload, allow_json_payload=True)
        assert result == {"a": 1}

    def test_inline_payload_rejected_by_default(self, tmp_path):
        payload = '{"a": 1}'
        with pytest.raises(FileNotFoundError):
            load_json_from_source(payload)

    def test_inline_list_payload(self):
        payload = '[1, 2, 3]'
        result = load_json_from_source(payload, allow_json_payload=True)
        assert result == [1, 2, 3]

    def test_base_dir_resolves_relative_path(self, tmp_path):
        data = {"x": 99}
        f = tmp_path / "rel.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = load_json_from_source("rel.json", base_dir=tmp_path)
        assert result == data


class TestLoadJsonFromCfg:
    def test_returns_none_when_key_absent(self, tmp_path):
        result = load_json_from_cfg({}, key="DIR_JSON_MAP")
        assert result is None

    def test_returns_none_when_value_empty_string(self, tmp_path):
        result = load_json_from_cfg({"DIR_JSON_MAP": ""}, key="DIR_JSON_MAP")
        assert result is None

    def test_loads_from_path_in_cfg(self, tmp_path):
        data = {"z": 7}
        f = tmp_path / "map.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = load_json_from_cfg({"DIR_JSON_MAP": str(f)})
        assert result == data


class TestSaveJsonFile:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "out.json"
        save_json_file({"a": 1, "b": [1, 2]}, path)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == {"a": 1, "b": [1, 2]}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.json"
        save_json_file({"x": 1}, path)
        assert path.exists()

    def test_stringify_keys(self, tmp_path):
        path = tmp_path / "out.json"
        save_json_file({1: "one", 2: "two"}, path, stringify_keys=True)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == {"1": "one", "2": "two"}

    def test_indent_is_applied(self, tmp_path):
        path = tmp_path / "out.json"
        save_json_file({"k": "v"}, path, indent=4)
        raw = path.read_text(encoding="utf-8")
        # indent=4 means at least 4 spaces before keys
        assert "    " in raw


# ---------------------------------------------------------------------------
# mapping_utils
# ---------------------------------------------------------------------------

class TestIsValidGroupedMappingObj:
    def test_valid_mapping(self):
        assert is_valid_grouped_mapping_obj({"a": ["b", "c"]}) is True

    def test_valid_with_tuple_values(self):
        assert is_valid_grouped_mapping_obj({"a": ("b",)}) is True

    def test_valid_with_set_values(self):
        assert is_valid_grouped_mapping_obj({"a": {"b"}}) is True

    def test_empty_dict_is_valid(self):
        assert is_valid_grouped_mapping_obj({}) is True

    def test_non_dict_returns_false(self):
        assert is_valid_grouped_mapping_obj([]) is False
        assert is_valid_grouped_mapping_obj("string") is False
        assert is_valid_grouped_mapping_obj(None) is False

    def test_string_value_returns_false(self):
        assert is_valid_grouped_mapping_obj({"a": "b"}) is False

    def test_non_str_key_returns_false(self):
        assert is_valid_grouped_mapping_obj({1: ["a"]}) is False

    def test_list_with_non_str_element_returns_false(self):
        assert is_valid_grouped_mapping_obj({"a": [1, 2]}) is False

    def test_int_value_returns_false(self):
        assert is_valid_grouped_mapping_obj({"a": 42}) is False


class TestLoadGroupedMapping:
    def test_from_dict_normalizes_keys_and_aliases(self):
        result = _load_grouped_mapping({"Hello World": ["ALIAS One"]})
        assert "hello world" in result
        assert "alias one" in result["hello world"]

    def test_type_error_for_invalid_input(self):
        with pytest.raises(TypeError):
            _load_grouped_mapping(42)

    def test_value_error_for_non_list_value(self):
        with pytest.raises(ValueError):
            _load_grouped_mapping({"key": "not_a_list"})

    def test_value_error_for_non_str_alias(self):
        with pytest.raises(ValueError):
            _load_grouped_mapping({"key": [123]})

    def test_loads_from_json_file(self, tmp_path):
        data = {"Apple": ["APPLE", "apple pie"]}
        f = tmp_path / "map.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = _load_grouped_mapping(str(f))
        assert "apple" in result

    def test_remove_spaces_false_keeps_spaces(self):
        result = _load_grouped_mapping({"Hello World": ["Alias One"]}, remove_spaces=False)
        # casefold still applied, strip not applied (no leading/trailing here anyway)
        assert "hello world" in result


class TestBuildReverseLookup:
    def test_canonical_maps_to_itself(self):
        mapping = {"apple": ["Apfel", "Pomme"]}
        normalized = _load_grouped_mapping(mapping)
        reverse = _build_reverse_lookup(normalized)
        assert reverse["apple"] == "apple"

    def test_aliases_map_to_canonical(self):
        mapping = {"apple": ["apfel", "pomme"]}
        result = _build_reverse_lookup(mapping)
        assert result["apfel"] == "apple"
        assert result["pomme"] == "apple"

    def test_conflicting_alias_raises_value_error(self):
        # Build a mapping that puts the same alias under two canonicals
        raw = {"apple": ["shared"], "orange": ["shared"]}
        with pytest.raises(ValueError, match="shared"):
            _build_reverse_lookup(raw)

    def test_empty_mapping_returns_empty_dict(self):
        assert _build_reverse_lookup({}) == {}

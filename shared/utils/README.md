# shared/utils/

**Canonical path:** `shared/utils/`

General-purpose utility modules shared across all evaluation tracks.

## Modules

- `array_utils.py` — numpy array helpers (e.g., `as_float32_array`)
- `file_utils.py` — file I/O helpers (e.g., `ensure_parent_dir`, `write_csv_file`, `write_text_file`, `remove_file_if_exists`)
- `json_utils.py` — JSON loading with support for local paths, remote URLs, and inline payloads (e.g., `load_json_from_source`, `save_json_file`)
- `mapping_utils.py` — grouped mapping lookup utilities (e.g., `is_valid_grouped_mapping_obj`, `_load_grouped_mapping`, `_build_reverse_lookup`)
- `path_utils.py` — path resolution helpers (e.g., `is_remote_location`)
- `text_utils.py` — text normalization (e.g., `normalize_for_matching`, `trim_item_value`)

## Usage

```python
from shared.utils.file_utils import ensure_parent_dir
from shared.utils.json_utils import load_json_from_source, save_json_file
from shared.utils.array_utils import as_float32_array
from shared.utils.text_utils import normalize_for_matching
from shared.utils.mapping_utils import is_valid_grouped_mapping_obj
from shared.utils.path_utils import is_remote_location
```

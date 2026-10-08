# CAREBench Summary Utilities

**Canonical path:** `shared/reporting/input_summary/utils`

This directory houses cross-cutting helpers used exclusively by the summary generation system to manage file creation, formatting, and cleanup.

## Core Modules

### `io_utils.py`
Provides safe abstraction wrappers for filesystem I/O operations (e.g., `write_csv`, `write_missingness_workbook`). It checks global configuration dictionaries (like `summary_cfg`) to determine if a specific output artifact has been disabled by the user before writing.

### `cleanup.py`
Maintains repository cleanliness by aggressively scanning the output directories for obsolete or legacy summary filenames and recursively removing them before generating the new snapshot.

### `logging_utils.py`
Sets up dedicated file handlers and formatters to record the high-level progress of the summary generation phase, isolating summary diagnostics from the master pipeline logs.

### `settings_loader.py`
Responsible for fetching and providing the default YAML configurations that dictate default plot sizes, text bins, and which diagnostic tables are enabled by default.

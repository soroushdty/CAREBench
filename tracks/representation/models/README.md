# tracks/representation/models/

Track 1 model class definitions for representation learning.

This directory contains the model classes copied from `models/`. It includes:

- **`ConstantCalibrator.py`** — Calibrator that outputs a constant probability
- **`EnsemblePredictor.py`** — Ensemble predictor combining multiple base models; includes `load_ensemble_predictor` and `write_ensemble_manifest`
- **`ModelRegistry.py`** — Registry for model constructors; includes `configure_registry` and `get_registry`
- **`MultiLabelModel.py`** — Multi-label classification model wrapper
- **`Preprocessor.py`** — Feature preprocessing pipeline

## Canonical path

`tracks/representation/models/`

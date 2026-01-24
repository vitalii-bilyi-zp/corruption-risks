# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Building restoration cost prediction system for Ukrainian war-damaged buildings. Uses machine learning (XGBoost primary, with Linear Regression and Random Forest alternatives) to predict restoration costs based on building attributes. Provides a Flask REST API with SHAP-based explainability.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (generates .pkl and .json artifacts)
python train.py          # XGBoost (primary)
python train_lr.py       # Linear Regression
python train_rf.py       # Random Forest

# Run prediction scripts standalone
python predict.py
python predict_lr.py
python predict_rf.py

# Start API server (runs on 0.0.0.0:5000)
python app.py
```

## Architecture

### Data Flow
```
Raw CSV → Training Script → Model (.pkl) + Metadata (.json)
                                    ↓
API Request → Pydantic Validation → Encoding (version-based) → Model Inference → Response
```

### API Endpoints (app.py)
- `GET /health` - Health check
- `POST /predict` - Basic prediction (returns cost only)
- `POST /predict_explain` - Prediction with SHAP explainability (contributions per feature group)

All endpoints except `/health` require `X-API-Key` header (configured via `API_KEY` env var).

### Input Schema
```python
{
    "area": float,        # Building area in m² (≥0)
    "floors": int,        # Number of floors (≥0)
    "building_type": str, # Categorical
    "damage_level": str,  # Categorical: Легке, Середнє, Тяжке
    "region": str,        # Categorical (22 Ukrainian regions/cities)
    "repair_type": str    # Categorical: Капітальний, Повна реконструкція, Поточний
}
```

### Generated Artifacts
Training scripts produce:
- `restoration_model*.pkl` - Serialized model
- `feature_columns.json` - Feature column names after encoding
- `feature_groups.json` - Mapping of encoded columns to original features + version field
- `global_importance.json` - SHAP-based feature importance percentages

### Feature Processing (encoding.py)
Centralized encoding module with version-based backward compatibility:

**Version 2.0 (current)** - Fuzzy encoding for ordinal features:
- `damage_level`, `repair_type` - triangular membership functions preserving gradient information
- `building_type`, `region` - standard one-hot encoding
- Ordinal positions: Легке→0.0, Середнє→0.5, Тяжке→1.0

**Version 1.0 (legacy)** - Pure one-hot encoding for all categorical features

Version is determined by `"version"` field in `feature_groups.json`. API auto-detects version at runtime.

### Encoding Comparison Script
```bash
python compare_encodings.py  # A/B comparison of one-hot vs fuzzy with k-fold CV
```

### Explainability
Primary: SHAP TreeExplainer for instance-level contributions. Fallback: XGBoost gain-based importance. Both aggregate one-hot columns back into human-readable feature groups.

## Environment Variables

- `API_KEY` - Required for API authentication
- `MODEL_PATH` - Model file path (default: `restoration_model.pkl`)
- `FEATURE_COLUMNS_PATH` - Feature columns JSON (default: `feature_columns.json`)
- `FEATURE_GROUPS_PATH` - Feature groups JSON (default: `feature_groups.json`)

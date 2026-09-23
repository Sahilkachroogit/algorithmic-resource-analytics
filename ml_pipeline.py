"""
ml_pipeline.py
==============
Ingests telemetry_raw.csv produced by telemetry_gen.cpp, cleans it with a per-group
rolling median filter, trains two RandomForestRegressor models (execution time
and memory footprint), reports MAE + R², and serializes the pipeline to
model.pkl.

Dependencies: pandas, scikit-learn, joblib
Run:  python ml_pipeline.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------------
df = pd.read_csv(
    "telemetry_raw.csv",
    dtype={
        "algorithm":          "category",
        "input_size":         "int64",
        "execution_time_us":  "float64",
        "memory_bytes":       "float64",
    },
)

# ---------------------------------------------------------------------------
# 2. Vectorized cleaning — rolling median spike filter (per algorithm group)
#    Replaces any sample whose execution_time_us deviates by more than
#    2× the local rolling median with that median, suppressing OS jitter.
#    All operations are vectorized via groupby + transform; no Python loops.
# ---------------------------------------------------------------------------
WINDOW = 5  # samples; odd keeps the current sample centred

def _rolling_median(s: pd.Series) -> pd.Series:
    """Return a rolling median series aligned to the original index."""
    return (
        s.rolling(window=WINDOW, min_periods=1, center=True)
         .median()
    )

# Compute smoothed series for both targets in a single groupby pass
smoothed = (
    df.groupby("algorithm", observed=True)[["execution_time_us", "memory_bytes"]]
      .transform(_rolling_median)
)

# Spike mask: samples where raw value deviates > 2× the local rolling median
# (guard against division-by-zero when median == 0 with np.where)
denom_time = smoothed["execution_time_us"].replace(0, np.nan)
spike_mask = (df["execution_time_us"] / denom_time).fillna(1.0) > 2.0

# Vectorised replacement: assign smoothed value only where spike detected
df["execution_time_us"] = np.where(
    spike_mask,
    smoothed["execution_time_us"],
    df["execution_time_us"],
)
# memory_bytes is cumulative/deterministic — apply median smoothing uniformly
# to reduce any quantisation noise without spike-gating
df["memory_bytes"] = smoothed["memory_bytes"]

# ---------------------------------------------------------------------------
# 3. Feature engineering (vectorised)
#    log1p of input_size linearises the geometric scale for the tree splits.
# ---------------------------------------------------------------------------
df["log_input_size"] = np.log1p(df["input_size"].to_numpy())

FEATURES  = ["log_input_size", "algorithm"]
TARGETS   = ["execution_time_us", "memory_bytes"]

X = df[FEATURES].copy()
y = df[TARGETS].to_numpy()          # shape (n, 2)

# ---------------------------------------------------------------------------
# 4. Train / test split  (stratify on algorithm to keep all classes in both)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=df["algorithm"].cat.codes,
)

# ---------------------------------------------------------------------------
# 5. Preprocessing + model pipeline
#    OrdinalEncoder handles the categorical 'algorithm' column inside the
#    sklearn Pipeline so the full object is serialisable with a single pickle.
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", ["log_input_size"]),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         ["algorithm"]),
    ],
    remainder="drop",
    sparse_threshold=0,
)

base_rf = RandomForestRegressor(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)

model = Pipeline([
    ("prep",  preprocessor),
    ("model", MultiOutputRegressor(base_rf, n_jobs=1)),
    # n_jobs=1 here; parallelism is already inside each RandomForest (n_jobs=-1)
])

model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 6. Evaluation — MAE and R² per target, then print clean summary
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

target_labels = ["execution_time_us", "memory_bytes"]
for i, label in enumerate(target_labels):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    r2  = r2_score(y_test[:, i], y_pred[:, i])
    print(f"{label:25s}  MAE={mae:>12.4f}  R²={r2:>8.6f}")

# ---------------------------------------------------------------------------
# 7. Serialize — compress=3 gives good size/speed tradeoff with no extra deps
# ---------------------------------------------------------------------------
joblib.dump(model, "model.pkl", compress=3)


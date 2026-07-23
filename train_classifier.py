"""
Phase 3: train a classifier to predict accident risk from the scene
variables extracted in Phases 1-2, replacing v1's hand-picked
DEFAULT_WEIGHTS (v1_legacy/risk_model.py) with data-derived feature
importances. See docs/01-phases-and-roadmap.md's Phase 3 section.

Known limitation this script has to evaluate honestly, not hide: risk_label
was derived (mine_risk_labels.py) from an explicit speed mentioned in each
video's caption where possible (risk_label_source == "text", 202/303 rows).
For the other 101/303 rows ("cv_fallback"), the label's base severity score
came from the SAME vehicle_speed bucket that is also a model input feature
here -- so for those rows, vehicle_speed and risk_label are correlated
partly by construction. This script reports accuracy on ALL rows and on
TEXT-ONLY rows separately, from the same out-of-fold predictions, so the
size of that gap is visible rather than hidden behind one flattering number.

Run: python train_classifier.py
Reads: data/scene_dataset.csv
Writes: outputs/classifier/{classifier.joblib, feature_importance.csv,
        cv_predictions.csv, evaluation_report.txt}
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_CSV = Path("data/scene_dataset.csv")
OUTPUT_DIR = Path("outputs/classifier")

CATEGORICAL_COLS = [
    "vehicle_speed", "pedestrian_distance", "closing_risk", "weather",
    "visibility", "road_type", "location", "obstacle", "braking",
]
NUMERIC_COLS = ["num_pedestrians", "traffic_density"]
TARGET_COL = "risk_label"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5


def load_dataset():
    """Splits scene_dataset.csv into X (the 11 scene variables), y
    (risk_label), and label_source (risk_label_source -- kept separate,
    not a feature, used only for the ALL vs. TEXT-ONLY evaluation split).

    Explicitly excluded from X: video_id/source/split (identifiers, not
    scene variables), risk_score_raw (the continuous pre-bucketing version
    of the target itself -- including it would be near-total leakage), and
    risk_label_source (see the module docstring -- it's about how the
    label was built, not a property of the scene).
    """
    df = pd.read_csv(DATA_CSV)
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = df[TARGET_COL].copy()
    label_source = df["risk_label_source"].copy()
    video_id = df["video_id"].copy()
    return X, y, label_source, video_id


def build_pipeline(random_state=RANDOM_STATE):
    """One ColumnTransformer + classifier, saved as a single fitted object
    so Phase 4 can call pipeline.predict(...) directly without
    reimplementing the encoding logic separately."""
    preprocess = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                CATEGORICAL_COLS,
            ),
            ("num", "passthrough", NUMERIC_COLS),
        ],
        verbose_feature_names_out=False,
    )
    classifier = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def _report_block(title, y_true, y_pred):
    lines = [title, "-" * len(title)]
    lines.append(classification_report(y_true, y_pred, digits=3))
    lines.append("Confusion matrix (rows=true, cols=predicted), labels sorted:")
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = "        " + "  ".join(f"{l:>8}" for l in labels)
    lines.append(header)
    for label, row in zip(labels, cm):
        lines.append(f"{label:>8}" + "".join(f"{v:>10}" for v in row))
    return "\n".join(lines)


def run_cross_validation(X, y, label_source, video_id):
    """Pooled out-of-fold predictions from a single 5-fold CV pass, then
    ALL vs. TEXT-ONLY reports built from that same table -- not two
    separately trained models. See the module docstring for why."""
    pipeline = build_pipeline()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)

    predictions = pd.DataFrame({
        "video_id": video_id,
        "risk_label": y,
        "predicted_label": y_pred,
        "risk_label_source": label_source,
    })

    all_report = _report_block("ALL (303 rows)", predictions["risk_label"], predictions["predicted_label"])

    text_only = predictions[predictions["risk_label_source"] == "text"]
    text_report = _report_block(
        f"TEXT-ONLY ({len(text_only)} rows)", text_only["risk_label"], text_only["predicted_label"]
    )

    cv_fallback = predictions[predictions["risk_label_source"] == "cv_fallback"]
    fallback_report = _report_block(
        f"CV_FALLBACK-ONLY ({len(cv_fallback)} rows, sanity companion)",
        cv_fallback["risk_label"], cv_fallback["predicted_label"],
    )

    # Stability check: is the pooled accuracy above an artifact of one
    # particular shuffle, or does it hold up across repeated CV splits?
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    scores = cross_val_score(build_pipeline(), X, y, cv=rskf, scoring="accuracy")
    stability_line = (
        f"Repeated CV accuracy ({N_SPLITS}-fold x {N_REPEATS} repeats): "
        f"mean={scores.mean():.3f}, std={scores.std():.3f}"
    )

    return predictions, all_report, text_report, fallback_report, stability_line


def run_baseline(X, y):
    """DummyClassifier sanity floor -- the real model's accuracy needs to
    clearly beat this before feature importances mean anything."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, y, cv=skf)
    report = _report_block("BASELINE: DummyClassifier(most_frequent)", y, y_pred)
    return report


def compute_feature_importance(fitted_pipeline):
    """Aggregates per-expanded-column importances (e.g. 'vehicle_speed_low')
    back to per-original-variable importances (e.g. 'vehicle_speed'), via
    longest-prefix matching -- not a naive split on '_', since several
    column names (e.g. 'vehicle_speed', 'pedestrian_distance') already
    contain underscores themselves."""
    importances = fitted_pipeline.named_steps["classifier"].feature_importances_
    expanded_names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()

    original_cols = sorted(CATEGORICAL_COLS + NUMERIC_COLS, key=len, reverse=True)
    agg = {c: 0.0 for c in original_cols}
    for name, imp in zip(expanded_names, importances):
        match = next(c for c in original_cols if name == c or name.startswith(c + "_"))
        agg[match] += imp

    result = (
        pd.DataFrame({"variable": agg.keys(), "importance": agg.values()})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return result


def check_fold_importance_stability(X, y):
    """Refits one model per CV fold and returns each fold's top-4 features,
    for the validation checkpoint: are the top features stable across
    folds, or is the final full-data ranking an artifact of which rows
    happened to be in it?"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_top_features = []
    for train_idx, _ in skf.split(X, y):
        pipeline = build_pipeline()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        importance = compute_feature_importance(pipeline)
        fold_top_features.append(list(importance["variable"].head(4)))
    return fold_top_features


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, label_source, video_id = load_dataset()
    print(f"Loaded {len(X)} rows, {len(CATEGORICAL_COLS + NUMERIC_COLS)} features, target={TARGET_COL}")

    baseline_report = run_baseline(X, y)
    predictions, all_report, text_report, fallback_report, stability_line = run_cross_validation(
        X, y, label_source, video_id
    )

    print("\n" + baseline_report)
    print("\n" + all_report)
    print("\n" + text_report)
    print("\n" + stability_line)

    fold_top_features = check_fold_importance_stability(X, y)
    print("\nTop-4 features by fold (stability check):")
    for i, feats in enumerate(fold_top_features):
        print(f"  fold {i + 1}: {feats}")

    # Final model: fit on ALL 303 rows -- this is what gets saved and used
    # by Phase 4. The CV fold models above were evaluation-only and are
    # discarded now; cross-validation's only job was to produce an honest
    # performance estimate before trusting this final fit.
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    importance = compute_feature_importance(final_pipeline)
    print("\nFeature importance (final model, fit on all 303 rows):")
    print(importance.to_string(index=False))

    joblib.dump(final_pipeline, OUTPUT_DIR / "classifier.joblib")
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "cv_predictions.csv", index=False)

    with open(OUTPUT_DIR / "evaluation_report.txt", "w") as f:
        f.write(baseline_report + "\n\n")
        f.write(stability_line + "\n\n")
        f.write(all_report + "\n\n")
        f.write(text_report + "\n\n")
        f.write(fallback_report + "\n\n")
        f.write("Top-4 features by fold (stability check):\n")
        for i, feats in enumerate(fold_top_features):
            f.write(f"  fold {i + 1}: {feats}\n")

    print(f"\nSaved model + reports to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

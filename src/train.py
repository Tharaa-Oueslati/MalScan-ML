"""
train.py
========
Training pipeline for MalScan-ML.

Trains and compares:
  - Random Forest classifier
  - XGBoost classifier

Saves best model + scaler to models/ directory.
Generates evaluation plots to reports/figures/.

Usage:
    python src/train.py --data data/processed/features.csv
    python src/train.py --data data/processed/features.csv --model both --cv 5
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import click
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from evaluate import (
    plot_roc_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    print_classification_report,
)
from utils import load_features, get_feature_columns, create_sample_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/figures")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_random_forest() -> Pipeline:
    """Random Forest pipeline with preprocessing."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",  # Handle imbalanced datasets
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_xgboost() -> XGBClassifier:
    """XGBoost classifier (handles its own scaling internally)."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=1,          # Adjust if class-imbalanced
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    feature_names: list[str],
    model_type: str,
    cv_folds: int = 5,
) -> dict:
    """
    Train a single model, cross-validate, evaluate on test set.
    Returns dict with model, metrics, and predictions.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Training: {model_type.upper()}")
    logger.info(f"{'='*50}")

    if model_type == "random_forest":
        model = build_random_forest()
    elif model_type == "xgboost":
        model = build_xgboost()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info(f"CV AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final training
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        precision_score, recall_score,
    )

    metrics = {
        "model_name": model_type,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "cv_auc_mean": cv_scores.mean(),
        "cv_auc_std": cv_scores.std(),
    }

    logger.info(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Test F1:        {metrics['f1']:.4f}")
    logger.info(f"Test AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall:    {metrics['recall']:.4f}")

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_names": feature_names,
    }


def get_feature_importances(result: dict) -> pd.Series:
    """Extract feature importances from a trained model."""
    model = result["model"]
    names = result["feature_names"]

    # Handle Pipeline wrapper
    if hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
    else:
        clf = model

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        return pd.Series(importances, index=names).sort_values(ascending=False)
    return pd.Series(dtype=float)


@click.command()
@click.option("--data", "-d", required=True, help="Path to features CSV")
@click.option("--model", "-m", default="both",
              type=click.Choice(["random_forest", "xgboost", "both"]),
              help="Which model to train")
@click.option("--test-size", default=0.2, help="Test split fraction (default: 0.2)")
@click.option("--cv", default=5, help="Cross-validation folds (default: 5)")
@click.option("--save-plots", is_flag=True, default=True, help="Save evaluation plots")
def main(data, model, test_size, cv, save_plots):
    """
    Train and evaluate malware detection classifiers.

    \b
    Example:
        python src/train.py --data data/processed/features.csv --model both --cv 5
    """
    # ── Load data ──────────────────────────────────
    data_path = Path(data)

    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Generating synthetic sample dataset for demonstration...")
        df = create_sample_dataset(n_samples=2000)
        data_path = Path("data/processed/features_sample.csv")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Sample dataset saved to {data_path}")
    else:
        df = pd.read_csv(data_path)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['label'].value_counts().to_string()}")

    # ── Feature prep ───────────────────────────────
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["label"].values

    logger.info(f"Feature dimensions: {X.shape[1]}")

    # ── Train/test split ───────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ── Training ───────────────────────────────────
    results = {}
    model_types = ["random_forest", "xgboost"] if model == "both" else [model]

    for mt in model_types:
        result = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            feature_names=feature_cols,
            model_type=mt,
            cv_folds=cv,
        )
        results[mt] = result

        # Print full classification report
        print_classification_report(y_test, result["y_pred"], mt)

    # ── Save models ────────────────────────────────
    for mt, result in results.items():
        model_path = MODELS_DIR / f"{mt}_model.pkl"
        joblib.dump(result["model"], model_path)
        logger.info(f"Saved model → {model_path}")

    # ── Save metrics summary ───────────────────────
    metrics_df = pd.DataFrame([r["metrics"] for r in results.values()])
    metrics_path = REPORTS_DIR / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics summary → {metrics_path}")

    # ── Plots ──────────────────────────────────────
    if save_plots:
        logger.info("Generating evaluation plots...")

        # ROC curves (all models on same plot)
        roc_data = {
            mt: (r["y_proba"], r["metrics"]["auc_roc"])
            for mt, r in results.items()
        }
        plot_roc_curves(y_test, roc_data, save_path=REPORTS_DIR / "roc_curves.png")

        # Confusion matrices
        for mt, result in results.items():
            plot_confusion_matrix(
                y_test, result["y_pred"],
                title=f"Confusion Matrix — {mt.replace('_', ' ').title()}",
                save_path=REPORTS_DIR / f"confusion_matrix_{mt}.png",
            )

        # Feature importances
        for mt, result in results.items():
            importances = get_feature_importances(result)
            if not importances.empty:
                plot_feature_importance(
                    importances.head(20),
                    title=f"Top 20 Feature Importances — {mt.replace('_', ' ').title()}",
                    save_path=REPORTS_DIR / f"feature_importance_{mt}.png",
                )

        logger.info(f"Plots saved to {REPORTS_DIR}/")

    # ── Final comparison table ──────────────────────
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    print(metrics_df.to_string(index=False))

    # ── Identify best model ─────────────────────────
    best_model_name = metrics_df.loc[metrics_df["auc_roc"].idxmax(), "model_name"]
    best_auc = metrics_df["auc_roc"].max()
    logger.info(f"\n✅ Best model: {best_model_name} (AUC-ROC: {best_auc:.4f})")

    # Save best model separately for easy loading by scan.py
    best_model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(results[best_model_name]["model"], best_model_path)
    joblib.dump(best_model_name, MODELS_DIR / "best_model_name.pkl")
    logger.info(f"Best model saved → {best_model_path}")


if __name__ == "__main__":
    main()

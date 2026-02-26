"""
enzyme_classifier.py
====================
Trains and evaluates an XGBoost classifier for 7-class enzyme
classification, with built-in handling for class imbalance,
stratified cross-validated hyperparameter search, and evaluation.

Usage
-----
  from enzyme_classifier import EnzymeClassifier

  clf = EnzymeClassifier(features_csv="data/features.csv")
  clf.load_data()
  clf.search_hyperparameters(n_iter=50)   # randomised CV search
  clf.train_best()                        # refit best params on full train set
  clf.evaluate()                          # classification report + metrics
  clf.save_model("models/best_xgb.json")

Dependencies: xgboost, scikit-learn, pandas, numpy
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import uniform, randint

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb


class EnzymeClassifier:
    """
    XGBoost-based enzyme classifier with class-imbalance handling
    and randomised hyperparameter search.

    Parameters
    ----------
    features_csv : str or Path
        Path to the CSV produced by ProteinFeatureExtractor.
    test_size : float
        Fraction held out for evaluation (default 0.2).
    random_state : int
        Seed for reproducibility.
    """

    CLASS_NAMES = [
        "Not enzyme",        # 0
        "Oxidoreductase",    # 1
        "Transferase",       # 2
        "Hydrolase",         # 3
        "Lyase",             # 4
        "Isomerase",         # 5
        "Ligase",            # 6
    ]

    def __init__(
        self,
        features_csv: str | Path,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.features_csv = Path(features_csv)
        self.test_size = test_size
        self.random_state = random_state

        # Populated by load_data()
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.feature_names: list[str] = []

        # Populated by search / train
        self.best_params: dict = {}
        self.model: xgb.XGBClassifier | None = None
        self.search_results: pd.DataFrame | None = None

    # ── Data loading ───────────────────────────────────────────────────────

    def load_data(self) -> None:
        """Load features CSV, split into stratified train/test sets."""
        df = pd.read_csv(self.features_csv)
        print(f"Loaded {len(df)} samples × {len(df.columns)} columns")

        y = df["label"].values
        X = df.drop(columns=["label"])
        self.feature_names = list(X.columns)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X.values, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        print(f"Train: {len(self.X_train)}  |  Test: {len(self.X_test)}")
        self._print_class_distribution("Train", self.y_train)
        self._print_class_distribution("Test", self.y_test)

    # ── Hyperparameter search ──────────────────────────────────────────────

    def search_hyperparameters(
        self,
        n_iter: int = 50,
        cv_folds: int = 5,
        scoring: str = "f1_macro",
        verbose: int = 1,
    ) -> dict:
        """
        Randomised cross-validated hyperparameter search.

        Parameters
        ----------
        n_iter : int
            Number of random parameter combinations to try.
        cv_folds : int
            Number of stratified CV folds.
        scoring : str
            Metric to optimise. 'f1_macro' recommended for imbalanced data.
        verbose : int
            Verbosity level for RandomizedSearchCV.

        Returns
        -------
        dict : best hyperparameters found.
        """
        self._check_loaded()

        # Compute sample weights to handle class imbalance
        sample_weights = compute_sample_weight("balanced", self.y_train)

        param_distributions = {
            "n_estimators": randint(100, 800),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.29),      # [0.01, 0.30]
            "subsample": uniform(0.6, 0.4),             # [0.6, 1.0]
            "colsample_bytree": uniform(0.4, 0.6),      # [0.4, 1.0]
            "min_child_weight": randint(1, 10),
            "gamma": uniform(0, 0.5),
            "reg_alpha": uniform(0, 1.0),               # L1 regularisation
            "reg_lambda": uniform(0.5, 2.0),            # L2 regularisation
        }

        base_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            use_label_encoder=False,
            tree_method="hist",
            random_state=self.random_state,
        )

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=verbose,
            refit=False,            # we refit manually with sample weights
        )

        print(f"\nSearching {n_iter} configurations "
              f"({cv_folds}-fold CV, scoring={scoring})...\n")

        search.fit(self.X_train, self.y_train, sample_weight=sample_weights)

        self.best_params = search.best_params_
        self.search_results = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score"
        )

        print(f"\nBest CV {scoring}: {search.best_score_:.4f}")
        print(f"Best params: {json.dumps(self.best_params, indent=2, default=str)}")
        return self.best_params

    # ── Training ───────────────────────────────────────────────────────────

    def train_best(self, params: dict | None = None) -> None:
        """
        Train the final model on the full training set using the best
        (or supplied) hyperparameters with balanced sample weights.
        """
        self._check_loaded()
        params = params or self.best_params
        if not params:
            raise RuntimeError(
                "No parameters available. Run search_hyperparameters() first "
                "or pass params explicitly."
            )

        sample_weights = compute_sample_weight("balanced", self.y_train)

        self.model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            use_label_encoder=False,
            tree_method="hist",
            random_state=self.random_state,
        )

        self.model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        print("Model trained on full training set.")

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """
        Evaluate the trained model on the held-out test set.

        Returns
        -------
        dict : dictionary of evaluation metrics.
        """
        self._check_model()

        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)

        # Core metrics
        macro_f1 = f1_score(self.y_test, y_pred, average="macro")
        weighted_f1 = f1_score(self.y_test, y_pred, average="weighted")
        bal_acc = balanced_accuracy_score(self.y_test, y_pred)
        mcc = matthews_corrcoef(self.y_test, y_pred)

        print("\n" + "=" * 60)
        print("EVALUATION ON HELD-OUT TEST SET")
        print("=" * 60)
        print(f"  Macro F1          : {macro_f1:.4f}")
        print(f"  Weighted F1       : {weighted_f1:.4f}")
        print(f"  Balanced Accuracy : {bal_acc:.4f}")
        print(f"  MCC               : {mcc:.4f}")
        print()
        print(classification_report(
            self.y_test, y_pred,
            target_names=self.CLASS_NAMES,
            digits=4,
            zero_division=0,
        ))

        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        return {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": bal_acc,
            "mcc": mcc,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict classes and probabilities for new data.

        Returns
        -------
        (predictions, probabilities) : tuple of arrays
        """
        self._check_model()
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return preds, probs

    # ── Feature importance ─────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Return top-N features by XGBoost gain importance.
        """
        self._check_model()
        importance = self.model.get_booster().get_score(
            importance_type="gain"
        )
        df = (
            pd.DataFrame.from_dict(importance, orient="index", columns=["gain"])
            .sort_values("gain", ascending=False)
            .head(top_n)
        )
        print(f"\nTop {top_n} features by gain:")
        print(df.to_string())
        return df

    # ── Persistence ────────────────────────────────────────────────────────

    def save_model(self, path: str | Path) -> None:
        """Save the trained XGBoost model to JSON."""
        self._check_model()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out))
        print(f"Model saved → {out}")

    def load_model(self, path: str | Path) -> None:
        """Load a previously saved XGBoost model."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        print(f"Model loaded ← {path}")

    # ── Internals ──────────────────────────────────────────────────────────

    def _check_loaded(self):
        if self.X_train is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

    def _check_model(self):
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_best() first.")

    @staticmethod
    def _print_class_distribution(name: str, y: np.ndarray) -> None:
        unique, counts = np.unique(y, return_counts=True)
        parts = "  ".join(f"{int(u)}:{c}" for u, c in zip(unique, counts))
        print(f"  {name} distribution → {parts}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_DIR = SCRIPT_DIR.parent
    FEATURES_CSV = PROJECT_DIR / "data" / "features.csv"
    MODEL_PATH = PROJECT_DIR / "models" / "best_xgb.json"

    clf = EnzymeClassifier(features_csv=FEATURES_CSV)
    clf.load_data()
    clf.search_hyperparameters(n_iter=50, cv_folds=5)
    clf.train_best()
    metrics = clf.evaluate()
    clf.feature_importance(top_n=20)
    clf.save_model(MODEL_PATH)
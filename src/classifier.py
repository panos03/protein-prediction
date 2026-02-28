"""
classifier.py
=============
Trains and evaluates an XGBoost classifier for 7-class enzyme
classification, with built-in handling for class imbalance,
stratified cross-validated hyperparameter search, and evaluation.

Usage
-----
  clf = EnzymeClassifier(features_csv="data/features.csv")
  clf.load_data()
  clf.search_hyperparameters(n_iter=50)   # randomised CV search
  clf.train_best()                        # refit best params on full train set
  clf.evaluate()                          # classification report + metrics
  clf.save_model("models/best_xgb.json")

Dependencies: xgboost, scikit-learn, pandas, numpy
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import time

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
)
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb



# TODO look over


class EnzymeClassifier:

    CLASS_NAMES = [
        "Not enzyme",        # 0
        "Oxidoreductase",    # 1
        "Transferase",       # 2
        "Hydrolase",         # 3
        "Lyase",             # 4
        "Isomerase",         # 5
        "Ligase",            # 6
    ]

    def __init__(self, features_csv, test_size=0.2, random_state=42):
        self.features_csv = Path(features_csv)
        self.test_size = test_size
        self.random_state = random_state    # for reproducibility of train/test split and CV folds

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []

        self.best_params = {}
        self.model = None
        self.search_results = None


    def load_data(self):
        # Load features CSV and split into stratified train/test sets
        df = pd.read_csv(self.features_csv)
        print(f"Loaded {len(df)} samples x {len(df.columns)} columns")

        y = df["label"].values
        X = df.drop(columns=["label"])
        self.feature_names = list(X.columns)

        # stratified split into train/test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X.values, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        print(f"Train: {len(self.X_train)}  |  Test: {len(self.X_test)}")
        self._print_class_distribution("Train", self.y_train)
        self._print_class_distribution("Test",  self.y_test)


    def search_hyperparameters(self, n_iter=20, cv_folds=3, scoring="f1_macro", verbose=1):
        # Randomised cross-validated hyperparameter search over the training set.
        # Only the 5 most impactful parameters are tuned; the rest are fixed at
        # sensible defaults to keep the search to ~60 fits (20 iter × 3 folds).
        self._check_loaded()

        # balanced weights so minority EC classes aren't drowned out by class 0
        sample_weights = compute_sample_weight("balanced", self.y_train)

        # discrete lists instead of continuous distributions — faster and easier to report
        param_distributions = {
            "n_estimators":     [50, 200, 400],         # number of trees
            "max_depth":        [4, 6, 8],              # tree depth — controls model complexity
            "learning_rate":    [0.05, 0.1, 0.2],       # shrinkage — lower = slower but more robust
            "subsample":        [0.7, 0.8, 1.0],        # fraction of training rows used per tree
            "colsample_bytree": [0.5, 0.7, 1.0],        # fraction of features used per tree
        }

        base_model = xgb.XGBClassifier(
            objective="multi:softprob",     # outputs class probabilities for all 7 classes
            num_class=7,
            eval_metric="mlogloss",
            tree_method="hist",             # histogram-based split finding — faster than exact
            min_child_weight=1,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            random_state=self.random_state,
        )

        # stratified so each fold has the same class ratio as the full training set
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,                # f1_macro — treats all 7 classes equally regardless of size
            cv=cv,
            random_state=self.random_state,
            n_jobs=2,
            verbose=verbose,
            refit=False,                    # we refit manually with sample weights in train_best()
            return_train_score=False,       # only need CV score, not train score
        )

        print(f"\nSearching {n_iter} configurations ({cv_folds}-fold CV, scoring={scoring})...\n")
        search.fit(self.X_train, self.y_train, sample_weight=sample_weights)

        self.best_params = search.best_params_

        # keep a clean results table: one row per configuration, sorted best → worst
        param_cols = [f"param_{p}" for p in param_distributions]
        self.search_results = (
            pd.DataFrame(search.cv_results_)
            [param_cols + ["mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]]
            .rename(columns=lambda c: c.replace("param_", ""))
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )

        print(f"\nBest CV {scoring}: {search.best_score_:.4f}")
        print(f"Best params: {json.dumps(self.best_params, indent=2, default=str)}")
        print(f"\nAll configurations (best → worst):")
        print(self.search_results.to_string(index=False))

        return self.best_params, self.search_results


    def train_best(self, params=None):
        # Train final model on the full training set using best (or supplied) params
        self._check_loaded()
        params = params or self.best_params
        if not params:
            raise RuntimeError(
                "No parameters available. Run search_hyperparameters() first "
                "or pass params explicitly."
            )

        # balanced weights — same upweighting of minority EC classes as in the search
        sample_weights = compute_sample_weight("balanced", self.y_train)

        self.model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.random_state,
        )

        self.model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        print("Model trained on full training set.")


    def evaluate(self):
        # Evaluate trained model on the held-out test set, print full report
        self._check_model()

        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)

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
            "macro_f1":          macro_f1,
            "weighted_f1":       weighted_f1,
            "balanced_accuracy": bal_acc,
            "mcc":               mcc,
            "confusion_matrix":  cm,
            "y_pred":            y_pred,
            "y_prob":            y_prob,
        }


    def predict(self, X):   # TODO return class label along with High/Medium/Low confidence -> explain how this conversion is made? new blind predict method?
        # Return (predicted classes, class probabilities) for new data
        self._check_model()
        return self.model.predict(X), self.model.predict_proba(X)


    def feature_importance(self, top_n=20):
        # Overall feature importance: aggregate gain across all trees and all classes
        self._check_model()
        importance = self.model.get_booster().get_score(importance_type="gain")
        df = (
            pd.DataFrame.from_dict(importance, orient="index", columns=["gain"])
            .sort_values("gain", ascending=False)
            .head(top_n)
        )
        print(f"\nTop {top_n} features by gain (overall):")
        print(df.to_string())
        return df


    def feature_importance_per_class(self, top_n=10):
        # Per-class feature importance by partitioning trees by their class assignment.
        # In XGBoost multi:softprob, one tree is grown per class per boosting round,
        # cycling as: tree 0 → class 0, tree 1 → class 1, ..., tree 6 → class 6,
        # tree 7 → class 0, tree 8 → class 1, ... so tree i belongs to class (i % 7).
        # Summing gain over all split nodes for each class gives class-specific importance.
        self._check_model()

        trees_df = self.model.get_booster().trees_to_dataframe()

        # leaf nodes have Feature == "Leaf" — exclude them, only keep split nodes
        splits = trees_df[trees_df["Feature"] != "Leaf"].copy()
        splits["class_idx"] = splits["Tree"] % len(self.CLASS_NAMES)

        # build a combined DataFrame: rows = features, columns = class names
        per_class = {}
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_splits = splits[splits["class_idx"] == class_idx]
            per_class[class_name] = (
                class_splits.groupby("Feature")["Gain"]
                .sum()
                .sort_values(ascending=False)
            )

        combined = pd.DataFrame(per_class).fillna(0)

        # print top-N for each class
        for class_name in self.CLASS_NAMES:
            top = combined[class_name].sort_values(ascending=False).head(top_n)
            print(f"\nTop {top_n} features for {class_name}:")
            print(top.to_string())

        return combined


    def save_model(self, path):
        # Save trained XGBoost model to JSON
        self._check_model()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out))
        print(f"Model saved → {out}")


    def load_model(self, path):
        # Load a previously saved XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        print(f"Model loaded ← {path}")


    def _check_loaded(self):
        if self.X_train is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")


    def _check_model(self):
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_best() first.")


    @staticmethod
    def _print_class_distribution(name, y):     # for visualising class imbalance and stratification
        unique, counts = np.unique(y, return_counts=True)
        parts = "  ".join(f"{int(u)}:{c}" for u, c in zip(unique, counts))
        print(f"  {name} distribution → {parts}")

    
    @staticmethod   # TODO use
    def confidence_label(prob):     # confidence level thresholds
        if prob >= 0.8:
            return "High"
        elif prob >= 0.5:
            return "Medium"
        else:
            return "Low"


def save_df_to_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    print(f"Saved → {path}")


def log_time(message):
    raw_time = time.time()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(raw_time))
    print(f"\n    [{formatted_time}] {message}\n")



if __name__ == "__main__":

    SCRIPT_DIR   = Path(__file__).parent
    PROJECT_DIR  = SCRIPT_DIR.parent
    FEATURES_CSV = PROJECT_DIR / "data" / "features.csv"
    MODEL_PATH   = PROJECT_DIR / "models" / "best_xgb.json"
    RESULTS_DIR  = PROJECT_DIR / "results"

    log_time("Starting pipeline")

    clf = EnzymeClassifier(features_csv=FEATURES_CSV)

    # ---------------- LOAD DATA ----------------
    clf.load_data()
    log_time("Data loaded")

    # ---------------- HYPERPARAM SEARCH ----------------
    best_params, search_results = clf.search_hyperparameters(n_iter=20, cv_folds=3)
    log_time("Hyperparameter search complete")

    save_df_to_csv(
        search_results,
        RESULTS_DIR / "hyperparameter_search_results.csv"
    )
    log_time("Hyperparameter search results saved")

    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    log_time("Best parameters saved")

    # ---------------- TRAIN ----------------
    clf.train_best()
    log_time("Model training complete")

    # ---------------- EVALUATION ----------------
    eval_results = clf.evaluate()
    log_time("Evaluation complete")

    metrics_df = pd.DataFrame([{
        "macro_f1": eval_results["macro_f1"],
        "weighted_f1": eval_results["weighted_f1"],
        "balanced_accuracy": eval_results["balanced_accuracy"],
        "mcc": eval_results["mcc"],
    }])

    save_df_to_csv(metrics_df, RESULTS_DIR / "evaluation_metrics.csv")
    log_time("Evaluation metrics saved")

    cm_df = pd.DataFrame(
        eval_results["confusion_matrix"],
        index=EnzymeClassifier.CLASS_NAMES,
        columns=EnzymeClassifier.CLASS_NAMES,
    )

    save_df_to_csv(cm_df, RESULTS_DIR / "confusion_matrix.csv")
    log_time("Confusion matrix saved")

    preds_df = pd.DataFrame({
        "y_true": clf.y_test,
        "y_pred": eval_results["y_pred"],
    })

    save_df_to_csv(preds_df, RESULTS_DIR / "predictions.csv")
    log_time("Predictions saved")

    # ---------------- FEATURE IMPORTANCE ----------------
    fi_df = clf.feature_importance(top_n=20)
    save_df_to_csv(fi_df, RESULTS_DIR / "feature_importance_overall.csv")
    log_time("Overall feature importance saved")

    fi_per_class = clf.feature_importance_per_class(top_n=10)
    save_df_to_csv(fi_per_class, RESULTS_DIR / "feature_importance_per_class.csv")
    log_time("Per-class feature importance saved")

    # ---------------- SAVE MODEL ----------------
    clf.save_model(MODEL_PATH)
    log_time("Model saved")

    log_time("Pipeline finished")

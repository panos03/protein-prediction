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

  coarse_params = { "learning_rate": [...], "max_depth": [...], ... }
  best_params, coarse_results = clf.search_hyperparameters(coarse_params)

  refined_params = { ... }   # narrowed manually based on coarse results
  best_params, refined_results = clf.search_hyperparameters(refined_params)

  clf.train_best()            # early stopping finds optimal n_estimators
  clf.evaluate()
  clf.save_model("models/best_xgb.json")

Dependencies: xgboost, scikit-learn, pandas, numpy
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import time

from sklearn.experimental import enable_halving_search_cv   # must be imported before HalvingRandomSearchCV
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    HalvingRandomSearchCV,
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

        self.best_params = {}
        self.model = None


    def load_data(self):
        # Load features CSV and split into stratified train/test sets
        df = pd.read_csv(self.features_csv)
        print(f"Loaded {len(df)} samples x {len(df.columns)} columns")

        # prepare data
        y = df["label"].values
        X = df.drop(columns=["label"])      # pass X as a DataFrame (not .values) so XGBoost retains real column names

        # stratified split — test set keeps the natural class distribution for realistic evaluation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        # Undersample class 0 for training set only - huge ~100:1 imbalance between class 0 and class 6
        # Note: we don't touch the test set to keep it representative of real-world distribution for evaluation
        train_df = pd.DataFrame(self.X_train)
        train_df["label"] = self.y_train
        class_0 = train_df[train_df["label"] == 0].sample(n=3000, random_state=self.random_state)
        train_df = pd.concat([train_df[train_df["label"] != 0], class_0]).sample(frac=1, random_state=self.random_state)
        self.X_train = train_df.drop(columns=["label"])
        self.y_train = train_df["label"].values

        print(f"Train: {len(self.X_train)}  |  Test: {len(self.X_test)}")
        self._print_class_distribution("Train", self.y_train)
        self._print_class_distribution("Test",  self.y_test)


    def search_hyperparameters(self, param_distributions, cv_folds=3, scoring="f1_macro", verbose=1):
        # Hyperparameter search using HalvingRandomSearchCV over the supplied param_distributions.
        # Call once with wide coarse ranges, then again with narrow refined ranges.
        # n_estimators is NOT searched — it is found by early stopping in train_best().
        # HalvingRandomSearchCV starts with many configs on a small sample budget,
        # eliminates the worst each round (factor=3 keeps top 1/3), and converges cheaply.
        self._check_loaded()

        # balanced weights so minority EC classes aren't drowned out by class 0
        sample_weights = compute_sample_weight("balanced", self.y_train)

        # n_estimators fixed at 300 during search — early stopping will tune it in train_best()
        base_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            tree_method="hist",             # histogram-based split finding — faster than exact
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
        )

        # stratified so each fold has the same class ratio as the full training set
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        search = HalvingRandomSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            factor=3,                       # keep top 1/3 of configs each halving round
            scoring=scoring,                # f1_macro — treats all 7 classes equally regardless of size
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=verbose,
            refit=False,                    # we refit manually with sample weights in train_best()
        )

        print(f"\nSearching ({cv_folds}-fold CV, scoring={scoring})...")
        search.fit(self.X_train, self.y_train, sample_weight=sample_weights)

        self.best_params = search.best_params_

        # clean results table: one row per config, with halving iteration and sample budget columns
        results = self._format_results(search.cv_results_, param_distributions)

        print(f"\nBest CV {scoring}: {search.best_score_:.4f}")
        print(f"Best params: {json.dumps(self.best_params, indent=2, default=str)}")
        print(f"\nAll configurations (best → worst):")
        print(results.to_string(index=False))

        return self.best_params, results


    def train_best(self, params=None):
        # Train final model on the full training set using best (or supplied) params.
        # Uses early stopping on a small validation split to find the optimal n_estimators,
        # then refits on the full training set with that number of trees.
        self._check_loaded()
        params = params or self.best_params
        if not params:
            raise RuntimeError(
                "No parameters available. Run search_hyperparameters() first "
                "or pass params explicitly."
            )

        # hold out 10% of training data as a validation set for early stopping only
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.1, stratify=self.y_train, random_state=self.random_state
        )
        weights_tr = compute_sample_weight("balanced", y_tr)

        # probe fit: large n_estimators, early stopping decides the optimal number
        probe = xgb.XGBClassifier(
            **params,
            n_estimators=2000,
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,      # stop if no improvement for 50 consecutive trees
            random_state=self.random_state,
        )
        probe.fit(X_tr, y_tr, sample_weight=weights_tr, eval_set=[(X_val, y_val)], verbose=False)
        best_n = probe.best_iteration + 1
        print(f"Early stopping: optimal n_estimators = {best_n}")

        # final fit: refit on full training set with the found n_estimators
        sample_weights = compute_sample_weight("balanced", self.y_train)
        self.model = xgb.XGBClassifier(
            **params,
            n_estimators=best_n,
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
        )
        self.model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        print(f"Model trained on full training set ({best_n} trees).")


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


    def _format_results(self, cv_results, param_distributions):
        # clean results table: param columns + halving iteration/budget cols + scores
        param_cols = [f"param_{p}" for p in param_distributions]
        halving_cols = ["iter", "n_resources"]      # which halving round and how many samples were used
        score_cols = ["mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]
        all_cols = param_cols + halving_cols + score_cols
        available = [c for c in all_cols if c in cv_results]   # only keep cols that exist
        return (
            pd.DataFrame(cv_results)[available]
            .rename(columns=lambda c: c.replace("param_", ""))
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )


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

    # ---------------- COARSE SEARCH ----------------
    coarse_params = {
        "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
        "max_depth":        [3, 5, 7, 9],
        "min_child_weight": [1, 5, 10, 20],
        "reg_lambda":       [1, 5, 10, 20],
        "reg_alpha":        [0, 0.1, 1, 5],
    }
    coarse_best, coarse_results = clf.search_hyperparameters(coarse_params, cv_folds=3)
    log_time("Coarse search complete")

    save_df_to_csv(coarse_results, RESULTS_DIR / "search_coarse.csv")
    log_time("Coarse search results saved")

    # # ---------------- REFINED SEARCH ---------------- TODO
    # # Manually narrow ranges around the coarse best values (inspect search_coarse.csv first).
    # # Edit these based on coarse_best above before running the refined search.
    # refined_params = {
    #     "learning_rate":    [coarse_best["learning_rate"]],    # placeholder — narrow manually
    #     "max_depth":        [coarse_best["max_depth"]],
    #     "min_child_weight": [coarse_best["min_child_weight"]],
    #     "reg_lambda":       [coarse_best["reg_lambda"]],
    #     "reg_alpha":        [coarse_best["reg_alpha"]],
    # }
    # refined_best, refined_results = clf.search_hyperparameters(refined_params, cv_folds=3)
    # log_time("Refined search complete")

    # save_df_to_csv(refined_results, RESULTS_DIR / "search_refined.csv")
    # log_time("Refined search results saved")

    # with open(RESULTS_DIR / "best_params.json", "w") as f:
    #     json.dump(refined_best, f, indent=2)
    # log_time("Best parameters saved")

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

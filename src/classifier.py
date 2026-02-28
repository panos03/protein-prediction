"""
classifier.py
=============
Two-stage XGBoost classifier for 7-class enzyme classification.

Stage 1 (binary):     enzyme vs non-enzyme
Stage 2 (multiclass): EC 1-6 classification (only for predicted enzymes)

This decomposition avoids the extreme class imbalance problem where
~30k non-enzymes overwhelm ~200-2000 enzyme samples per class.

Usage
-----
  clf = EnzymeClassifier(features_csv="data/features.csv")
  clf.load_data()
  clf.search_hyperparameters(stage=1, param_distributions={...})
  clf.search_hyperparameters(stage=2, param_distributions={...})
  clf.train_best()
  clf.evaluate()

Dependencies: xgboost, scikit-learn, pandas, numpy, imbalanced-learn
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
from imblearn.over_sampling import SMOTE

import xgboost as xgb


# TODO look over. change american comments


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

    EC_NAMES = CLASS_NAMES[1:]  # just the 6 enzyme classes

    def __init__(self, features_csv, test_size=0.2, random_state=42):
        self.features_csv = Path(features_csv)
        self.test_size = test_size
        self.random_state = random_state

        # Full dataset splits (7-class labels)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []

        # Stage-specific training data (prepared in load_data)
        self.X_train_s1 = None      # all training samples
        self.y_train_s1 = None      # binary: 0 = not enzyme, 1 = enzyme
        self.X_train_s2 = None      # enzyme samples only
        self.y_train_s2 = None      # labels 1–6

        # Models and params
        self.model_s1 = None        # binary classifier
        self.model_s2 = None        # 6-class enzyme classifier
        self.best_params_s1 = {}
        self.best_params_s2 = {}

        self.base_params_s1 = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",              # histogram-based split finding — faster than exact
            "subsample": 0.8,                   # row subsampling for regularization and speed
            "colsample_bytree": 0.8,            # feature subsampling for regularization and speed
            "random_state": self.random_state,
        }
        self.base_params_s2 = {
            "objective": "multi:softprob",
            "num_class": 6,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "subsample": 0.8,   
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
        }


    def load_data(self):
        df = pd.read_csv(self.features_csv)
        print(f"Loaded {len(df)} samples x {len(df.columns)} columns")

        y = df["label"].values
        X = df.drop(columns=["label"])      # pass X as a DataFrame (not .values) so XGBoost retains real column names
        self.feature_names = list(X.columns)

        # Stratified train/test split — test set untouched for realistic evaluation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        print(f"Train: {len(self.X_train)}  |  Test: {len(self.X_test)}")
        self._print_class_distribution("Train", self.y_train)
        self._print_class_distribution("Test",  self.y_test)

        # Prepare stage-specific training sets
        self._prepare_stages()

    def _prepare_stages(self):
        # --> Stage 1: binary (enzyme vs not enzyme)

        # change to binary labels
        self.y_train_s1 = (self.y_train > 0).astype(int)   # 0 = not enzyme, 1 = enzyme
        train_df = pd.DataFrame(self.X_train.values, columns=self.feature_names)
        train_df["binary_label"] = self.y_train_s1

        # Undersample class 0 to ~3× the largest enzyme class for balance
        n_enzyme = int(self.y_train_s1.sum())
        n_non_enzyme = len(self.y_train_s1) - n_enzyme
        n_class0_target = min(n_enzyme * 3, n_non_enzyme)
        class_0_samples = train_df[train_df["binary_label"] == 0].sample(
            n=n_class0_target, random_state=self.random_state
        )

        enzyme_samples = train_df[train_df["binary_label"] == 1]
        s1_df = pd.concat([class_0_samples, enzyme_samples]).sample(
            frac=1, random_state=self.random_state
        )
        self.X_train_s1 = s1_df.drop(columns=["binary_label"])
        self.y_train_s1 = s1_df["binary_label"].values

        print(f"\nStage 1 training: {len(self.X_train_s1)} samples")
        self._print_class_distribution("  S1", self.y_train_s1)

        # --> Stage 2: 6-class enzyme only
        enzyme_mask = self.y_train > 0
        self.X_train_s2 = pd.DataFrame(
            self.X_train.values[enzyme_mask], columns=self.feature_names
        )
        self.y_train_s2 = self.y_train[enzyme_mask] - 1     # map labels 1-6 to 0-5, because XGBoost multi:softprob expects 0-based class indices

        # SMOTE minority enzyme classes up to 500 samples
        self.X_train_s2, self.y_train_s2 = self._smote_minority(
            self.X_train_s2, self.y_train_s2, target=500
        )

        print(f"\nStage 2 training: {len(self.X_train_s2)} samples")
        self._print_class_distribution("  S2", self.y_train_s2)

    def _smote_minority(self, X, y, target=500):
        # SMOTE to bring minority classes up to target count, using interpolation
        feature_cols = list(X.columns)
        unique, counts = np.unique(y, return_counts=True)
        smote_strategy = {
            int(c): target
            for c, n in zip(unique, counts)
            if n < target
        }
        if smote_strategy:
            smote = SMOTE(sampling_strategy=smote_strategy, random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X, y)
            X = pd.DataFrame(X_res, columns=feature_cols)
            y = y_res
            print(f"  SMOTE applied: {smote_strategy}")

        return X, y


    def search_hyperparameters(self, stage, param_distributions, cv_folds=3, verbose=1):
        # Hyperparameter search using HalvingRandomSearchCV over the supplied param_distributions.
        # HalvingRandomSearchCV starts with many configs on a small sample budget,
        # eliminates the worst each round (factor=3 keeps top 1/3), and converges cheaply.
        self._check_loaded()

        if stage == 1:
            X, y = self.X_train_s1, self.y_train_s1
            scoring = "f1"          # binary F1 for stage 1
        elif stage == 2:
            X, y = self.X_train_s2, self.y_train_s2
            scoring = "f1_macro"    # treats all classes equally regardless of size
        else:
            raise ValueError("stage must be 1 or 2")

        sample_weights = compute_sample_weight("balanced", y)

        base_params = self.base_params_s1 if stage == 1 else self.base_params_s2

        base_model = xgb.XGBClassifier(**base_params)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        search = HalvingRandomSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            factor=3,                               # keep top 1/3 of configs each halving round
            min_resources=min(500, len(X) // 3),    # minimum samples per config — prevents tiny folds missing minority classes
            scoring=scoring,
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=verbose,
            refit=False,                            # we refit manually with sample weights in train_best()
        )

        print(f"\nStage {stage} search ({cv_folds}-fold CV, scoring={scoring})...")
        search.fit(X, y, sample_weight=sample_weights)

        best = search.best_params_
        results = self._format_results(search.cv_results_, param_distributions)

        if stage == 1:
            self.best_params_s1 = best
        else:
            self.best_params_s2 = best

        print(f"\nStage {stage} best CV {scoring}: {search.best_score_:.4f}")
        print(f"Best params: {json.dumps(best, indent=2, default=str)}")
        print(f"\nAll configurations (best → worst):")
        print(results.to_string(index=False))

        return best, results


    def train_best(self, params_s1=None, params_s2=None):

        self._check_loaded()
        params_s1 = params_s1 or self.best_params_s1
        params_s2 = params_s2 or self.best_params_s2

        if not params_s1 or not params_s2:
            # fall back to saved json files if available
            json_path_s1 = self.features_csv.parent.parent / "results" / "best_params_s1.json"
            json_path_s2 = self.features_csv.parent.parent / "results" / "best_params_s2.json"
            if json_path_s1.exists() and json_path_s1.stat().st_size > 0:
                with open(json_path_s1) as f:
                    params_s1 = json.load(f)
                print(f"Loaded best params for stage 1 from {json_path_s1}")
            if json_path_s2.exists() and json_path_s2.stat().st_size > 0:
                with open(json_path_s2) as f:
                    params_s2 = json.load(f)
                print(f"Loaded best params for stage 2 from {json_path_s2}")
            if not params_s1 or not params_s2:
                raise RuntimeError(
                    "Missing parameters for one or both stages. "
                    "Run search_hyperparameters() for stages 1 and 2 first."
                )

        # --> Train stage 1: binary
        weights_s1 = compute_sample_weight("balanced", self.y_train_s1)
        self.model_s1 = xgb.XGBClassifier(
            **params_s1,
            **self.base_params_s1,
        )
        self.model_s1.fit(self.X_train_s1, self.y_train_s1, sample_weight=weights_s1)
        print(f"Stage 1 trained ({params_s1.get('n_estimators', '?')} trees)")

        # --> Train stage 2: 6-class
        weights_s2 = compute_sample_weight("balanced", self.y_train_s2)
        self.model_s2 = xgb.XGBClassifier(
            **params_s2,
            **self.base_params_s2,
        )
        self.model_s2.fit(self.X_train_s2, self.y_train_s2, sample_weight=weights_s2)
        print(f"Stage 2 trained ({params_s2.get('n_estimators', '?')} trees)")


    def predict(self, X):   # TODO return class label along with High/Medium/Low confidence -> explain how this conversion is made? new blind predict method?
        # Two-stage prediction:
        #   1. Binary: enzyme or not?
        #   2. If enzyme: which EC class?

        # Returns (predictions, probabilities) where:
        #   - predictions: array of class labels 0-6
        #   - probabilities: array of shape (n, 7) with class probabilities

        self._check_model()

        n = len(X)
        predictions = np.zeros(n, dtype=int)
        probabilities = np.zeros((n, 7))

        # Stage 1: binary prediction
        s1_probs = self.model_s1.predict_proba(X)   # shape (n, 2): [P(not enzyme), P(enzyme)]
        s1_pred = (s1_probs[:, 1] >= 0.5).astype(int)   # P(enzyme) is the second col

        # Samples predicted as non-enzyme
        not_enzyme_mask = s1_pred == 0
        probabilities[not_enzyme_mask, 0] = s1_probs[not_enzyme_mask, 0]
        # spread remaining probability evenly across enzyme classes (uninformed. but won't be used)
        if not_enzyme_mask.any():
            enzyme_remainder = s1_probs[not_enzyme_mask, 1]
            probabilities[not_enzyme_mask, 1:] = enzyme_remainder[:, np.newaxis] / 6
        # note: predictions array not changed - as already initialised to 0

        # Stage 2: samples predicted as enzyme
        enzyme_mask = s1_pred == 1
        if enzyme_mask.any():
            X_enzyme = X[enzyme_mask] if hasattr(X, 'iloc') else X[enzyme_mask] # support both DataFrame and array inputs
            s2_probs = self.model_s2.predict_proba(X_enzyme)    # shape (m, 6): [P(EC1), ..., P(EC6)]
            s2_pred = self.model_s2.predict(X_enzyme) + 1       # remap labels 0-5 to 1-6

            predictions[enzyme_mask] = s2_pred

            # Combine probabilities: P(class k) = P(enzyme) * P(EC k | enzyme)
            p_enzyme = s1_probs[enzyme_mask, 1]
            probabilities[enzyme_mask, 0] = s1_probs[enzyme_mask, 0]
            probabilities[enzyme_mask, 1:] = s2_probs * p_enzyme[:, np.newaxis]

        return predictions, probabilities


    def evaluate(self):
        self._check_model()

        y_pred, y_prob = self.predict(self.X_test)

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

        # Stage 1 standalone metrics
        y_binary_true = (self.y_test > 0).astype(int)
        y_binary_pred = (y_pred > 0).astype(int)
        s1_f1 = f1_score(y_binary_true, y_binary_pred)
        print(f"\n  Stage 1 (binary enzyme detection):")
        print(f"    F1: {s1_f1:.4f}")
        print(classification_report(
            y_binary_true, y_binary_pred,
            target_names=["Not enzyme", "Enzyme"],
            digits=4,
        ))

        # Full 7-class report
        print(f"  Stage 1 + 2 (full 7-class):")
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


    def feature_importance(self, stage=2, top_n=20):
        model = self.model_s1 if stage == 1 else self.model_s2
        if model is None:
            raise RuntimeError(f"Stage {stage} model not trained.")

        importance = model.get_booster().get_score(importance_type="gain")
        df = (
            pd.DataFrame.from_dict(importance, orient="index", columns=["gain"])
            .sort_values("gain", ascending=False)
            .head(top_n)
        )
        print(f"\nTop {top_n} features by gain (stage {stage}):")
        print(df.to_string())
        return df


    def feature_importance_per_class(self, top_n=10):
        # For stage 2 only
        # TODO keep comment? :
        # Per-class feature importance by partitioning trees by their class assignment.
        # In XGBoost multi:softprob, one tree is grown per class per boosting round,
        # cycling as: tree 0 → class 0, tree 1 → class 1, ..., tree 5 → class 5,
        # tree 6 → class 0, tree 7 → class 1, ... so tree i belongs to class (i % 6).
        # Summing gain over all split nodes for each class gives class-specific importance.
        if self.model_s2 is None:
            raise RuntimeError("Stage 2 model not trained.")

        trees_df = self.model_s2.get_booster().trees_to_dataframe()

        # leaf nodes have Feature == "Leaf" — exclude them, only keep split nodes
        splits = trees_df[trees_df["Feature"] != "Leaf"].copy()
        splits["class_idx"] = splits["Tree"] % len(self.EC_NAMES)

        # build a combined DataFrame: rows = features, columns = class names
        per_class = {}
        for class_idx, class_name in enumerate(self.EC_NAMES):
            class_splits = splits[splits["class_idx"] == class_idx]
            per_class[class_name] = (
                class_splits.groupby("Feature")["Gain"]
                .sum()
                .sort_values(ascending=False)
            )

        combined = pd.DataFrame(per_class).fillna(0)

        # print top-N for each class
        for class_name in self.EC_NAMES:
            top = combined[class_name].sort_values(ascending=False).head(top_n)
            print(f"\nTop {top_n} features for {class_name}:")
            print(top.to_string())

        return combined


    def save_model(self, folder):
        s1_path = Path(folder) / "model_s1.xgb"
        s2_path = Path(folder) / "model_s2.xgb"
        self.model_s1.save_model(str(s1_path))
        self.model_s2.save_model(str(s2_path))
        print(f"Stage 1 saved → {s1_path}")
        print(f"Stage 2 saved → {s2_path}")


    def load_model(self, folder):
        s1_path = Path(folder) / "model_s1.xgb"
        s2_path = Path(folder) / "model_s2.xgb"
        self.model_s1 = xgb.XGBClassifier()
        self.model_s1.load_model(str(s1_path))
        self.model_s2 = xgb.XGBClassifier()
        self.model_s2.load_model(str(s2_path))
        print(f"Models loaded ← {s1_path}, {s2_path}")


    def _format_results(self, cv_results, param_distributions):
        param_cols = [f"param_{p}" for p in param_distributions]
        halving_cols = ["iter", "n_resources"]
        score_cols = ["mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]
        all_cols = param_cols + halving_cols + score_cols
        available = [c for c in all_cols if c in cv_results]
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
        if self.model_s1 is None or self.model_s2 is None:
            raise RuntimeError("Both stages must be trained. Call train_best() first.")

    @staticmethod
    def _print_class_distribution(name, y):
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
    MODEL_DIR   = PROJECT_DIR / "models"
    RESULTS_DIR  = PROJECT_DIR / "results"

    log_time("Starting pipeline")

    clf = EnzymeClassifier(features_csv=FEATURES_CSV)

    # ---------------- LOAD DATA ----------------
    clf.load_data()
    log_time("Data loaded")

    # ---------------- STAGE 1 SEARCH ----------------
    s1_params = {
        "n_estimators":     [100, 300, 500],
        "learning_rate":    [0.05, 0.1, 0.2],
        "max_depth":        [3, 5, 7],
        "min_child_weight": [1, 3, 5],
        "reg_lambda":       [0.5, 1, 5],
    }
    best_s1, results_s1 = clf.search_hyperparameters(stage=1, param_distributions=s1_params)
    log_time("Stage 1 search complete")
    save_df_to_csv(results_s1, RESULTS_DIR / "search_stage1.csv")
    with open(RESULTS_DIR / "best_params_s1.json", "w") as f:
        json.dump(best_s1, f, indent=2)

    # ---------------- STAGE 2 SEARCH ----------------
    s2_params = {
        "n_estimators":     [100, 300, 500],
        "learning_rate":    [0.05, 0.1, 0.2],
        "max_depth":        [3, 5, 7],
        "min_child_weight": [1, 3, 5],
        "reg_lambda":       [0.5, 1, 5],
    }
    best_s2, results_s2 = clf.search_hyperparameters(stage=2, param_distributions=s2_params)
    log_time("Stage 2 search complete")
    save_df_to_csv(results_s2, RESULTS_DIR / "search_stage2.csv")
    with open(RESULTS_DIR / "best_params_s2.json", "w") as f:
        json.dump(best_s2, f, indent=2)

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

    cm_df = pd.DataFrame(
        eval_results["confusion_matrix"],
        index=EnzymeClassifier.CLASS_NAMES,
        columns=EnzymeClassifier.CLASS_NAMES,
    )
    save_df_to_csv(cm_df, RESULTS_DIR / "confusion_matrix.csv")

    preds_df = pd.DataFrame({
        "y_true": clf.y_test,
        "y_pred": eval_results["y_pred"],
    })
    save_df_to_csv(preds_df, RESULTS_DIR / "predictions.csv")

    # ---------------- FEATURE IMPORTANCE ----------------
    fi_s1 = clf.feature_importance(stage=1, top_n=20)
    save_df_to_csv(fi_s1, RESULTS_DIR / "feature_importance_stage1.csv")

    fi_s2 = clf.feature_importance(stage=2, top_n=20)
    save_df_to_csv(fi_s2, RESULTS_DIR / "feature_importance_stage2.csv")

    fi_per_class = clf.feature_importance_per_class(top_n=10)
    save_df_to_csv(fi_per_class, RESULTS_DIR / "feature_importance_per_class.csv")

    # ---------------- SAVE MODEL ----------------
    clf.save_model(MODEL_DIR)
    log_time("Model saved")

    log_time("Pipeline finished")
    
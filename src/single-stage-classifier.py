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
from imblearn.over_sampling import SMOTE



class EnzymeClassifierSingleStage:

    CLASS_NAMES = [
        "Not enzyme",        # 0
        "Oxidoreductase",    # 1
        "Transferase",       # 2
        "Hydrolase",         # 3
        "Lyase",             # 4
        "Isomerase",         # 5
        "Ligase",            # 6
    ]

    def __init__(self, features_csv, results_dir, test_size=0.2, random_state=42):
        self.features_csv = Path(features_csv)
        self.results_dir = Path(results_dir)
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

        # stratified split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        # Note: we don't touch the test set to keep it representative of real-world distribution for evaluation
        self.X_train, self.y_train = self._handle_imbalance(self.X_train, self.y_train)

        print(f"Train: {len(self.X_train)}  |  Test: {len(self.X_test)}")
        self._print_class_distribution("Train", self.y_train)
        self._print_class_distribution("Test",  self.y_test)


    def search_hyperparameters(self, param_distributions, cv_folds=3, scoring="f1_macro", verbose=1):
        # Hyperparameter search using HalvingRandomSearchCV over the supplied param_distributions.
        # HalvingRandomSearchCV starts with many configs on a small sample budget,
        # eliminates the worst each round (factor=3 keeps top 1/3), and converges cheaply.
        self._check_loaded()

        # balanced weights so minority EC classes aren't drowned out by class 0
        sample_weights = compute_sample_weight("balanced", self.y_train)

        base_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            tree_method="hist",             # histogram-based split finding — faster than exact
            subsample=0.8,                  # row subsampling for regularization and speed
            colsample_bytree=0.8,           # feature subsampling for regularization and speed
            random_state=self.random_state,
        )

        # stratified so each fold has the same class ratio as the full training set
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        search = HalvingRandomSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            factor=3,                       # keep top 1/3 of configs each halving round
            min_resources=500,              # minimum samples per config — prevents tiny folds missing minority classes
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
        self._check_loaded()
        params = params or self.best_params
        if not params:
            # fall back to results/best_params.json if it exists and has content
            json_path = self.results_dir / "best_params.json"
            if json_path.exists() and json_path.stat().st_size > 0:
                with open(json_path) as f:
                    params = json.load(f)
                print(f"Loaded best params from {json_path}")
            else:
                raise RuntimeError("No parameters. Run search_hyperparameters() first.")

        sample_weights = compute_sample_weight("balanced", self.y_train)

        self.model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
        )
        self.model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        print(f"Model trained on full training set ({params.get('n_estimators', '?')} trees).")


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


    def _handle_imbalance(self, X, y):
        # Undersample class 0 - huge ~100:1 imbalance between class 0 and class 6
        df = pd.DataFrame(X)
        df["label"] = y
        class_0 = df[df["label"] == 0].sample(n=3000, random_state=self.random_state)
        df = pd.concat([df[df["label"] != 0], class_0]).sample(frac=1, random_state=self.random_state)
        X = df.drop(columns=["label"])
        y = df["label"].values

        # SMOTE on minority EC classes: generate synthetic samples for classes below smote_target
        smote_target = 500
        feature_cols = list(X.columns)
        unique, counts = np.unique(y, return_counts=True)
        smote_strategy = {
            int(c): smote_target
            for c, n in zip(unique, counts)
            if n < smote_target and c != 0      # don't oversample class 0
        }
        if smote_strategy:
            smote = SMOTE(sampling_strategy=smote_strategy, random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X, y)
            X = pd.DataFrame(X_res, columns=feature_cols)   # restore column names lost by SMOTE
            y = y_res
            print(f"SMOTE applied to classes: {smote_strategy}")

        return X, y


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
    MODEL_PATH   = PROJECT_DIR / "ablations" / "single-stage" / "models" / "single_stage_model.json"
    RESULTS_DIR  = PROJECT_DIR / "ablations" / "single-stage" / "results"

    log_time("Starting pipeline")

    clf = EnzymeClassifierSingleStage(features_csv=FEATURES_CSV, results_dir=RESULTS_DIR)

    # ---------------- LOAD DATA ----------------
    clf.load_data()
    log_time("Data loaded")

    # ---------------- HYPERPARAMETER SEARCH ----------------
    params = {
    "n_estimators":     [100, 300, 500],
    "learning_rate":    [0.05, 0.1, 0.2],
    "max_depth":        [3, 5, 7],
    "min_child_weight": [1, 3, 5],
    "reg_lambda":       [0.5, 1, 5],
    }
    search_best, search_results = clf.search_hyperparameters(params, cv_folds=3)
    log_time("Search complete")

    save_df_to_csv(search_results, RESULTS_DIR / "search_results.csv")
    log_time("Search results saved")

    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump(search_best, f, indent=2)
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
        index=EnzymeClassifierSingleStage.CLASS_NAMES,
        columns=EnzymeClassifierSingleStage.CLASS_NAMES,
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

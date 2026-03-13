import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


def top_unique_hyperparams(csv_path, n=10):
    # Reads hyperparameter search results CSV and returns top-n unique
    # hyperparameter configurations by mean_test_score

    # NOTE: For configs that appear across multiple iterations, 
    # only the latest iteration's result is kept before ranking.

    df = pd.read_csv(csv_path, index_col=0)

    hp_cols = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "reg_lambda"]

    # For each unique hyperparameter combo, keep only the latest iteration
    df_deduplicated = (
        df.sort_values("iter", ascending=False)
          .drop_duplicates(subset=hp_cols, keep="first")
    )

    # Sort by mean score descending and take top-n
    top = (
        df_deduplicated.sort_values("mean_test_score", ascending=False)
                       .head(n).reset_index(drop=True)
    )
    top.index += 1  # rank starts at 1 not 0

    display_cols = hp_cols + ["iter", "mean_test_score", "std_test_score"]
    result = top[display_cols].copy()

    # Pretty-print
    print(f"Top {n} unique hyperparameter configurations from: {csv_path}\n")
    fmt = result.copy()
    fmt["score"] = (
        fmt["mean_test_score"].map("{:.3f}".format)
        + " ± "
        + fmt["std_test_score"].map("{:.3f}".format)
    )
    print(fmt.drop(columns=["mean_test_score", "std_test_score"]).to_string(index=True))

    return result


CLASS_NAMES = ["Not enzyme", "Oxidoreductase", "Transferase", "Hydrolase", "Lyase", "Isomerase", "Ligase"]


def classification_reports(predictions_csv_path):
    # Reads predictions CSV (y_true, y_pred) and prints/returns both
    # the binary Stage 1 report and the full 7-class Stage 1+2 report.

    preds = pd.read_csv(predictions_csv_path, index_col=0)
    y_true = preds["y_true"].values
    y_pred = preds["y_pred"].values

    y_binary_true = (y_true > 0).astype(int)
    y_binary_pred = (y_pred > 0).astype(int)

    report_s1 = classification_report(
        y_binary_true, y_binary_pred,
        target_names=["Not enzyme", "Enzyme"],
        digits=4,
        output_dict=True,
    )
    report_s2 = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    print(f"Stage 1 (binary enzyme detection) — {predictions_csv_path}")
    print(classification_report(y_binary_true, y_binary_pred, target_names=["Not enzyme", "Enzyme"], digits=4))
    print(f"Stage 1+2 (full 7-class) — {predictions_csv_path}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    df_s1 = pd.DataFrame(report_s1).T
    df_s2 = pd.DataFrame(report_s2).T
    return df_s1, df_s2


def plot_confusion_matrix(csv_path, save_path=None, diagonal_zero=False, inter_enzyme=False):
    
    cm = pd.read_csv(csv_path, index_col=0)
    labels = cm.columns.tolist()

    data = cm.values.astype(float)
    if diagonal_zero:       # NOTE: Diagonal zeroed out to highlight misclassifications only
        np.fill_diagonal(data, 0)
    if inter_enzyme:      # NOTE: Inter-enzyme misclassifications only (remove non-enzyme row+col)
        non_enzyme_idx = labels.index("Not enzyme")
        data = np.delete(data, non_enzyme_idx, axis=0)
        data = np.delete(data, non_enzyme_idx, axis=1)
        labels.pop(non_enzyme_idx)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    diag_zero_str = " (diagonal zeroed)" if diagonal_zero else ""
    inter_enzyme_str = " (inter-enzyme only)" if inter_enzyme else ""
    ax.set_title(f"Confusion Matrix{diag_zero_str}{inter_enzyme_str}", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    top_unique_hyperparams("results/search_stage1.csv")
    print("\n" + "="*80 + "\n")
    top_unique_hyperparams("results/search_stage2.csv")

    # plot_confusion_matrix("results/confusion_matrix.csv")
    # plot_confusion_matrix("results/confusion_matrix.csv", diagonal_zero=True, inter_enzyme=True)

    print("\n" + "="*80 + "\n")
    classification_reports("results/predictions.csv")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    plot_confusion_matrix("results/confusion_matrix.csv")
    # plot_confusion_matrix("results/confusion_matrix.csv", diagonal_zero=True, inter_enzyme=True)

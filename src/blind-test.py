"""
Required submission format:

    SEQ01 1 Confidence High
    SEQ02 0 Confidence Medium
    ...

REMINDER: Confidence is derived from the probability of the predicted class
  - Class 0 (non-enzyme): P(not enzyme) from Stage 1 binary classifier
  - Classes 1-6 (enzyme): P(enzyme) * P(EC k | enzyme)

Thresholds (see EnzymeClassifier.confidence_label):
  High >= 0.8      Medium >= 0.5      Low < 0.5
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from feature_extractor import ProteinFeatureExtractor
from classifier import EnzymeClassifier

BLIND_FASTA      = PROJECT_DIR / "blind-test" / "blind_ec_test.fasta"
FEATURES_CSV     = PROJECT_DIR / "data" / "features.csv"
MODEL_DIR        = PROJECT_DIR / "models"
BLIND_FEAT_CSV   = PROJECT_DIR / "blind-test" / "blind_features.csv"
OUTPUT_TXT       = PROJECT_DIR / "blind-test" / "blind_predictions.txt"



def extract_features():
    # Don't re-run slow extraction if features already exist
    if BLIND_FEAT_CSV.exists():
        print(f"Loading cached features <- {BLIND_FEAT_CSV}")
        return pd.read_csv(BLIND_FEAT_CSV)

    extractor = ProteinFeatureExtractor(fasta_dir="", verbose=False)    # NOTE: fasta_dir not used since we're passing a single FASTA path to extract_for_blind_test()
    df = extractor.extract_for_blind_test(BLIND_FASTA)

    BLIND_FEAT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BLIND_FEAT_CSV, index=False)
    print(f"Features saved -> {BLIND_FEAT_CSV}")
    return df
    


if __name__ == "__main__":
    
    # 1. Extract (or load cached) features
    feat_df = extract_features()
    seq_ids = feat_df["seq_id"].tolist()
    X_blind = feat_df.drop(columns=["seq_id"])

    # 2. Load classifier (load_data populates feature_names and the test set for validation)
    clf = EnzymeClassifier(features_csv=FEATURES_CSV, results_dir=PROJECT_DIR / "results")
    clf.load_data()
    clf.load_model(MODEL_DIR)

    # 3. Validate confidence thresholds on the test set
    clf.validate_confidence()

    # 4. Align feature columns to training order (ensure same order and no missing features)
    X_blind = X_blind[clf.feature_names]

    # 5. Predict
    predictions, probabilities = clf.predict(X_blind)

    # Confidence score = P(predicted class):
    #   class 0 -> probabilities[i, 0] = P(not enzyme) from Stage 1
    #   class k -> probabilities[i, k] = P(enzyme) * P(EC k | enzyme)

    # Take the probability of the PREDICTED class for each sample, 
    # rather than having an array of probabilities for all classes
    conf_scores = probabilities[range(len(predictions)), predictions]

    # 6. Format and write predictions
    lines = []
    for seq_id, pred, score in zip(seq_ids, predictions, conf_scores):
        conf = clf.confidence_label(score)
        lines.append(f"{seq_id} {pred} Confidence {conf}")

    print("Blind test predictions:")
    for line in lines:
        print(f"  {line}")

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"\nPredictions saved -> {OUTPUT_TXT}")

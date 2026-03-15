# Protein Function Prediction

Given a FASTA sequence, the model predicts whether it is an enzyme and, if so, its EC class (EC1–EC6).

A two-stage hierarchical XGBoost classifier is used.

Each sequence is represented by 927 features: classical physicochemical descriptors (607) and ESM-2 protein language model embeddings (320).

# Setup

```bash
pip install -r requirements.txt
```

# Usage

Feature extraction, training, evaluation, ablations and data analysis can be performed by running the scripts in the src folder.

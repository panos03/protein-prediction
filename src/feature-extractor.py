"""
protein_feature_extractor.py
============================
Reads protein sequences from FASTA files and extracts numerical features
for enzyme classification.

Features extracted per sequence (574 total)
--------------------------------------------
  length               : sequence length                             (1)
  AAC                  : amino acid composition                      (20)
  DPC                  : dipeptide composition                       (400)
  physicochemical      : MW, pI, aromaticity, instability_index,
                         GRAVY, charge_at_pH7                        (6)
  CTD                  : Composition / Transition / Distribution
                         over 7 physicochemical property groups      (147)

Class labels
------------
  class0-not_an_enzyme → 0
  ec1-oxidoreductases  → 1
  ec2-transferases     → 2
  ec3-hydrolases       → 3
  ec4-lyases           → 4
  ec5-isomerases       → 5
  ec6-ligases          → 6

Usage
-----
  from protein_feature_extractor import ProteinFeatureExtractor

  extractor = ProteinFeatureExtractor(fasta_dir="data/fasta-files")
  df = extractor.run()          # returns a pandas DataFrame
  extractor.to_csv("data/features.csv")

Dependencies: biopython, pandas
"""

import itertools
import math
import sys
from collections import Counter
from pathlib import Path
import time

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# !!! TODO: look through, change

class ProteinFeatureExtractor:
    """
    Extracts numerical features from protein FASTA files for
    enzyme classification.

    Parameters
    ----------
    fasta_dir : str or Path
        Directory containing .fasta FASTA files whose filenames begin with
        a recognised class prefix (e.g. "class0-…", "ec1-…", …, "ec6-…").
    min_length : int
        Sequences shorter than this are silently skipped (default 2).
    """

    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    AA_SET = set(AMINO_ACIDS)

    CLASS_MAP = {
        "class0": 0,
        "ec1": 1,
        "ec2": 2,
        "ec3": 3,
        "ec4": 4,
        "ec5": 5,
        "ec6": 6,
    }

    # Dubchak et al. (1995) physicochemical groupings for CTD features.
    # Each property maps to three groups covering all 20 standard AAs.
    CTD_PROPERTIES: dict[str, list[set]] = {
        "hydrophobicity": [set("RKEDQN"), set("GASTPHY"), set("CVLIMFW")],
        "volume": [set("GASTCPD"), set("NVEQIL"), set("MHKFRYW")],
        "polarity": [set("LIFWCMVY"), set("PATGS"), set("HQRKNED")],
        "polarizability": [set("GASDT"), set("CPNVEQIL"), set("KMHFRYW")],
        "charge": [set("KR"), set("ACFGHILMNPQSTVWY"), set("DE")],
        "secondary_structure": [set("EALMQKRH"), set("VIYCWFT"), set("GNPSD")],
        "solvent_accessibility": [set("ALFCGIVW"), set("RKQEND"), set("MSPTHY")],
    }

    def __init__(self, fasta_dir: str | Path, min_length: int = 2, verbose: bool = False):
        self.fasta_dir = Path(fasta_dir)
        self.min_length = min_length
        self.verbose = verbose
        self._df: pd.DataFrame | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Parse all FASTA files, extract features, return a DataFrame."""
        fasta_files = sorted(self.fasta_dir.glob("*.fasta"))
        if not fasta_files:
            sys.exit(f"No .fasta FASTA files found in {self.fasta_dir}")

        rows: list[dict] = []
        skipped = 0
        start_time = time.time()

        for path in fasta_files:
            self._print_if_verbose(f"\n\nExtracting features for sequences in {path.name}\n")
            label = self._label_from_filename(path.name)
            records = list(SeqIO.parse(str(path), "fasta"))
            print(f"    {path.name:<45} {len(records):>5} sequences  (label {label})")

            num_records = len(records)
            for i, record in enumerate(records):
                if i % 100 == 0:
                    self._print_if_verbose(f"Extracted features for sequence {i}/{num_records}")
                seq = self._clean(str(record.seq))
                if len(seq) < self.min_length:
                    skipped += 1
                    self._print_if_verbose(f"   !! skipped {record}")
                    continue
                try:
                    feats = self._extract(seq)
                    feats["label"] = label
                    rows.append(feats)
                except Exception as exc:
                    skipped += 1
                    print(f"    !! skipped ({exc}): {seq[:40]}...")

        if not rows:
            sys.exit("No features extracted — check your FASTA files.")

        self._print_if_verbose(f"Extraction complete. Took {(time.time() - start_time):.1f} seconds")

        self._df = pd.DataFrame(rows)
        print(f"\nExtracted {len(self._df)} samples x "
              f"{len(self._df.columns) - 1} features")
        if skipped:
            print(f"  ({skipped} sequences skipped)")
        return self._df

    def to_csv(self, path: str | Path) -> None:
        """Write the feature DataFrame to CSV."""
        if self._df is None:
            raise RuntimeError("Call .run() before .to_csv()")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(out, index=False)
        print(f"Saved → {out}")

    # ── Private helpers ────────────────────────────────────────────────────

    def _label_from_filename(self, filename: str) -> int:
        for prefix, label in self.CLASS_MAP.items():
            if filename.startswith(prefix):
                return label
        raise ValueError(
            f"Cannot determine label from '{filename}'. "
            f"Expected prefix in {list(self.CLASS_MAP)}"
        )

    def _clean(self, seq: str) -> str:
        """Uppercase and retain only the 20 canonical amino acids."""
        return "".join(aa for aa in seq.upper() if aa in self.AA_SET)

    # ── Feature extraction ─────────────────────────────────────────────────

    def _extract(self, seq: str) -> dict:
        """Compute all features for a single cleaned sequence."""
        feats: dict = {}
        feats.update(self._feat_length(seq))
        feats.update(self._feat_aac(seq))
        feats.update(self._feat_dpc(seq))
        feats.update(self._feat_physicochemical(seq))
        feats.update(self._feat_ctd(seq))
        return feats

    def _feat_length(self, seq: str) -> dict:
        return {"length": len(seq)}

    def _feat_aac(self, seq: str) -> dict:
        """Amino Acid Composition: frequency of each of the 20 AAs."""
        n = len(seq)
        counts = Counter(seq)
        return {f"AAC_{aa}": counts.get(aa, 0) / n for aa in self.AMINO_ACIDS}

    def _feat_dpc(self, seq: str) -> dict:
        """Dipeptide Composition: frequency of all 400 dipeptide pairs."""
        n = len(seq) - 1
        if n <= 0:
            return {
                f"DPC_{a}{b}": 0.0
                for a, b in itertools.product(self.AMINO_ACIDS, repeat=2)
            }
        counts = Counter(seq[i : i + 2] for i in range(n))
        return {
            f"DPC_{a}{b}": counts.get(f"{a}{b}", 0) / n
            for a, b in itertools.product(self.AMINO_ACIDS, repeat=2)
        }

    def _feat_physicochemical(self, seq: str) -> dict:
        """Six global physicochemical descriptors via Biopython."""
        pa = ProteinAnalysis(seq)
        return {
            "MW": pa.molecular_weight(),
            "pI": pa.isoelectric_point(),
            "aromaticity": pa.aromaticity(),
            "instability_index": pa.instability_index(),
            "GRAVY": pa.gravy(),
            "charge_at_pH7": pa.charge_at_pH(7.0),
        }

    def _feat_ctd(self, seq: str) -> dict:
        """
        Composition-Transition-Distribution features (147 total).

        For each of 7 physicochemical properties the 20 amino acids are
        split into 3 groups, then three descriptor types are computed:

        Composition  (C):  fraction of residues in each group         → 3
        Transition   (T):  fraction of adjacent pairs that cross
                           groups (3 pairwise combinations)           → 3
        Distribution (D):  normalised position of the 0th, 25th,
                           50th, 75th, 100th percentile residue
                           for each group                             → 15
                                                                  ──────
                                                          21 per property
                                                         * 7 properties
                                                         = 147 features
        """
        features: dict[str, float] = {}
        n = len(seq)

        for prop, groups in self.CTD_PROPERTIES.items():
            # Map each residue to its 1-based group index
            labels = [self._aa_group(aa, groups) for aa in seq]

            # ── Composition ──
            for g in (1, 2, 3):
                features[f"CTDC_{prop}_G{g}"] = labels.count(g) / n

            # ── Transition ──
            n_adj = n - 1
            trans: Counter = Counter()
            for i in range(n_adj):
                a, b = labels[i], labels[i + 1]
                if a != b:
                    trans[tuple(sorted((a, b)))] += 1
            for g1, g2 in ((1, 2), (1, 3), (2, 3)):
                features[f"CTDT_{prop}_G{g1}G{g2}"] = (
                    trans.get((g1, g2), 0) / n_adj if n_adj > 0 else 0.0
                )

            # ── Distribution ──
            for g in (1, 2, 3):
                positions = [i for i, lbl in enumerate(labels) if lbl == g]
                cnt = len(positions)
                if cnt == 0:
                    for pct in (0, 25, 50, 75, 100):
                        features[f"CTDD_{prop}_G{g}_p{pct}"] = 0.0
                else:
                    pct_indices = [
                        0,
                        max(0, math.ceil(cnt * 0.25) - 1),
                        max(0, math.ceil(cnt * 0.50) - 1),
                        max(0, math.ceil(cnt * 0.75) - 1),
                        cnt - 1,
                    ]
                    for pct, idx in zip((0, 25, 50, 75, 100), pct_indices):
                        features[f"CTDD_{prop}_G{g}_p{pct}"] = (
                            (positions[idx] + 1) / n
                        )

        return features
    
    def _print_if_verbose(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _aa_group(aa: str, groups: list[set]) -> int:
        """Return 1-based group index for an amino acid under a CTD property."""
        for i, grp in enumerate(groups, start=1):
            if aa in grp:
                return i
        return 0  # should not happen after cleaning


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_DIR = SCRIPT_DIR.parent
    FASTA_DIR = PROJECT_DIR / "data" / "fasta-files"
    OUTPUT_CSV = PROJECT_DIR / "data" / "features.csv"

    total_features = 1 + 20 + 400 + 6 + 147  # = 574
    print("Protein Feature Extractor")
    print(f"  FASTA dir : {FASTA_DIR}")
    print(f"  Output    : {OUTPUT_CSV}")
    print(f"  Features  : {total_features}  "
          f"(1 length + 20 AAC + 400 DPC + 6 physico + 147 CTD)\n")

    extractor = ProteinFeatureExtractor(fasta_dir=FASTA_DIR, verbose=True)
    extractor.run()
    extractor.to_csv(OUTPUT_CSV)
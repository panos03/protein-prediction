"""
Features extracted per sequence (607 total)

length               : sequence length                             (1)
AAC                  : amino acid composition                      (20)
DPC                  : dipeptide composition                       (400)
physicochemical      : MW, pI, aromaticity, instability_index,
                       GRAVY, charge_at_pH7, extinction coeffs,
                       flexibility mean/std                        (10)
secondary_structure  : helix, turn, sheet fractions                (3)
catalytic_residues   : positional stats + clustering for           (13)
                       key catalytic AAs (C, H, S, D, E)
motifs               : counts of conserved catalytic motifs        (11)
complexity           : Shannon entropy                             (2)
CTD                  : Composition / Transition / Distribution
                       over 7 physicochemical property groups      (147)
"""

import itertools
import math
import re
import sys
from collections import Counter
from pathlib import Path
import time

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# !!! TODO: look through, change

class ProteinFeatureExtractor:

    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

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
    # Each property maps to three groups, covering all 20 standard AAs.
    CTD_PROPERTIES = {
        "hydrophobicity": [set("RKEDQN"), set("GASTPHY"), set("CVLIMFW")],
        "volume": [set("GASTCPD"), set("NVEQIL"), set("MHKFRYW")],
        "polarity": [set("LIFWCMVY"), set("PATGS"), set("HQRKNED")],
        "polarizability": [set("GASDT"), set("CPNVEQIL"), set("KMHFRYW")],
        "charge": [set("KR"), set("ACFGHILMNPQSTVWY"), set("DE")],
        "secondary_structure": [set("EALMQKRH"), set("VIYCWFT"), set("GNPSD")],
        "solvent_accessibility": [set("ALFCGIVW"), set("RKQEND"), set("MSPTHY")],
    }

    # Key catalytic residues — disproportionately found in active sites
    CATALYTIC_RESIDUES = list("CHSDE")

    # Conserved motifs associated with catalytic activity (PROSITE-style)
    CATALYTIC_MOTIFS = {
        "CxxC":      r"C.{2}C",       # redox-active disulfide (EC1)
        "GxGxxG":    r"G.G.{2}G",     # Rossmann fold (EC1, EC2)
        "GxSxG":     r"G.S.G",        # serine hydrolase triad (EC3)
        "GxxxxGK":   r"G.{4}GK",      # P-loop / Walker A, ATP binding (EC6)
        "HxxH":      r"H.{2}H",       # metal-binding (various metalloenzymes)
        "DxE":       r"D.E",          # acid-base catalysis (EC4, EC5)
        "ExD":       r"E.D",          # acid-base catalysis (EC4, EC5)
        "HExxH":     r"HE.{2}H",      # zinc-binding (EC3, EC4)
        "HRD":       r"HRD",          # kinase active site (EC2)
        "CxxxxxR":   r"C.{5}R",       # phosphatase signature (EC3)
    }

    def __init__(self, fasta_dir, min_length=2, verbose=False):
        self.fasta_dir = Path(fasta_dir)
        self.min_length = min_length
        self.verbose = verbose
        self._df = None


    def run(self):

        # Parse all FASTA files, extract features, return a DataFrame

        fasta_files = sorted(self.fasta_dir.glob("*.fasta"))
        if not fasta_files:
            sys.exit(f"No .fasta FASTA files found in {self.fasta_dir}")

        rows = []
        skipped = 0
        start_time = time.time()

        for path in fasta_files:

            self._print_if_verbose(f"\n\nExtracting features for sequences in {path.name}\n")
            label = self._label_from_filename(path.name)
            records = list(SeqIO.parse(str(path), "fasta"))     # parse using Biopython framework
            print(f"    {path.name:<45} {len(records):>5} sequences  (label {label})")

            num_records = len(records)
            for i, record in enumerate(records):
                if i % 100 == 0:
                    self._print_if_verbose(f"Extracted features for sequence {i:>5}/{num_records:<5}")
                seq = self._clean(str(record.seq))
                if len(seq) < self.min_length:
                    skipped += 1
                    self._print_if_verbose(f"   !! skipped: \n{record}")
                    continue
                try:
                    feats = self._extract(seq)
                    feats["label"] = label
                    rows.append(feats)
                except Exception as exc:
                    skipped += 1
                    print(f"    !! skipped ({exc}): {seq[:40]}...")

        if not rows:
            sys.exit("No features extracted — check FASTA files.")

        self._print_if_verbose(f"\nExtraction complete. Took {(time.time() - start_time):.1f} seconds")

        self._df = pd.DataFrame(rows)
        print(f"\nExtracted {len(self._df)} samples x "
              f"{len(self._df.columns) - 1} features")
        if skipped:
            print(f"  ({skipped} sequences skipped)")

        return self._df


    def to_csv(self, path):

        if self._df is None:
            raise RuntimeError("Call .run() before .to_csv()")
        
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(out, index=False)
        print(f"Saved → {out}")


    def _label_from_filename(self, filename):

        for prefix, label in self.CLASS_MAP.items():
            if filename.startswith(prefix):
                return label
        raise ValueError(
            f"Cannot determine label from '{filename}'. "
            f"Expected prefix in {list(self.CLASS_MAP)}"
        )


    def _clean(self, seq):
        # ensure seq is uppercase and has only letters of the 20 amino acids
        return "".join(aa for aa in seq.upper() if aa in self.AMINO_ACIDS)


    def _extract(self, seq):
        # Compute all features
        feats = {}
        feats.update(self._feat_length(seq))
        feats.update(self._feat_aac(seq))
        feats.update(self._feat_dpc(seq))
        feats.update(self._feat_physicochemical(seq))
        feats.update(self._feat_secondary_structure(seq))
        feats.update(self._feat_catalytic_residues(seq))
        feats.update(self._feat_motifs(seq))
        feats.update(self._feat_complexity(seq))
        feats.update(self._feat_ctd(seq))
        return feats


    def _feat_length(self, seq):
        return {"length": len(seq)}


    def _feat_aac(self, seq):
        # Amino Acid Composition: frequency of each of the 20 AAs
        n = len(seq)
        counts = Counter(seq)
        return {f"AAC_{aa}": counts.get(aa, 0) / n for aa in self.AMINO_ACIDS}


    def _feat_dpc(self, seq):
        # Dipeptide Composition: frequency of all 400 dipeptide pairs
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


    def _feat_physicochemical(self, seq):
        # 10 physicochemical features from Biopython
        pa = ProteinAnalysis(seq)
        ext_reduced, ext_oxidized = pa.molar_extinction_coefficient()   # how strongly the protein absorbs UV light (reduced = all Cys free; oxidized = disulfide bonds formed)
        flex = pa.flexibility()     # per-residue backbone flexibility scores (Vihinen scale); summarised to mean/std below
        return {
            "MW": pa.molecular_weight(),
            "pI": pa.isoelectric_point(),
            "aromaticity": pa.aromaticity(),
            "instability_index": pa.instability_index(),
            "GRAVY": pa.gravy(),
            "charge_at_pH7": pa.charge_at_pH(7.0),
            "extinction_coeff_reduced": ext_reduced,
            "extinction_coeff_oxidized": ext_oxidized,
            "flexibility_mean": np.mean(flex),
            "flexibility_std": np.std(flex),
        }


    def _feat_secondary_structure(self, seq):
        # Predicted secondary structure content: estimates helix, turn,
        # and sheet inclination from amino acid composition
        pa = ProteinAnalysis(seq)       # using Biopython framework
        helix, turn, sheet = pa.secondary_structure_fraction()
        return {
            "SS_helix": helix,
            "SS_turn": turn,
            "SS_sheet": sheet,
        }


    def _feat_catalytic_residues(self, seq):
        # Positional and clustering features for catalytic AAs (C, H, S, D, E). 
        # Enzymes tend to have catalytic residues clustered near the active site; 
        # non-enzymes tend to have them dispersed.

        # NOTE: frequency is NOT included here — it would duplicate AAC.
        # Instead we capture WHERE catalytic residues sit (mean position,
        # spread) and HOW they cluster (gap statistics).

        n = len(seq)
        features = {}
        all_positions = []

        # Position statistics for each catalytic residue
        for aa in self.CATALYTIC_RESIDUES:
            positions = [i for i, c in enumerate(seq) if c == aa]
            count = len(positions)

            if count > 0:
                normed_positions = [p / n for p in positions]
                features[f"CAT_{aa}_mean_pos"] = np.mean(normed_positions)
                features[f"CAT_{aa}_std_pos"] = np.std(normed_positions) if count > 1 else 0.0
            else:
                features[f"CAT_{aa}_mean_pos"] = 0.0
                features[f"CAT_{aa}_std_pos"] = 0.0

            all_positions.extend(positions)

        # Gap statistics across all catalytic residues
        if len(all_positions) >= 2:
            sorted_pos = sorted(all_positions)
            gaps = [sorted_pos[i+1] - sorted_pos[i] for i in range(len(sorted_pos)-1)]
            features["CAT_max_gap"] = max(gaps) / n
            features["CAT_mean_gap"] = np.mean(gaps) / n
            features["CAT_std_gap"] = np.std(gaps) / n if len(gaps) > 1 else 0.0
        else:
            features["CAT_max_gap"] = 1.0
            features["CAT_mean_gap"] = 1.0
            features["CAT_std_gap"] = 0.0

        return features


    def _feat_motifs(self, seq):
        # Counts of conserved catalytic motifs - these are associated with catalytic activity
        # Each motif is EC-class-specific - see CATALYTIC_MOTIFS for details.
        features = {}
        total = 0
        for name, pattern in self.CATALYTIC_MOTIFS.items():
            count = len(re.findall(pattern, seq))   # using regex to find occurrences of motif in seq
            features[f"MOTIF_{name}"] = count
            total += count
        features["MOTIF_total"] = total
        return features


    def _feat_complexity(self, seq):
        # Shannon entropy of amino acid composition. Enzymes tend toward
        # moderate-high complexity (diverse residues for catalysis);
        # low-complexity regions suggest disordered/repetitive non-enzymes.
        n = len(seq)
        counts = Counter(seq)
        probs = [c / n for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(20)
        return {
            "COMP_entropy": entropy,
            "COMP_entropy_norm": entropy / max_entropy,
        }


    def _feat_ctd(self, seq):
        # Composition-Transition-Distribution features (147 total).

        # For each of 7 physicochemical properties the 20 amino acids are
        # split into 3 groups, then three descriptor types are computed:

        # Composition  (C):  fraction of residue (in sequence) in each group         → 3

        # Transition   (T):  fraction of adjacent pairs that cross
        #                    groups (3 pairwise combinations)           → 3
        # (how often groups alternate in the sequence / whether they are the same together)

        # Distribution (D):  normalised position of the 0th, 25th,
        #                    50th, 75th, 100th percentile residue
        #                    for each group                             → 15
        # (where in the sequence each group sits)

        # 21 descriptors per property * 7 properties = 147 features

        features = {}
        n = len(seq)

        for prop, groups in self.CTD_PROPERTIES.items():
            # Map each AA to its group index
            labels = [self._aa_group(aa, groups) for aa in seq]

            # Composition
            for g in (1, 2, 3):
                features[f"CTDC_{prop}_G{g}"] = labels.count(g) / n

            # Transition
            n_adj = n - 1
            trans = Counter()
            for i in range(n_adj):
                a, b = labels[i], labels[i + 1]
                if a != b:
                    trans[tuple(sorted((a, b)))] += 1
            for g1, g2 in ((1, 2), (1, 3), (2, 3)):
                features[f"CTDT_{prop}_G{g1}G{g2}"] = (
                    trans.get((g1, g2), 0) / n_adj if n_adj > 0 else 0.0
                )

            # Distribution
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
    def _aa_group(aa, groups):
        # Return 1-based group index, for an amino acid under a CTD property
        for i, grp in enumerate(groups, start=1):
            if aa in grp:
                return i
        return 0  # should not happen after cleaning



if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).parent
    PROJECT_DIR = SCRIPT_DIR.parent
    FASTA_DIR = PROJECT_DIR / "data" / "fasta-files"
    OUTPUT_CSV = PROJECT_DIR / "data" / "features.csv"

    total_features = 1 + 20 + 400 + 10 + 3 + 13 + 11 + 2 + 147  # = 607
    print("Protein Feature Extractor")
    print(f"  FASTA dir : {FASTA_DIR}")
    print(f"  Output    : {OUTPUT_CSV}")
    print(f"  Expected Features  : {total_features}\n")

    extractor = ProteinFeatureExtractor(fasta_dir=FASTA_DIR, verbose=True)
    df = extractor.run()
    extractor.to_csv(OUTPUT_CSV)

    print(f"Dataframe shape: {df.shape}")

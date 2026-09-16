"""Reviewer-audited MPR>=90% prediction pipeline for manuscript use.

Key safeguards: strict fold-local data-dependent screening, nested feature/model
selection, development-only primary-model and threshold locking, independent
external validation, transparent diagnostic/exploratory outputs, and unified
vector/raster manuscript figures.
"""

from __future__ import annotations

import json
import hashlib
import platform
from importlib import metadata as importlib_metadata
import re
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from statsmodels.stats.proportion import proportion_confint
    HAS_PROPORTION_CONFINT = True
except Exception:
    HAS_PROPORTION_CONFINT = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

INPUT_XLSX = "ts.xlsx"
INPUT_SHEET = "Sheet6"
OUTDIR = Path("mpr90_stable_rf_rfe_outputs")

CENTER_COL = "center"
MPR_COL = "mpr"
TRAIN_CENTER = "yz"   # center 2 / development
VALID_CENTER = "zy"   # center 1 / locked external validation

MPR_CUTOFF = 0.90
MPR_TOLERANCE = 1e-8
ALLOW_PERCENT_SCALE_AUTODETECT = False
RANDOM_STATE = 20260712

# Repeated nested cross-validation. Balanced is recommended for the present
# sample size. Thorough doubles outer repeats and bootstrap iterations.
ANALYSIS_INTENSITY = "balanced"  # "balanced" or "thorough"
if ANALYSIS_INTENSITY == "balanced":
    N_OUTER_SPLITS = 5
    N_OUTER_REPEATS = 5
    N_INNER_SPLITS = 4
    N_BOOTSTRAP = 1000
elif ANALYSIS_INTENSITY == "thorough":
    N_OUTER_SPLITS = 5
    N_OUTER_REPEATS = 10
    N_INNER_SPLITS = 5
    N_BOOTSTRAP = 2000
else:
    raise ValueError("ANALYSIS_INTENSITY must be 'balanced' or 'thorough'.")
N_CALIBRATION_BOOTSTRAP = min(500, N_BOOTSTRAP)

# Candidate quality control.
MAX_MISSING_RATE = 0.40
MIN_NONMISSING = 20
MAX_CATEGORICAL_LEVELS = 12
MAX_DOMINANT_LEVEL_RATE = 0.98
MIN_CATEGORY_FREQUENCY = 3

# EPV>=10 hard feature/parameter budget.
EPV_MIN_PER_PARAMETER = 10
FEATURE_BUDGET_BASIS = "minority_class"  # conservative default
MIN_FEATURE_BUDGET = 1
MAX_FEATURE_BUDGET = 12

# Stable random-forest recursive feature elimination (RF-RFE).
# The selector is deterministic because every split/forest uses fixed seeds.
# Within each outer-training fold, the RF selector is tuned by inner CV, then
# recursive elimination is driven by grouped random-forest importance; stability is quantified across repeated outer folds.
RF_RFE_SELECTOR_INNER_SPLITS = 3
RF_RFE_SELECTOR_GRID = {
    "model__max_depth": [3, None],
    "model__min_samples_leaf": [4, 8],
    "model__max_features": ["sqrt", 0.8],
}
RF_RFE_N_ESTIMATORS = 600
RF_RFE_SELECTOR_N_JOBS = 1

# Final-feature stability lock. Features must be selected in at least 60% of
# outer folds to enter the locked feature set. We do NOT fill the EPV budget
# with unstable variables. If fewer than two variables meet the threshold, the
# two highest-frequency variables are retained and clearly flagged as fallback.
FINAL_STABILITY_SELECTION_FREQUENCY = 0.60
MIN_FINAL_LOCKED_RAW_FEATURES = 2

# Model selection is development-only. Mean repeat OOF AUC is primary. Models
# within this margin of the best AUC are tie-broken by averaged-OOF Brier score
# and then model simplicity. External AUC never enters this rule.
AUC_SELECTION_MARGIN = 0.010

# Primary threshold is secondary to AUC and is locked from averaged development
# OOF probabilities only.
THRESHOLD_POLICY = "youden"  # "youden", "sensitivity_0.80", "fixed_0.50"

# Prediction-time profile.
PREDICTION_PROFILE = "preoperative"
VALID_PREDICTION_PROFILES = {"preoperative"}

# Hyperparameter search parallelism. GridSearchCV can use all cores; estimators
# themselves are kept at one worker to avoid nested over-subscription.
GRIDSEARCH_N_JOBS = -1

# Optional extended libraries are used when installed.
RUN_EXTENDED_MODEL_SET = True
USE_CUDA_IF_AVAILABLE = False  # CPU default improves cross-machine reproducibility

# External permutation importance is diagnostic only and never used for feature
# or model selection.
PERMUTATION_REPEATS_EXTERNAL = 30


# Reviewer-facing strictness and figure export.
# All data-dependent candidate screening is repeated inside every outer-training fold.
STRICT_OUTER_FOLD_SCREENING = True
# Silent selector fallbacks can hide model-fitting failures; disabled by default.
ALLOW_RF_SELECTOR_FALLBACK = False
# If the minority outcome class cannot support even one parameter at the prespecified
# EPV heuristic, stop rather than silently violating the stated complexity rule.
STRICT_EPV_BUDGET = True

FIGURE_DPI = 600
FIGURE_FORMATS = ("svg", "pdf", "png", "tiff")
FIGURE_DIR = OUTDIR / "figures"
MAIN_FIGURE_MAX_FEATURES = 12

# Consistent manuscript palette (color-blind-conscious, high contrast on white).
# Nature-skill inspired NMI pastel palette.
# Use one signal family (indigo/slate), one neutral family, and one soft accent.
# Green/red are intentionally avoided because they are reserved mainly for
# directional gains/losses in the nature-figure stance.
COLOR_PRIMARY = "#484878"       # NMI pastel baseline_dark: primary/locked evidence
COLOR_SECONDARY = "#7884B4"     # NMI pastel baseline_mid: secondary signal
COLOR_SOFT = "#B4C0E4"          # NMI pastel baseline_soft: uncertainty/error bars
COLOR_ACCENT = "#E4CCD8"        # NMI pastel ours_base: prespecified threshold accent
COLOR_ACCENT_DARK = "#9A4D8E"   # nature-figure violet: threshold annotation accent
COLOR_NEUTRAL_LIGHT = "#D8D8D8"
COLOR_NEUTRAL = "#A8A8A8"
COLOR_NEUTRAL_DARK = "#606060"
COLOR_DARK = "#272727"
COLOR_PALE = "#F3F3F6"

# Exact source-column aliases. Matching is case-insensitive and punctuation-
# tolerant via normalize_column_key().
COLUMN_ALIASES = {
    "centre": "center", "site": "center", "hospital": "center",
    "patientid": "id", "patient_id": "id", "caseid": "id", "case_id": "id",
    "childscore": "child_score", "child_score": "child_score",
    "childpughscore": "child_score", "child_pugh_score": "child_score",
    "ecogps": "ecog", "ecog_ps": "ecog",
    "tumorsize": "tumorsizeb", "tumor_size": "tumorsizeb", "baseline_tumor_size": "tumorsizeb",
    "tumornumber": "tumornum", "tumor_number": "tumornum", "tumorcount": "tumornum",
    "portalveintumorthrombus": "pvtt", "portal_vein_tumor_thrombus": "pvtt",
    "vasculartumorthrombus": "vtt", "vascular_tumor_thrombus": "vtt",
    "totalbilirubin": "tbil", "total_bilirubin": "tbil", "albumin": "alb",
    "wbc_before": "wbcb", "baseline_wbc": "wbcb", "wbc_after": "wbca", "followup_wbc": "wbca",
    "afp_before": "afpb", "baseline_afp": "afpb", "afp_after": "afpa", "followup_afp": "afpa",
    "nlr_before": "nlrb", "baseline_nlr": "nlrb", "sii_before": "siib", "baseline_sii": "siib",
    "tace_cycles": "tacecycles", "targettherapy": "targeted", "targetedtherapy": "targeted",
    "targeted_therapy": "targeted", "immunotherapy": "imm", "immune_therapy": "imm",
    "necrosis": "mpr", "necrosis_fraction": "mpr", "mpr_fraction": "mpr",
    "recurrence_status": "status", "recurrencefreesurvival": "rfs", "recurrence_free_survival": "rfs",
    "mrecist": "mrecist_response", "mrecistresponse": "mrecist_response",
    "mrecist_response": "mrecist_response", "radiologicresponse": "mrecist_response",
    "radiologicalresponse": "mrecist_response", "imagingresponse": "mrecist_response",
    "rcr": "rcr", "radiologiccr": "rcr", "radiologicalcr": "rcr", "radiologic_complete_response": "rcr",
    "ccr": "ccr", "clinicalcr": "ccr", "clinical_complete_response": "ccr",
    "pivkaiib": "pivka_b", "pivkab": "pivka_b", "dcpb": "pivka_b", "baselinepivka": "pivka_b",
    "pivkaiia": "pivka_a", "pivkaa": "pivka_a", "dcpa": "pivka_a", "followuppivka": "pivka_a",
    "enhancingtumorsizeb": "enhancing_size_b", "baselineenhancingsize": "enhancing_size_b",
    "enhancingtumorsizea": "enhancing_size_a", "posttreatmentenhancingsize": "enhancing_size_a",
}

PREOPERATIVE_NUMERIC_CANDIDATES = [
    "age", "bmi", "child_score", "ecog", "tumorsizeb", "tumornum_count",
    "log1p_ast", "log1p_alt", "log1p_tbil", "alb", "log1p_wbcb",
    "log1p_pt", "log1p_afpb", "log1p_nlrb", "log1p_siib", "log1p_crp",
    "log1p_pivka_b",
    # Treatment exposure available before surgery / pathology.
    "tacecycles",
]
PREOPERATIVE_CATEGORICAL_CANDIDATES = [
    "gender", "bclc", "cnlc_major", "pvtt_any", "vtt_any",
    "mrecist_response", "rcr_any", "ccr_any",
    "targeted_any", "immunotherapy_any",
]

# User-confirmed postoperative/follow-up variables. These are hard excluded from
# candidate predictors and are also removed from the engineered modelling table.
POSTOPERATIVE_FOLLOWUP_RAW_COLUMNS = {
    "afpa", "wbca", "pivka_a", "enhancing_size_a",
}
POSTOPERATIVE_DERIVED_COLUMNS = {
    "afp_log_reduction", "wbc_log_change", "pivka_log_reduction",
    "enhancing_tumor_log_reduction",
}

if PREDICTION_PROFILE not in VALID_PREDICTION_PROFILES:
    raise ValueError(f"Invalid PREDICTION_PROFILE={PREDICTION_PROFILE!r}")
NUMERIC_CANDIDATES = list(PREOPERATIVE_NUMERIC_CANDIDATES)
CATEGORICAL_CANDIDATES = list(PREOPERATIVE_CATEGORICAL_CANDIDATES)

MUTUALLY_EXCLUSIVE_GROUPS = {
    "stage_system": ["bclc", "cnlc_major"],
    "vascular_invasion": ["pvtt_any", "vtt_any"],
    "radiologic_response": ["mrecist_response", "rcr_any", "ccr_any"],
}

LEAKAGE_OR_NONPREDICTOR_COLUMNS = {
    "center": "development/external split variable",
    "id": "identifier",
    "mpr": "prediction outcome",
    "mvi": "postoperative pathological variable",
    "status": "postoperative recurrence outcome",
    "rfs": "postoperative survival outcome",
    "afpa": "postoperative/follow-up AFP; user-confirmed unavailable at preoperative prediction time",
    "wbca": "postoperative/follow-up WBC; user-confirmed unavailable at preoperative prediction time",
    "pivka_a": "postoperative/follow-up PIVKA-II/DCP",
    "enhancing_size_a": "postoperative/follow-up enhancing tumor measurement",
    "afp_log_reduction": "depends on postoperative/follow-up afpa",
    "wbc_log_change": "depends on postoperative/follow-up wbca",
    "pivka_log_reduction": "depends on postoperative/follow-up pivka_a",
    "enhancing_tumor_log_reduction": "depends on postoperative/follow-up enhancing_size_a",
}

FEATURE_LABELS = {
    "age": "Age", "bmi": "BMI", "child_score": "Child-Pugh score",
    "ecog": "ECOG performance status", "tumorsizeb": "Baseline tumor size",
    "tumornum_count": "Tumor number", "log1p_ast": "log(1+AST)",
    "log1p_alt": "log(1+ALT)", "log1p_tbil": "log(1+total bilirubin)",
    "alb": "Albumin", "log1p_wbcb": "log(1+baseline WBC)",
    "log1p_pt": "log(1+PT)", "log1p_afpb": "log(1+baseline AFP)",
    "log1p_nlrb": "log(1+baseline NLR)", "log1p_siib": "log(1+baseline SII)",
    "log1p_crp": "log(1+CRP)", "log1p_pivka_b": "log(1+baseline PIVKA-II/DCP)",
    "afp_log_reduction": "AFP log reduction", "pivka_log_reduction": "PIVKA-II/DCP log reduction",
    "enhancing_tumor_log_reduction": "Enhancing-tumor log reduction",
    "wbc_log_change": "WBC log change", "tacecycles": "TACE cycles before prediction",
    "mrecist_response": "mRECIST response", "rcr_any": "Radiological complete response",
    "ccr_any": "Clinical complete response", "gender": "Sex", "bclc": "BCLC stage",
    "cnlc_major": "CNLC stage", "pvtt_any": "PVTT", "vtt_any": "Any vascular tumor thrombus",
    "targeted_any": "Targeted therapy exposure", "immunotherapy_any": "Immunotherapy exposure",
}

FEATURE_DICTIONARY = [
    ("age", "age", "numeric", "baseline", "Age"),
    ("gender", "gender", "categorical", "baseline", "Sex"),
    ("bmi", "bmi", "numeric", "baseline", "Body mass index"),
    ("bclc", "bclc", "categorical", "baseline", "BCLC stage"),
    ("cnlc", "cnlc_major", "categorical", "baseline", "CNLC stage"),
    ("child_score", "child_score", "numeric", "baseline", "Child-Pugh score"),
    ("ecog", "ecog", "numeric", "baseline", "ECOG performance status"),
    ("tumorsizeb", "tumorsizeb", "numeric", "baseline", "Baseline maximum tumor size"),
    ("tumornum", "tumornum_count", "numeric", "baseline", "Baseline tumor number"),
    ("pvtt", "pvtt_any", "categorical", "baseline", "Portal-vein tumor thrombus present"),
    ("vtt", "vtt_any", "categorical", "baseline", "Any vascular tumor thrombus present"),
    ("ast", "log1p_ast", "numeric", "baseline", "AST, log transformed"),
    ("alt", "log1p_alt", "numeric", "baseline", "ALT, log transformed"),
    ("tbil", "log1p_tbil", "numeric", "baseline", "Total bilirubin, log transformed"),
    ("alb", "alb", "numeric", "baseline", "Albumin"),
    ("wbcb", "log1p_wbcb", "numeric", "baseline", "Baseline WBC, log transformed"),
    ("pt", "log1p_pt", "numeric", "baseline", "Prothrombin time, log transformed"),
    ("afpb", "log1p_afpb", "numeric", "baseline", "Baseline AFP, log transformed"),
    ("nlrb", "log1p_nlrb", "numeric", "baseline", "Baseline NLR, log transformed"),
    ("siib", "log1p_siib", "numeric", "baseline", "Baseline SII, log transformed"),
    ("crp", "log1p_crp", "numeric", "baseline", "CRP, log transformed"),
    ("tacecycles", "tacecycles", "numeric", "preoperative", "TACE cycles accrued before surgery/prediction"),
    ("targeted", "targeted_any", "categorical", "preoperative", "Any targeted therapy before surgery/prediction"),
    ("imm", "immunotherapy_any", "categorical", "preoperative", "Any immunotherapy before surgery/prediction"),
    ("pivka_b", "log1p_pivka_b", "numeric", "baseline", "Baseline PIVKA-II/DCP, log transformed"),
    ("mrecist_response", "mrecist_response", "categorical", "preoperative", "Preoperative mRECIST response"),
    ("rcr", "rcr_any", "categorical", "preoperative", "Radiological complete response before surgery"),
    ("ccr", "ccr_any", "categorical", "preoperative", "Clinical complete response before surgery"),
]

MODEL_COMPLEXITY_RANK = {
    "RidgeLogistic": 0,
    "ElasticNetLogistic": 1,
    "SVM_RBF": 2,
    "RandomForest": 3,
    "ExtraTrees": 4,
    "HistGradientBoosting": 5,
    "XGBoost": 6,
    "LightGBM": 7,
    "CatBoost": 8,
}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, ensure_ascii=False, default=_json_default)

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_versions() -> Dict[str, str]:
    packages = [
        "numpy", "pandas", "scikit-learn", "scipy", "statsmodels",
        "matplotlib", "joblib", "xgboost", "lightgbm", "catboost",
    ]
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions



def has_cuda_gpu() -> bool:
    if not USE_CUDA_IF_AVAILABLE:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


CUDA_AVAILABLE = has_cuda_gpu()

# Two audited, nonfatal warnings are suppressed selectively.
# Do NOT globally suppress warnings: convergence/numerical/fitting warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Found unknown categories in columns .* during transform\. These unknown categories will be encoded as all zeros",
    category=UserWarning,
)


def exact_proportion_ci(successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    if HAS_PROPORTION_CONFINT:
        low, high = proportion_confint(int(successes), int(total), alpha=alpha, method="beta")
        return float(low), float(high)
    p = successes / total
    z = 1.959963984540054
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def safe_auc(y_true, probability) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(probability)
    y_true = y_true[valid]
    probability = probability[valid]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    if np.nanstd(probability) <= 1e-12:
        return 0.5
    try:
        return float(roc_auc_score(y_true, probability))
    except Exception:
        return np.nan


def safe_average_precision(y_true, probability) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(probability)
    y_true = y_true[valid]
    probability = probability[valid]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y_true, probability))
    except Exception:
        return np.nan


def positive_probability(model, X) -> np.ndarray:
    probability = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    matches = np.where(classes == 1)[0]
    if len(matches) != 1:
        raise RuntimeError(f"Could not identify positive class 1 in classes={classes.tolist()}")
    result = probability[:, int(matches[0])]
    return np.clip(np.asarray(result, dtype=float), 1e-8, 1 - 1e-8)


def bootstrap_auc_ci(
    y_true,
    probability,
    n_bootstrap: Optional[int] = None,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float, float]:
    if n_bootstrap is None:
        n_bootstrap = N_BOOTSTRAP
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    point = safe_auc(y_true, probability)
    if not np.isfinite(point):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        value = safe_auc(y_true[indices], probability[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return point, np.nan, np.nan
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def calibration_intercept_slope(y_true, probability) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    logit = np.log(probability / (1 - probability))
    try:
        X = sm.add_constant(logit, has_constant="add")
        fit = sm.GLM(y_true, X, family=sm.families.Binomial()).fit()
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def bootstrap_calibration_ci(
    y_true,
    probability,
    n_bootstrap: Optional[int] = None,
    seed: int = RANDOM_STATE,
) -> Dict[str, float]:
    if n_bootstrap is None:
        n_bootstrap = N_CALIBRATION_BOOTSTRAP
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    point_intercept, point_slope = calibration_intercept_slope(y_true, probability)
    rng = np.random.default_rng(seed)
    intercepts, slopes = [], []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        intercept, slope = calibration_intercept_slope(y_true[indices], probability[indices])
        if np.isfinite(intercept):
            intercepts.append(intercept)
        if np.isfinite(slope):
            slopes.append(slope)
    int_low, int_high = (np.percentile(intercepts, [2.5, 97.5]) if intercepts else (np.nan, np.nan))
    slope_low, slope_high = (np.percentile(slopes, [2.5, 97.5]) if slopes else (np.nan, np.nan))
    return {
        "calibration_intercept": point_intercept,
        "calibration_intercept_95CI_low": float(int_low),
        "calibration_intercept_95CI_high": float(int_high),
        "calibration_slope": point_slope,
        "calibration_slope_95CI_low": float(slope_low),
        "calibration_slope_95CI_high": float(slope_high),
    }


def expected_calibration_error(y_true, probability, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, n_bins - 1)
    result = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        result += float(np.mean(mask)) * abs(float(np.mean(y_true[mask])) - float(np.mean(probability[mask])))
    return float(result)


def null_brier_score(y_true) -> float:
    y_true = np.asarray(y_true, dtype=int)
    prevalence = float(np.mean(y_true))
    return float(np.mean((y_true - prevalence) ** 2))


def brier_skill_score(y_true, probability) -> float:
    model_brier = brier_score_loss(y_true, probability)
    baseline = null_brier_score(y_true)
    if baseline <= 1e-12:
        return np.nan
    return float(1 - model_brier / baseline)


def safe_binary_splits(y: Sequence[int], requested: int) -> int:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Both binary classes are required; counts={counts.tolist()}")
    return max(2, min(int(requested), int(counts.min())))





def normalize_column_key(name: object) -> str:
    """Normalize Excel headers without using data values or outcomes."""
    text = str(name).strip().lower()
    text = text.replace("％", "%")
    text = re.sub(r"[\s\-/\\()\[\]{}]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff%]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    compact = text.replace("_", "")
    return COLUMN_ALIASES.get(text, COLUMN_ALIASES.get(compact, text))


def standardize_input_columns(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply deterministic aliases and fail loudly on canonical-name collisions."""
    original_columns = list(data.columns)
    canonical_columns = [normalize_column_key(column) for column in original_columns]
    rows = []
    for original, canonical in zip(original_columns, canonical_columns):
        rows.append({
            "original_column": str(original),
            "canonical_column": canonical,
            "renamed": str(original) != canonical,
        })
    audit = pd.DataFrame(rows)
    duplicates = audit[audit["canonical_column"].duplicated(keep=False)]
    if len(duplicates):
        raise ValueError(
            "Multiple input columns map to the same canonical name. Resolve these "
            f"columns before analysis:\n{duplicates.to_string(index=False)}"
        )
    result = data.copy()
    result.columns = canonical_columns
    return result, audit


def _parse_numeric_cell(value) -> Tuple[float, str]:
    if value is None:
        return np.nan, "missing"
    try:
        if pd.isna(value):
            return np.nan, "missing"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return (number, "direct_numeric") if np.isfinite(number) else (np.nan, "invalid")

    text = str(value).strip()
    if text == "" or text.lower() in {"na", "n/a", "nan", "none", "null", "missing"}:
        return np.nan, "missing"
    text = text.replace("，", ",").replace("。", ".").replace("％", "%")
    text = text.replace(",", "")
    status = "parsed_text"
    if ".." in text and re.fullmatch(r"[-+]?\d+\.\.\d+", text):
        text = text.replace("..", ".", 1)
        status = "repaired_duplicate_decimal"
    percent = "%" in text
    text = text.replace("%", "")
    # Accept common inequality-prefixed or unit-suffixed laboratory entries and
    # retain the observed numeric boundary. Every such conversion is audited.
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if match is None:
        return np.nan, "invalid"
    try:
        number = float(match.group(0))
    except ValueError:
        return np.nan, "invalid"
    if not np.isfinite(number):
        return np.nan, "invalid"
    if percent:
        number = number / 100.0
        status = "parsed_percent"
    elif match.group(0) != text:
        status = "parsed_numeric_substring"
    return number, status


def parse_numeric_series(series: pd.Series, feature: str) -> Tuple[pd.Series, Dict[str, object]]:
    values = []
    statuses = []
    for value in series.tolist():
        parsed, status = _parse_numeric_cell(value)
        values.append(parsed)
        statuses.append(status)
    parsed_series = pd.Series(values, index=series.index, dtype=float)
    counts = pd.Series(statuses).value_counts().to_dict()
    audit = {
        "source_feature": feature,
        "n_rows": int(len(series)),
        "raw_nonmissing": int(series.notna().sum()),
        "parsed_nonmissing": int(parsed_series.notna().sum()),
        "invalid_to_missing": int(counts.get("invalid", 0)),
        "repaired_duplicate_decimal": int(counts.get("repaired_duplicate_decimal", 0)),
        "parsed_percent": int(counts.get("parsed_percent", 0)),
        "parsed_numeric_substring": int(counts.get("parsed_numeric_substring", 0)),
    }
    return parsed_series, audit


def safe_log1p(series: pd.Series, source_feature: str) -> Tuple[pd.Series, Dict[str, object]]:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_negative = numeric < 0
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = numeric.notna() & ~invalid_negative
    result.loc[valid] = np.log1p(numeric.loc[valid])
    return result, {
        "derived_feature": f"log1p_{source_feature}",
        "source_feature": source_feature,
        "derivation": "log1p",
        "source_nonmissing": int(numeric.notna().sum()),
        "derived_nonmissing": int(result.notna().sum()),
        "negative_values_set_missing": int(invalid_negative.sum()),
    }


def derive_any_indicator(series: pd.Series, feature: str) -> Tuple[pd.Series, Dict[str, object]]:
    negative_tokens = {
        "0", "0.0", "no", "none", "absent", "negative", "false", "n",
        "否", "无", "未", "未见", "阴性", "没有",
    }
    positive_tokens = {
        "1", "1.0", "yes", "present", "positive", "true", "y",
        "是", "有", "阳性",
    }
    missing_tokens = {"", "na", "n/a", "nan", "null", "missing"}
    output = []
    unknown_positive = 0
    for value in series.tolist():
        if value is None:
            output.append(np.nan)
            continue
        try:
            if pd.isna(value):
                output.append(np.nan)
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip().lower()
        if text in missing_tokens:
            output.append(np.nan)
            continue
        try:
            number = float(text)
            output.append("0" if number <= 0 else "1")
            continue
        except (TypeError, ValueError, OverflowError):
            pass
        if text in negative_tokens or any(token in text for token in ["无", "阴性", "negative", "absent"]):
            output.append("0")
        elif text in positive_tokens:
            output.append("1")
        else:
            # A named PVTT/VTT grade, drug name or regimen is evidence of exposure.
            output.append("1")
            unknown_positive += 1
    result = pd.Series(output, index=series.index, dtype=object)
    return result, {
        "derived_feature": feature,
        "source_nonmissing": int(series.notna().sum()),
        "derived_nonmissing": int(result.notna().sum()),
        "positive_n": int((result == "1").sum()),
        "negative_n": int((result == "0").sum()),
        "nonstandard_nonempty_values_treated_positive": int(unknown_positive),
    }


def engineer_clinical_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map Sheet6 columns and construct PREOPERATIVE predictors only.

    User-confirmed postoperative/follow-up variables (the *_a variables) are
    deliberately excluded before modelling. No predictor derived from them is
    created. This prevents temporal leakage into the preoperative MPR model.
    """
    result = data.copy()

    # Hard-remove postoperative/follow-up columns from the modelling table.
    drop_post = [c for c in POSTOPERATIVE_FOLLOWUP_RAW_COLUMNS if c in result.columns]
    if drop_post:
        result = result.drop(columns=drop_post)

    numeric_sources = [
        "age", "bmi", "child_score", "ecog", "tumorsizeb", "tumornum",
        "ast", "alt", "tbil", "alb", "wbcb", "pt", "afpb", "nlrb",
        "siib", "crp", "tacecycles", "pivka_b", "enhancing_size_b",
    ]
    parsing_rows = []
    for feature in numeric_sources:
        if feature not in result.columns:
            continue
        result[feature], audit = parse_numeric_series(result[feature], feature)
        parsing_rows.append(audit)

    derivation_rows = []
    if "tumornum" in result.columns:
        result["tumornum_count"] = result["tumornum"]
        derivation_rows.append({
            "derived_feature": "tumornum_count",
            "source_feature": "tumornum",
            "derivation": "numeric identity after audited parsing",
            "source_nonmissing": int(result["tumornum"].notna().sum()),
            "derived_nonmissing": int(result["tumornum_count"].notna().sum()),
            "negative_values_set_missing": int((result["tumornum_count"] < 0).sum()),
        })
        result.loc[result["tumornum_count"] < 0, "tumornum_count"] = np.nan

    for source_feature in [
        "ast", "alt", "tbil", "wbcb", "pt", "afpb", "nlrb", "siib",
        "crp", "pivka_b",
    ]:
        if source_feature not in result.columns:
            continue
        transformed, audit = safe_log1p(result[source_feature], source_feature)
        result[f"log1p_{source_feature}"] = transformed
        derivation_rows.append(audit)

    if "cnlc" in result.columns:
        result["cnlc_major"] = result["cnlc"].map(normalize_categorical_scalar).astype(object)
        derivation_rows.append({
            "derived_feature": "cnlc_major",
            "source_feature": "cnlc",
            "derivation": "categorical normalization; original stage retained",
            "source_nonmissing": int(result["cnlc"].notna().sum()),
            "derived_nonmissing": int(result["cnlc_major"].notna().sum()),
            "negative_values_set_missing": 0,
        })

    if "mrecist_response" in result.columns:
        result["mrecist_response"] = result["mrecist_response"].map(normalize_categorical_scalar).astype(object)
        derivation_rows.append({
            "derived_feature": "mrecist_response",
            "source_feature": "mrecist_response",
            "derivation": "categorical normalization of preoperative mRECIST response",
            "source_nonmissing": int(result["mrecist_response"].notna().sum()),
            "derived_nonmissing": int(result["mrecist_response"].notna().sum()),
            "negative_values_set_missing": 0,
        })

    for source_feature, target_feature in [
        ("pvtt", "pvtt_any"),
        ("vtt", "vtt_any"),
        ("targeted", "targeted_any"),
        ("imm", "immunotherapy_any"),
        ("rcr", "rcr_any"),
        ("ccr", "ccr_any"),
    ]:
        if source_feature not in result.columns:
            continue
        result[target_feature], audit = derive_any_indicator(result[source_feature], target_feature)
        audit["source_feature"] = source_feature
        audit["derivation"] = "binary preoperative indicator/exposure"
        audit["negative_values_set_missing"] = 0
        derivation_rows.append(audit)

    # Explicit guard: no postoperative-derived predictor may survive.
    forbidden = [c for c in POSTOPERATIVE_DERIVED_COLUMNS if c in result.columns]
    if forbidden:
        result = result.drop(columns=forbidden)

    return result, pd.DataFrame(parsing_rows), pd.DataFrame(derivation_rows)


def write_feature_design_audits(data: pd.DataFrame) -> None:
    rows = []
    candidate_set = set(NUMERIC_CANDIDATES) | set(CATEGORICAL_CANDIDATES)
    for source, engineered, feature_type, timing, description in FEATURE_DICTIONARY:
        rows.append({
            "source_column": source,
            "engineered_feature": engineered,
            "feature_type": feature_type,
            "timing": timing,
            "description": description,
            "active_profile": PREDICTION_PROFILE,
            "candidate_in_active_profile": engineered in candidate_set,
            "engineered_column_present": engineered in data.columns,
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "feature_dictionary_and_timing.csv", index=False)

    excluded_rows = []
    for column, reason in LEAKAGE_OR_NONPREDICTOR_COLUMNS.items():
        excluded_rows.append({
            "column": column,
            "present": column in data.columns,
            "excluded_from_prediction": True,
            "reason": reason,
        })
    pd.DataFrame(excluded_rows).to_csv(OUTDIR / "excluded_leakage_and_nonpredictor_columns.csv", index=False)



def normalize_mpr_fraction(series: pd.Series) -> pd.Series:
    numeric, audit = parse_numeric_series(series, MPR_COL)
    pd.DataFrame([audit]).to_csv(OUTDIR / "mpr_numeric_parsing_audit.csv", index=False)
    observed = numeric.dropna()
    if len(observed) == 0:
        return numeric
    if observed.min() < -MPR_TOLERANCE:
        raise ValueError("MPR/necrosis fraction contains negative values.")
    if observed.max() > 1 + MPR_TOLERANCE:
        if ALLOW_PERCENT_SCALE_AUTODETECT and observed.max() <= 100 + MPR_TOLERANCE:
            numeric = numeric / 100.0
        else:
            raise ValueError(
                "Sheet6 mpr must be a 0-1 fraction. Values above 1 were found. "
                "Convert percentages such as 90 to 0.90 before running."
            )
    if (numeric.dropna() > 1 + MPR_TOLERANCE).any():
        raise ValueError("MPR fraction remains above 1 after normalization.")
    return numeric.clip(lower=0.0, upper=1.0)



def validate_candidate_features(
    train_df: pd.DataFrame,
    numeric_candidates: Optional[Sequence[str]] = None,
    categorical_candidates: Optional[Sequence[str]] = None,
    *,
    output_path: Optional[Path] = None,
    context: str = "development",
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Outcome-independent candidate screening fitted only on the supplied data.

    In the primary nested-CV analysis this function is called separately inside
    every outer-training fold. The full-development call is used only to define
    the final model after the model-development procedure has been evaluated.
    """
    numeric_candidates = list(NUMERIC_CANDIDATES if numeric_candidates is None else numeric_candidates)
    categorical_candidates = list(CATEGORICAL_CANDIDATES if categorical_candidates is None else categorical_candidates)
    rows: List[Dict[str, object]] = []
    valid_numeric: List[str] = []
    valid_categorical: List[str] = []

    for feature in numeric_candidates:
        if feature not in train_df.columns:
            rows.append({
                "context": context, "feature": feature, "type": "numeric",
                "present": False, "n_nonmissing": 0, "missing_rate": np.nan,
                "n_unique": 0, "dominant_value_rate": np.nan,
                "valid": False, "reason": "missing_column",
            })
            continue
        values = pd.to_numeric(train_df[feature], errors="coerce")
        n_nonmissing = int(values.notna().sum())
        missing_rate = float(values.isna().mean())
        n_unique = int(values.nunique(dropna=True))
        dominant_rate = float(values.value_counts(normalize=True, dropna=True).max()) if values.notna().any() else np.nan
        valid = (
            n_nonmissing >= MIN_NONMISSING
            and missing_rate <= MAX_MISSING_RATE
            and n_unique > 1
            and (not np.isfinite(dominant_rate) or dominant_rate <= MAX_DOMINANT_LEVEL_RATE)
        )
        reason = "" if valid else "insufficient_usable_variation_or_missingness"
        rows.append({
            "context": context, "feature": feature, "type": "numeric",
            "present": True, "n_nonmissing": n_nonmissing,
            "missing_rate": missing_rate, "n_unique": n_unique,
            "dominant_value_rate": dominant_rate, "valid": bool(valid), "reason": reason,
        })
        if valid:
            valid_numeric.append(feature)

    for feature in categorical_candidates:
        if feature not in train_df.columns:
            rows.append({
                "context": context, "feature": feature, "type": "categorical",
                "present": False, "n_nonmissing": 0, "missing_rate": np.nan,
                "n_unique": 0, "dominant_value_rate": np.nan,
                "valid": False, "reason": "missing_column",
            })
            continue
        values = train_df[feature].replace("", np.nan)
        n_nonmissing = int(values.notna().sum())
        missing_rate = float(values.isna().mean())
        n_unique = int(values.nunique(dropna=True))
        dominant_rate = float(values.value_counts(normalize=True, dropna=True).max()) if values.notna().any() else np.nan
        valid = (
            n_nonmissing >= MIN_NONMISSING
            and missing_rate <= MAX_MISSING_RATE
            and n_unique > 1
            and n_unique <= MAX_CATEGORICAL_LEVELS
            and (not np.isfinite(dominant_rate) or dominant_rate <= MAX_DOMINANT_LEVEL_RATE)
        )
        if n_unique > MAX_CATEGORICAL_LEVELS:
            reason = "too_many_levels"
        elif np.isfinite(dominant_rate) and dominant_rate > MAX_DOMINANT_LEVEL_RATE:
            reason = "near_zero_variance_dominant_level"
        elif not valid:
            reason = "insufficient_usable_variation_or_missingness"
        else:
            reason = ""
        rows.append({
            "context": context, "feature": feature, "type": "categorical",
            "present": True, "n_nonmissing": n_nonmissing,
            "missing_rate": missing_rate, "n_unique": n_unique,
            "dominant_value_rate": dominant_rate, "valid": bool(valid), "reason": reason,
        })
        if valid:
            valid_categorical.append(feature)

    table = pd.DataFrame(rows)
    if output_path is not None:
        table.to_csv(output_path, index=False)
    if len(valid_numeric) + len(valid_categorical) < 2:
        raise RuntimeError(
            f"Fewer than two eligible candidate features remain after outcome-independent screening ({context})."
        )
    return valid_numeric, valid_categorical, table


def duplicate_identifier_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["patient_id", "id", "sample_id", "case_id", "name"]:
        if column not in df.columns:
            continue
        duplicated = df[column].notna() & df[column].duplicated(keep=False)
        rows.append({
            "identifier": column,
            "duplicated_rows": int(duplicated.sum()),
            "duplicated_unique_values": int(df.loc[duplicated, column].nunique()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTDIR / "audit_duplicate_identifiers.csv", index=False)
    return result


def continuous_smd(x_train, x_valid) -> float:
    x_train = pd.to_numeric(x_train, errors="coerce").dropna()
    x_valid = pd.to_numeric(x_valid, errors="coerce").dropna()
    if len(x_train) < 2 or len(x_valid) < 2:
        return np.nan
    pooled_sd = np.sqrt((x_train.var(ddof=1) + x_valid.var(ddof=1)) / 2)
    if not np.isfinite(pooled_sd) or pooled_sd <= 1e-12:
        return 0.0
    return float((x_valid.mean() - x_train.mean()) / pooled_sd)


def binary_smd(p_train: float, p_valid: float) -> float:
    pooled_var = (p_train * (1 - p_train) + p_valid * (1 - p_valid)) / 2
    if pooled_var <= 1e-12:
        return 0.0
    return float((p_valid - p_train) / np.sqrt(pooled_var))





def normalize_categorical_scalar(value):
    """Deterministically normalize mixed Excel categorical cell types."""
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return np.nan
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError, OverflowError):
        pass
    return text


def normalize_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Apply type-only normalization without learning from outcome/distribution."""
    result = data.copy()
    for feature in CATEGORICAL_CANDIDATES:
        if feature in result.columns:
            result[feature] = result[feature].map(normalize_categorical_scalar).astype(object)
    return result


def make_onehot_encoder():
    # sklearn >=1.1 supports infrequent-category pooling. The fallback preserves
    # compatibility with older installations. Pooling is fitted within each fold.
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=MIN_CATEGORY_FREQUENCY,
            drop="first",
            sparse_output=False,
        )
    except TypeError:
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=MIN_CATEGORY_FREQUENCY,
                drop="first",
                sparse=False,
            )
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def make_preprocessor(numeric: Sequence[str], categorical: Sequence[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, list(numeric)))
    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", make_onehot_encoder()),
        ])
        transformers.append(("categorical", categorical_pipeline, list(categorical)))
    if not transformers:
        raise ValueError("No valid features are available for preprocessing.")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )


def feature_group(feature: str) -> Optional[str]:
    for group_name, members in MUTUALLY_EXCLUSIVE_GROUPS.items():
        if feature in members:
            return group_name
    return None


def transformed_to_raw_feature(name: str, numeric: Sequence[str], categorical: Sequence[str]) -> Optional[str]:
    name = str(name)
    if name in numeric:
        return name
    # SimpleImputer(add_indicator=True) emits names such as
    # missingindicator_log1p_afpb in recent sklearn versions.
    for prefix in ["missingindicator_", "missing_indicator_"]:
        if name.startswith(prefix):
            candidate = name[len(prefix):]
            if candidate in numeric:
                return candidate
    for raw_feature in sorted(categorical, key=len, reverse=True):
        if name == raw_feature or name.startswith(raw_feature + "_"):
            return raw_feature
    return None



def task_feature_budget(y, basis: str = FEATURE_BUDGET_BASIS) -> Dict[str, int]:
    """Conservative predictor-degree-of-freedom budget motivated by EPV>=10.

    This is an anti-overfitting design constraint, not a claim that classical EPV
    theory exactly characterizes every nonlinear learner. The smaller outcome
    class is used by default. Categorical predictors consume k-1 approximate df.
    """
    y = np.asarray(y, dtype=int)
    positive_n = int(np.sum(y == 1))
    negative_n = int(np.sum(y == 0))
    positive_event_budget = positive_n // EPV_MIN_PER_PARAMETER
    minority_budget = min(positive_n, negative_n) // EPV_MIN_PER_PARAMETER
    if basis == "positive_events":
        raw_budget = int(positive_event_budget)
    elif basis == "minority_class":
        raw_budget = int(minority_budget)
    else:
        raise ValueError("FEATURE_BUDGET_BASIS must be 'minority_class' or 'positive_events'.")
    if STRICT_EPV_BUDGET and raw_budget < MIN_FEATURE_BUDGET:
        raise ValueError(
            "The prespecified EPV complexity rule cannot support the minimum feature budget: "
            f"positive_n={positive_n}, negative_n={negative_n}, raw_budget={raw_budget}."
        )
    budget = max(MIN_FEATURE_BUDGET, min(MAX_FEATURE_BUDGET, raw_budget))
    return {
        "positive_n": positive_n,
        "negative_n": negative_n,
        "positive_event_budget": int(positive_event_budget),
        "minority_class_budget": int(minority_budget),
        "raw_effective_df_budget": int(raw_budget),
        "effective_df_budget": int(budget),
        "epv_rule_satisfied": bool(raw_budget >= MIN_FEATURE_BUDGET),
    }


def raw_feature_df_cost(train_df: pd.DataFrame, feature: str, numeric: Sequence[str]) -> int:
    """Approximate model degrees of freedom contributed by one raw feature.

    Numeric features cost 1 df. Categorical variables cost k-1 df after the
    drop-first one-hot encoding used by this script; missingness is counted as a
    potential additional level when present. This makes the EPV cap apply to
    effective model parameters rather than only raw column count.
    """
    if feature in numeric:
        return 1
    values = train_df[feature].copy()
    missing_present = bool(values.isna().any() or (values.astype(str).str.strip() == "").any())
    nonmissing = values.replace("", np.nan).dropna().astype(str)
    n_levels = int(nonmissing.nunique()) + (1 if missing_present else 0)
    return max(1, n_levels - 1)


def selected_feature_df_cost(train_df: pd.DataFrame, features: Sequence[str], numeric: Sequence[str]) -> int:
    return int(sum(raw_feature_df_cost(train_df, f, numeric) for f in features))

# =============================================================================
# 6. STABLE NESTED RANDOM-FOREST RECURSIVE FEATURE ELIMINATION (RF-RFE)
# =============================================================================


def tune_rf_rfe_selector(
    train_df: pd.DataFrame,
    y,
    numeric: Sequence[str],
    categorical: Sequence[str],
    seed: int,
) -> Tuple[Dict[str, object], float]:
    """Tune the RF selector inside the current training sample only."""
    y = np.asarray(y, dtype=int)
    features = list(numeric) + list(categorical)
    if not features:
        raise ValueError("No features available to tune the RF-RFE selector.")
    pipe = Pipeline([
        ("preprocess", make_preprocessor(numeric, categorical)),
        ("model", RandomForestClassifier(
            n_estimators=RF_RFE_N_ESTIMATORS,
            class_weight=None,
            random_state=seed,
            n_jobs=1,
        )),
    ])
    inner_splits = safe_binary_splits(y, RF_RFE_SELECTOR_INNER_SPLITS)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator=pipe,
        param_grid=RF_RFE_SELECTOR_GRID,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=RF_RFE_SELECTOR_N_JOBS,
        refit=True,
        error_score="raise",
    )
    try:
        search.fit(train_df[features], y)
        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        score = float(search.best_score_)
        if not np.isfinite(score):
            raise RuntimeError("RF selector tuning returned a non-finite best CV AUC.")
        return best, score
    except Exception as error:
        if ALLOW_RF_SELECTOR_FALLBACK:
            warnings.warn(
                f"RF selector tuning failed ({error}); using prespecified fallback hyperparameters.",
                RuntimeWarning,
            )
            return {"max_depth": 3, "min_samples_leaf": 4, "max_features": "sqrt"}, np.nan
        raise RuntimeError(f"RF-RFE selector tuning failed: {error}") from error


def _rf_rfe_step_importance(
    train_df: pd.DataFrame,
    y,
    current_features: Sequence[str],
    all_numeric: Sequence[str],
    all_categorical: Sequence[str],
    rf_params: Dict[str, object],
    seed: int,
) -> pd.DataFrame:
    """Grouped raw-feature RF impurity importance for one RFE step.

    The selector is fitted only on the current outer-training sample. One-hot
    dummy importances are summed back to the originating clinical predictor.
    These are MDI/impurity importances (not permutation importance); stability is
    provided by rerunning the complete selector across repeated outer folds.
    """
    y = np.asarray(y, dtype=int)
    current_features = list(current_features)
    current_numeric = [f for f in current_features if f in all_numeric]
    current_categorical = [f for f in current_features if f in all_categorical]
    preprocessor = make_preprocessor(current_numeric, current_categorical)
    X = preprocessor.fit_transform(train_df[current_features])
    rf = RandomForestClassifier(
        n_estimators=RF_RFE_N_ESTIMATORS,
        class_weight=None,
        random_state=seed,
        n_jobs=1,
        **rf_params,
    )
    rf.fit(X, y)
    transformed_names = [str(x) for x in preprocessor.get_feature_names_out()]
    transformed_imp = np.asarray(rf.feature_importances_, dtype=float)
    grouped_sum = {f: 0.0 for f in current_features}
    grouped_n = {f: 0 for f in current_features}
    for name, importance in zip(transformed_names, transformed_imp):
        raw = transformed_to_raw_feature(name, current_numeric, current_categorical)
        if raw is None:
            continue
        grouped_sum[raw] += float(importance)
        grouped_n[raw] += 1
    train_auc = safe_auc(y, positive_probability(rf, X))
    rows = []
    for feature in current_features:
        total_importance = float(grouped_sum.get(feature, 0.0))
        ncols = max(1, int(grouped_n.get(feature, 0)))
        rows.append({
            "feature": feature,
            "grouped_RF_MDI_importance": total_importance,
            "grouped_RF_MDI_mean_per_transformed_column": total_importance / ncols,
            "transformed_column_n": ncols,
            "selector_training_AUC_diagnostic": train_auc,
        })
    return pd.DataFrame(rows)


def select_stable_rf_rfe_features(
    train_df: pd.DataFrame,
    y,
    numeric: Sequence[str],
    categorical: Sequence[str],
    k: int,
    seed: int,
) -> Tuple[List[str], pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """Deterministic RF-RFE under a prespecified effective-df budget.

    The RF selector is tuned by CV inside this training sample, then recursive
    elimination uses grouped RF MDI importance. The routine itself is fold-local;
    stability is quantified later across repeated outer-training folds.
    """
    y = np.asarray(y, dtype=int)
    all_features = list(numeric) + list(categorical)
    if not all_features:
        raise ValueError("No candidate features available for RF-RFE.")

    rf_params, selector_cv_auc = tune_rf_rfe_selector(
        train_df=train_df, y=y, numeric=numeric, categorical=categorical, seed=seed + 17
    )
    df_cost = {f: raw_feature_df_cost(train_df, f, numeric) for f in all_features}
    current = list(all_features)
    path_rows: List[Dict[str, object]] = []
    elimination_info: Dict[str, Dict[str, object]] = {}
    step = 0
    final_step_table: Optional[pd.DataFrame] = None

    while True:
        step += 1
        step_table = _rf_rfe_step_importance(
            train_df=train_df,
            y=y,
            current_features=current,
            all_numeric=numeric,
            all_categorical=categorical,
            rf_params=rf_params,
            seed=seed + 10000 * step,
        )
        current_df = selected_feature_df_cost(train_df, current, numeric)
        step_table["step"] = step
        step_table["current_raw_feature_n"] = len(current)
        step_table["current_effective_df"] = current_df
        step_table["epv_df_budget"] = int(k)
        step_table["feature_df_cost"] = step_table["feature"].map(df_cost)

        if current_df <= int(k):
            step_table["action"] = "retained_final"
            step_table["dropped_this_step"] = False
            path_rows.extend(step_table.to_dict("records"))
            final_step_table = step_table.copy()
            break

        importance_map = dict(zip(step_table["feature"], step_table["grouped_RF_MDI_importance"]))
        # Lower MDI is eliminated first. When tied, prefer dropping a predictor
        # that consumes more df, has lower prespecified clinical priority, then name.
        drop_feature = sorted(
            current,
            key=lambda f: (
                float(importance_map.get(f, -np.inf)),
                -int(df_cost[f]),
                -int(_priority_rank(f)),
                f,
            ),
        )[0]
        step_table["action"] = np.where(step_table["feature"] == drop_feature, "drop", "retain")
        step_table["dropped_this_step"] = step_table["feature"] == drop_feature
        path_rows.extend(step_table.to_dict("records"))
        row = step_table.loc[step_table["feature"] == drop_feature].iloc[0]
        elimination_info[drop_feature] = {
            "elimination_step": step,
            "importance_at_elimination": float(row["grouped_RF_MDI_importance"]),
        }
        current.remove(drop_feature)
        if not current:
            raise RuntimeError("RF-RFE eliminated every candidate feature.")

    if final_step_table is None:
        raise RuntimeError("RF-RFE did not produce a final step.")
    final_features = list(current)
    final_imp = dict(zip(final_step_table["feature"], final_step_table["grouped_RF_MDI_importance"]))
    final_order = sorted(
        final_features,
        key=lambda f: (-float(final_imp.get(f, -np.inf)), _priority_rank(f), f),
    )

    ranking_rows = []
    total_initial = len(all_features)
    for feature in all_features:
        if feature in final_features:
            rank = final_order.index(feature) + 1
            elimination_step = np.nan
            score = float(final_imp.get(feature, np.nan))
        else:
            info = elimination_info.get(feature, {})
            elimination_step = info.get("elimination_step", np.nan)
            rank = len(final_features) + (
                total_initial - int(elimination_step) + 1
                if np.isfinite(elimination_step) else total_initial
            )
            score = float(info.get("importance_at_elimination", np.nan))
        ranking_rows.append({
            "feature": feature,
            "selected": feature in final_features,
            "selected_rank": rank if feature in final_features else np.nan,
            "rf_rfe_overall_rank": rank,
            "feature_df_cost": int(df_cost[feature]),
            "selector_score": score,
            "rf_rfe_grouped_MDI_importance": score,
            "elimination_step": elimination_step,
        })
    ranking = pd.DataFrame(ranking_rows).sort_values("rf_rfe_overall_rank")
    params = {
        "selector_method": "nested_deterministic_RF_RFE_grouped_MDI_with_outer_fold_stability_lock",
        "rf_selector_best_params": rf_params,
        "rf_selector_inner_CV_AUC": selector_cv_auc,
        "epv_df_budget": int(k),
        "selected_raw_feature_n": int(len(final_features)),
        "selected_effective_df": int(selected_feature_df_cost(train_df, final_features, numeric)),
    }
    return final_order, ranking, params, pd.DataFrame(path_rows)


# =============================================================================
# 13. REDUNDANCY CONTROL, INTERPRETABILITY, AND LITERATURE BENCHMARKS
# =============================================================================

# Development-only, outcome-agnostic redundancy control. This is applied after
# quality control and before any model fitting. Every removal is written to CSV.
NUMERIC_CORRELATION_PRUNE_THRESHOLD = 0.90
GROUP_PREFERENCE = {
    "stage_system": ["bclc", "cnlc_major"],
    "vascular_invasion": ["vtt_any", "pvtt_any"],
    "radiologic_response": ["mrecist_response", "ccr_any", "rcr_any"],
}
CLINICAL_FEATURE_PRIORITY = [
    "mrecist_response", "ccr_any", "rcr_any",
    "log1p_afpb", "log1p_pivka_b", "alb", "tumornum_count",
    "vtt_any", "pvtt_any", "tumorsizeb", "child_score", "ecog",
    "tacecycles", "targeted_any", "immunotherapy_any", "log1p_nlrb",
]


def _priority_rank(feature: str) -> int:
    try:
        return CLINICAL_FEATURE_PRIORITY.index(feature)
    except ValueError:
        return len(CLINICAL_FEATURE_PRIORITY) + 100


def prune_candidate_redundancy(
    train_df: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    output_path: Optional[Path] = None,
    context: str = "development",
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Outcome-agnostic redundancy control fitted only on supplied predictors."""
    numeric_kept = list(numeric)
    categorical_kept = list(categorical)
    rows: List[Dict[str, object]] = []

    for group_name, members in MUTUALLY_EXCLUSIVE_GROUPS.items():
        available = [f for f in members if f in categorical_kept]
        if len(available) <= 1:
            continue
        preference = GROUP_PREFERENCE.get(group_name, available)
        stats_rows = []
        for feature in available:
            values = train_df[feature].replace("", np.nan)
            stats_rows.append({
                "feature": feature,
                "missing_rate": float(values.isna().mean()),
                "n_unique": int(values.nunique(dropna=True)),
                "preference_rank": preference.index(feature) if feature in preference else 999,
            })
        stats_table = pd.DataFrame(stats_rows).sort_values(
            ["missing_rate", "preference_rank", "n_unique"], ascending=[True, True, False]
        )
        kept = str(stats_table.iloc[0]["feature"])
        for feature in available:
            if feature == kept:
                continue
            categorical_kept.remove(feature)
            rows.append({
                "context": context, "type": "conceptual_group", "group": group_name,
                "feature_a": kept, "feature_b": feature, "association": np.nan,
                "kept": kept, "dropped": feature,
                "rule": "lower_missingness_then_prespecified_clinical_preference",
            })

    if len(numeric_kept) > 1:
        matrix = train_df[numeric_kept].apply(pd.to_numeric, errors="coerce")
        corr = matrix.corr(method="spearman").abs()
        dropped = set()
        for i, feature_a in enumerate(numeric_kept):
            for feature_b in numeric_kept[i + 1:]:
                if feature_a in dropped or feature_b in dropped:
                    continue
                value = corr.loc[feature_a, feature_b]
                if not np.isfinite(value) or value < NUMERIC_CORRELATION_PRUNE_THRESHOLD:
                    continue
                a = pd.to_numeric(train_df[feature_a], errors="coerce")
                b = pd.to_numeric(train_df[feature_b], errors="coerce")
                key_a = (float(a.isna().mean()), _priority_rank(feature_a), -int(a.nunique(dropna=True)), feature_a)
                key_b = (float(b.isna().mean()), _priority_rank(feature_b), -int(b.nunique(dropna=True)), feature_b)
                kept, drop = (feature_a, feature_b) if key_a <= key_b else (feature_b, feature_a)
                dropped.add(drop)
                rows.append({
                    "context": context, "type": "numeric_spearman", "group": "",
                    "feature_a": feature_a, "feature_b": feature_b,
                    "association": float(value), "kept": kept, "dropped": drop,
                    "rule": "lower_missingness_then_prespecified_clinical_priority_then_more_unique_values",
                })
        numeric_kept = [f for f in numeric_kept if f not in dropped]

    audit = pd.DataFrame(rows, columns=[
        "context", "type", "group", "feature_a", "feature_b", "association",
        "kept", "dropped", "rule",
    ])
    if output_path is not None:
        audit.to_csv(output_path, index=False)
    return numeric_kept, categorical_kept, audit






# =============================================================================
# 8. BINARY MPR COHORT PREPARATION AND AUDIT
# =============================================================================

def load_binary_mpr_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET)
    source, column_audit = standardize_input_columns(raw)
    column_audit.to_csv(OUTDIR / "column_name_mapping_audit.csv", index=False)

    missing_required = [c for c in [CENTER_COL, MPR_COL] if c not in source.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after alias mapping: {missing_required}")

    source, parsing_audit, derivation_audit = engineer_clinical_features(source)
    parsing_audit.to_csv(OUTDIR / "numeric_parsing_audit.csv", index=False)
    derivation_audit.to_csv(OUTDIR / "derived_feature_audit.csv", index=False)
    write_feature_design_audits(source)

    # Cohort-flow audit: no RFS/status-based exclusions are allowed.
    flow = []
    flow.append({"step": "raw_sheet_rows", "n_remaining": int(len(source)), "n_excluded_at_step": 0})

    center_ok = source[CENTER_COL].notna() & (source[CENTER_COL].astype(str).str.strip() != "")
    excluded_center = source.loc[~center_ok].copy()
    source = source.loc[center_ok].copy()
    flow.append({"step": "exclude_missing_center", "n_remaining": int(len(source)), "n_excluded_at_step": int(len(excluded_center))})

    source[CENTER_COL] = source[CENTER_COL].astype(str).str.strip().str.lower()
    source = normalize_categorical_columns(source)

    mpr_present = source[MPR_COL].notna()
    excluded_mpr = source.loc[~mpr_present].copy()
    analysis = source.loc[mpr_present].copy()
    flow.append({"step": "exclude_missing_pathological_necrosis", "n_remaining": int(len(analysis)), "n_excluded_at_step": int(len(excluded_mpr))})

    analysis[MPR_COL] = normalize_mpr_fraction(analysis[MPR_COL])
    parsed_ok = analysis[MPR_COL].notna()
    excluded_unparseable = analysis.loc[~parsed_ok].copy()
    analysis = analysis.loc[parsed_ok].copy()
    flow.append({"step": "exclude_unparseable_pathological_necrosis", "n_remaining": int(len(analysis)), "n_excluded_at_step": int(len(excluded_unparseable))})

    analysis["mpr_fraction"] = analysis[MPR_COL].astype(float)
    analysis["mpr_binary"] = (analysis["mpr_fraction"] >= MPR_CUTOFF - MPR_TOLERANCE).astype(int)

    train = analysis[analysis[CENTER_COL] == TRAIN_CENTER].copy().reset_index(drop=True)
    valid = analysis[analysis[CENTER_COL] == VALID_CENTER].copy().reset_index(drop=True)
    if len(train) == 0 or len(valid) == 0:
        observed = analysis[CENTER_COL].value_counts().to_dict()
        raise ValueError(
            f"Could not resolve development/external centers. TRAIN_CENTER={TRAIN_CENTER!r}, "
            f"VALID_CENTER={VALID_CENTER!r}, observed={observed}"
        )
    if train["mpr_binary"].nunique() < 2 or valid["mpr_binary"].nunique() < 2:
        raise ValueError("Both development and external cohorts must contain MPR and non-MPR patients.")

    flow.append({"step": f"development_center_{TRAIN_CENTER}", "n_remaining": int(len(train)), "n_excluded_at_step": np.nan})
    flow.append({"step": f"external_center_{VALID_CENTER}", "n_remaining": int(len(valid)), "n_excluded_at_step": np.nan})
    pd.DataFrame(flow).to_csv(OUTDIR / "cohort_flow.csv", index=False)

    excluded_rows = []
    for reason, frame in [
        ("missing_center", excluded_center),
        ("missing_pathological_necrosis", excluded_mpr),
        ("unparseable_pathological_necrosis", excluded_unparseable),
    ]:
        if len(frame):
            for idx in frame.index:
                excluded_rows.append({
                    "source_row_index": int(idx),
                    "reason": reason,
                    "id": frame.loc[idx, "id"] if "id" in frame.columns else np.nan,
                })
    pd.DataFrame(excluded_rows, columns=["source_row_index", "reason", "id"]).to_csv(
        OUTDIR / "cohort_exclusion_audit.csv", index=False
    )

    analysis.to_csv(OUTDIR / "analysis_ready_engineered_data.csv", index=False)
    return source, analysis, train, valid


def run_binary_data_integrity_audit(
    analysis_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> None:
    duplicate_identifier_audit(analysis_df)
    rows = []
    for label, frame in [("development", train_df), ("external", valid_df)]:
        y = frame["mpr_binary"].astype(int)
        rows.append({
            "dataset": label,
            "n": int(len(frame)),
            "MPR_n": int(y.sum()),
            "non_MPR_n": int((1-y).sum()),
            "MPR_rate": float(y.mean()),
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "outcome_prevalence_by_dataset.csv", index=False)

    nr = []
    for f in numeric:
        tr = pd.to_numeric(train_df[f], errors="coerce")
        va = pd.to_numeric(valid_df[f], errors="coerce")
        nr.append({
            "feature": f,
            "development_mean": tr.mean(), "development_sd": tr.std(),
            "external_mean": va.mean(), "external_sd": va.std(),
            "development_missing_rate": tr.isna().mean(),
            "external_missing_rate": va.isna().mean(),
            "SMD_external_minus_development": continuous_smd(tr, va),
        })
    pd.DataFrame(nr).to_csv(OUTDIR / "center_heterogeneity_numeric.csv", index=False)

    cr = []
    for f in categorical:
        tr = train_df[f].replace("", np.nan).fillna("Missing").astype(str)
        va = valid_df[f].replace("", np.nan).fillna("Missing").astype(str)
        for level in sorted(set(tr.unique()) | set(va.unique())):
            p_tr = float((tr == level).mean())
            p_va = float((va == level).mean())
            cr.append({
                "feature": f, "level": level,
                "development_proportion": p_tr, "external_proportion": p_va,
                "SMD_external_minus_development": binary_smd(p_tr, p_va),
            })
    pd.DataFrame(cr).to_csv(OUTDIR / "center_heterogeneity_categorical.csv", index=False)


# =============================================================================
# 9. MODEL DEFINITIONS AND INNER-CV HYPERPARAMETER OPTIMISATION
# =============================================================================

def available_model_names() -> List[str]:
    models = [
        "RidgeLogistic",
        "ElasticNetLogistic",
        "SVM_RBF",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
    ]
    if RUN_EXTENDED_MODEL_SET and HAS_XGB:
        models.append("XGBoost")
    if RUN_EXTENDED_MODEL_SET and HAS_LGB:
        models.append("LightGBM")
    if RUN_EXTENDED_MODEL_SET and HAS_CATBOOST:
        models.append("CatBoost")
    return models

MODEL_NAMES = available_model_names()


def make_estimator_and_grid(model_name: str, seed: int) -> Tuple[object, Dict[str, list]]:
    """Return an estimator and a compact clinically appropriate search grid.

    All grids are intentionally modest because n≈128 and the effective feature
    budget is small. The optimisation objective is inner-CV ROC AUC.
    """
    if model_name == "RidgeLogistic":
        estimator = LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight=None,
            max_iter=20000, random_state=seed,
        )
        grid = {"model__C": [0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0]}
        return estimator, grid

    if model_name == "ElasticNetLogistic":
        estimator = LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight=None,
            max_iter=30000, random_state=seed,
        )
        grid = {
            "model__C": [0.03, 0.10, 0.30, 1.0, 3.0],
            "model__l1_ratio": [0.25, 0.50, 0.75, 1.00],
        }
        return estimator, grid

    if model_name == "SVM_RBF":
        estimator = SVC(
            kernel="rbf", probability=True, class_weight=None,
            random_state=seed,
        )
        grid = {
            "model__C": [0.10, 0.30, 1.0, 3.0],
            "model__gamma": ["scale", 0.03, 0.10, 0.30],
        }
        return estimator, grid

    if model_name == "RandomForest":
        estimator = RandomForestClassifier(
            n_estimators=500, class_weight=None, random_state=seed, n_jobs=1,
        )
        grid = {
            "model__max_depth": [2, 4, None],
            "model__min_samples_leaf": [4, 8],
            "model__max_features": ["sqrt", 1.0],
        }
        return estimator, grid

    if model_name == "ExtraTrees":
        estimator = ExtraTreesClassifier(
            n_estimators=500, class_weight=None, random_state=seed, n_jobs=1,
        )
        grid = {
            "model__max_depth": [2, 4, None],
            "model__min_samples_leaf": [4, 8],
            "model__max_features": ["sqrt", 1.0],
        }
        return estimator, grid

    if model_name == "HistGradientBoosting":
        estimator = HistGradientBoostingClassifier(random_state=seed)
        grid = {
            "model__learning_rate": [0.03, 0.08],
            "model__max_leaf_nodes": [3, 7],
            "model__min_samples_leaf": [8, 15],
            "model__l2_regularization": [0.0, 3.0],
        }
        return estimator, grid

    if model_name == "XGBoost" and HAS_XGB:
        params = dict(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=seed, verbosity=0, n_jobs=1,
        )
        if CUDA_AVAILABLE:
            params["device"] = "cuda"
        estimator = xgb.XGBClassifier(**params)
        grid = {
            "model__n_estimators": [150, 300],
            "model__learning_rate": [0.03, 0.08],
            "model__max_depth": [2, 3],
            "model__min_child_weight": [3, 8],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
            "model__reg_lambda": [10.0],
        }
        return estimator, grid

    if model_name == "LightGBM" and HAS_LGB:
        estimator = lgb.LGBMClassifier(
            objective="binary", random_state=seed, verbosity=-1, n_jobs=1,
        )
        grid = {
            "model__n_estimators": [150, 300],
            "model__learning_rate": [0.03, 0.08],
            "model__num_leaves": [3, 7],
            "model__max_depth": [3],
            "model__min_child_samples": [8, 15],
            "model__reg_lambda": [3.0],
        }
        return estimator, grid

    if model_name == "CatBoost" and HAS_CATBOOST:
        estimator = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", verbose=False,
            random_seed=seed, thread_count=1, allow_writing_files=False,
        )
        grid = {
            "model__iterations": [200, 400],
            "model__depth": [2, 3],
            "model__learning_rate": [0.03, 0.08],
            "model__l2_leaf_reg": [5.0],
        }
        return estimator, grid

    raise ValueError(f"Unsupported or unavailable model: {model_name}")


def build_tuned_search(
    model_name: str,
    train_df: pd.DataFrame,
    y,
    selected_features: Sequence[str],
    all_numeric: Sequence[str],
    all_categorical: Sequence[str],
    seed: int,
) -> GridSearchCV:
    numeric = [f for f in selected_features if f in all_numeric]
    categorical = [f for f in selected_features if f in all_categorical]
    preprocessor = make_preprocessor(numeric, categorical)
    estimator, grid = make_estimator_and_grid(model_name, seed)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    inner_splits = safe_binary_splits(y, N_INNER_SPLITS)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=GRIDSEARCH_N_JOBS,
        refit=True,
        return_train_score=False,
        error_score=np.nan,
    )


def pipeline_positive_probability(fitted_pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(fitted_pipeline.classes_)
    pos = np.where(classes == 1)[0]
    if len(pos) != 1:
        raise RuntimeError(f"Positive class 1 not found in classes={classes}")
    return np.asarray(fitted_pipeline.predict_proba(X)[:, int(pos[0])], dtype=float)


# =============================================================================
# 10. OUTER-FOLD FEATURE CACHE AND REPEATED NESTED OOF
# =============================================================================

def prepare_outer_fold_cache(
    train_df: pd.DataFrame,
    y,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    """Prepare strictly nested outer folds including fold-local X-only screening."""
    y = np.asarray(y, dtype=int)
    n = len(y)
    cache: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    path_tables = []
    selector_param_rows = []
    qc_tables = []
    redundancy_tables = []
    candidate_pool = list(dict.fromkeys(list(numeric) + list(categorical)))

    for repeat in range(N_OUTER_REPEATS):
        outer_seed = RANDOM_STATE + 1000 * repeat
        n_splits = safe_binary_splits(y, N_OUTER_SPLITS)
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=outer_seed)
        for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(np.zeros(n), y), start=1):
            fold_train = train_df.iloc[tr_idx].copy()
            y_train = y[tr_idx]
            context = f"repeat{repeat + 1:02d}_fold{fold:02d}"

            if STRICT_OUTER_FOLD_SCREENING:
                fold_numeric, fold_categorical, qc = validate_candidate_features(
                    fold_train,
                    numeric_candidates=numeric,
                    categorical_candidates=categorical,
                    output_path=None,
                    context=context,
                )
                fold_numeric, fold_categorical, redundancy = prune_candidate_redundancy(
                    fold_train, fold_numeric, fold_categorical, output_path=None, context=context
                )
            else:
                fold_numeric, fold_categorical = list(numeric), list(categorical)
                qc = pd.DataFrame()
                redundancy = pd.DataFrame()

            if not qc.empty:
                qc = qc.copy(); qc.insert(0, "fold", fold); qc.insert(0, "repeat", repeat + 1)
                qc_tables.append(qc)
            if not redundancy.empty:
                redundancy = redundancy.copy(); redundancy.insert(0, "fold", fold); redundancy.insert(0, "repeat", repeat + 1)
                redundancy_tables.append(redundancy)

            eligible_features = list(fold_numeric) + list(fold_categorical)
            if len(eligible_features) < 2:
                raise RuntimeError(f"{context}: fewer than two features after fold-local screening.")
            budget_info = task_feature_budget(y_train)
            selected, ranking, selector_params, elimination_path = select_stable_rf_rfe_features(
                train_df=fold_train,
                y=y_train,
                numeric=fold_numeric,
                categorical=fold_categorical,
                k=budget_info["effective_df_budget"],
                seed=outer_seed + fold,
            )

            qc_reason = {}
            if not qc.empty:
                qc_reason = dict(zip(qc["feature"], qc["reason"]))
            redundancy_dropped = set(redundancy["dropped"].tolist()) if not redundancy.empty else set()
            rank_lookup = ranking.set_index("feature").to_dict("index")
            for feature in candidate_pool:
                info = rank_lookup.get(feature)
                screened_out = info is None
                if feature in redundancy_dropped:
                    screen_reason = "outcome_agnostic_redundancy_pruning"
                elif screened_out:
                    screen_reason = qc_reason.get(feature, "not_eligible_in_outer_training_fold")
                else:
                    screen_reason = ""
                selection_rows.append({
                    "repeat": repeat + 1,
                    "fold": fold,
                    "feature": feature,
                    "eligible_after_fold_screening": not screened_out,
                    "screen_reason": screen_reason,
                    "selected": bool(info["selected"]) if info is not None else False,
                    "selected_rank": info["selected_rank"] if info is not None else np.nan,
                    "feature_df_cost": info["feature_df_cost"] if info is not None else np.nan,
                    "selector_score": info["selector_score"] if info is not None else np.nan,
                    "elimination_step": info["elimination_step"] if info is not None else np.nan,
                    "fold_epv_df_budget": budget_info["effective_df_budget"],
                    "fold_selected_effective_df": selector_params["selected_effective_df"],
                })

            if not elimination_path.empty:
                ep = elimination_path.copy()
                ep.insert(0, "fold", fold); ep.insert(0, "repeat", repeat + 1)
                path_tables.append(ep)
            selector_param_rows.append({
                "repeat": repeat + 1,
                "fold": fold,
                "rf_selector_inner_CV_AUC": selector_params.get("rf_selector_inner_CV_AUC", np.nan),
                "rf_selector_best_params_json": json.dumps(
                    selector_params.get("rf_selector_best_params", {}), ensure_ascii=False, default=_json_default
                ),
                "fold_epv_df_budget": budget_info["effective_df_budget"],
                "eligible_features": "|".join(eligible_features),
                "selected_features": "|".join(selected),
                "selected_effective_df": selector_params["selected_effective_df"],
            })
            cache.append({
                "repeat": repeat,
                "fold": fold,
                "train_idx": tr_idx,
                "valid_idx": va_idx,
                "selected_features": list(selected),
                "eligible_numeric": list(fold_numeric),
                "eligible_categorical": list(fold_categorical),
                "budget_info": budget_info,
                "seed": outer_seed + fold,
            })

    history = pd.DataFrame(selection_rows)
    history.to_csv(OUTDIR / "foldwise_RF_RFE_feature_selection_history.csv", index=False)
    history.to_csv(OUTDIR / "foldwise_feature_selection_history.csv", index=False)
    if qc_tables:
        pd.concat(qc_tables, ignore_index=True).to_csv(OUTDIR / "foldwise_candidate_quality_control.csv", index=False)
    if redundancy_tables:
        pd.concat(redundancy_tables, ignore_index=True).to_csv(OUTDIR / "foldwise_candidate_redundancy_pruning.csv", index=False)
    if path_tables:
        pd.concat(path_tables, ignore_index=True).to_csv(OUTDIR / "foldwise_RF_RFE_elimination_path.csv", index=False)
    pd.DataFrame(selector_param_rows).to_csv(OUTDIR / "foldwise_RF_RFE_selector_hyperparameters.csv", index=False)
    return cache, history


def run_repeated_nested_oof_for_model(
    model_name: str,
    train_df: pd.DataFrame,
    y,
    numeric: Sequence[str],
    categorical: Sequence[str],
    fold_cache: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    y = np.asarray(y, dtype=int)
    n = len(y)
    repeat_oof = np.full((N_OUTER_REPEATS, n), np.nan, dtype=float)
    tuning_rows = []

    for item in fold_cache:
        repeat = int(item["repeat"])
        fold = int(item["fold"])
        tr_idx = np.asarray(item["train_idx"], dtype=int)
        va_idx = np.asarray(item["valid_idx"], dtype=int)
        features = list(item["selected_features"])
        seed = int(item["seed"] + 100000 * (MODEL_COMPLEXITY_RANK.get(model_name, 0) + 1))
        X_tr = train_df.iloc[tr_idx][features]
        X_va = train_df.iloc[va_idx][features]
        y_tr = y[tr_idx]
        search = build_tuned_search(
            model_name=model_name,
            train_df=train_df.iloc[tr_idx],
            y=y_tr,
            selected_features=features,
            all_numeric=numeric,
            all_categorical=categorical,
            seed=seed,
        )
        search.fit(X_tr, y_tr)
        probability = pipeline_positive_probability(search.best_estimator_, X_va)
        repeat_oof[repeat, va_idx] = probability
        tuning_rows.append({
            "model": model_name,
            "repeat": repeat + 1,
            "fold": fold,
            "inner_best_AUC": float(search.best_score_) if np.isfinite(search.best_score_) else np.nan,
            "best_params_json": json.dumps(search.best_params_, ensure_ascii=False, default=_json_default),
            "selected_features": "|".join(features),
            "selected_effective_df": selected_feature_df_cost(
                train_df.iloc[tr_idx], features, [f for f in numeric if f in features]
            ),
        })

    if np.isnan(repeat_oof).any():
        raise RuntimeError(f"Incomplete OOF predictions for model={model_name}")
    repeat_rows = []
    for r in range(N_OUTER_REPEATS):
        p = repeat_oof[r]
        repeat_rows.append({
            "model": model_name,
            "repeat": r + 1,
            "OOF_AUC": safe_auc(y, p),
            "OOF_PR_AUC": safe_average_precision(y, p),
            "OOF_Brier": float(brier_score_loss(y, p)),
            "OOF_Brier_skill": brier_skill_score(y, p),
        })
    repeat_metrics = pd.DataFrame(repeat_rows)
    averaged_oof = np.mean(repeat_oof, axis=0)
    return {
        "model": model_name,
        "repeat_oof": repeat_oof,
        "averaged_oof": averaged_oof,
        "repeat_metrics": repeat_metrics,
        "tuning_history": pd.DataFrame(tuning_rows),
    }


def summarize_oof_result(result: Dict[str, object], y) -> Dict[str, object]:
    y = np.asarray(y, dtype=int)
    rm = result["repeat_metrics"]
    p = np.asarray(result["averaged_oof"], dtype=float)
    point, low, high = bootstrap_auc_ci(y, p, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE + 77)
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "model": result["model"],
        "mean_repeat_OOF_AUC": float(rm["OOF_AUC"].mean()),
        "sd_repeat_OOF_AUC": float(rm["OOF_AUC"].std(ddof=1)),
        "min_repeat_OOF_AUC": float(rm["OOF_AUC"].min()),
        "max_repeat_OOF_AUC": float(rm["OOF_AUC"].max()),
        "proportion_repeats_AUC_above_0_5": float((rm["OOF_AUC"] > 0.5).mean()),
        "averaged_repeated_OOF_AUC": point,
        "averaged_repeated_OOF_AUC_95CI_low": low,
        "averaged_repeated_OOF_AUC_95CI_high": high,
        "averaged_repeated_OOF_PR_AUC": safe_average_precision(y, p),
        "averaged_repeated_OOF_Brier": float(brier_score_loss(y, p)),
        "averaged_repeated_OOF_Brier_skill": brier_skill_score(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def select_primary_model(oof_table: pd.DataFrame) -> Tuple[str, str]:
    table = oof_table.copy()
    best_auc = float(table["mean_repeat_OOF_AUC"].max())
    eligible = table[table["mean_repeat_OOF_AUC"] >= best_auc - AUC_SELECTION_MARGIN].copy()
    eligible["complexity_rank"] = eligible["model"].map(MODEL_COMPLEXITY_RANK).fillna(999)
    eligible = eligible.sort_values(
        ["averaged_repeated_OOF_Brier", "complexity_rank", "mean_repeat_OOF_AUC"],
        ascending=[True, True, False],
    )
    chosen = str(eligible.iloc[0]["model"])
    reason = (
        f"Development-only selection: models within {AUC_SELECTION_MARGIN:.3f} of the best "
        f"mean repeated-OOF AUC ({best_auc:.3f}) were tie-broken by lower averaged-OOF "
        f"Brier score and then lower model complexity. External results were not used."
    )
    return chosen, reason


# =============================================================================
# 11. STABILITY-LOCKED FINAL FEATURES AND FULL-DEVELOPMENT TUNING
# =============================================================================

def build_selection_frequency(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_folds = history[["repeat", "fold"]].drop_duplicates().shape[0]
    for feature, group in history.groupby("feature"):
        selected = group[group["selected"] == True]
        rows.append({
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "selected_fold_n": int(len(selected)),
            "total_outer_folds": int(total_folds),
            "selection_frequency": float(len(selected) / max(total_folds, 1)),
            "mean_selected_rank": float(selected["selected_rank"].mean()) if len(selected) else np.nan,
            "median_selected_rank": float(selected["selected_rank"].median()) if len(selected) else np.nan,
            "mean_RF_RFE_grouped_MDI_importance": float(group["selector_score"].replace([np.inf, -np.inf], np.nan).mean()),
            "feature_df_cost_mode": int(group["feature_df_cost"].mode().iloc[0]) if len(group["feature_df_cost"].dropna()) else np.nan,
        })
    table = pd.DataFrame(rows).sort_values(
        ["selection_frequency", "mean_selected_rank", "mean_RF_RFE_grouped_MDI_importance"],
        ascending=[False, True, False],
    )
    table.to_csv(OUTDIR / "foldwise_RF_RFE_selection_frequency.csv", index=False)
    table.to_csv(OUTDIR / "foldwise_feature_selection_frequency.csv", index=False)
    return table


def lock_final_features(
    train_df: pd.DataFrame,
    y,
    numeric: Sequence[str],
    categorical: Sequence[str],
    frequency_table: pd.DataFrame,
) -> Tuple[List[str], pd.DataFrame, Dict[str, object]]:
    """Lock one deterministic, stability-selected feature set from development.

    Final selection rule is prespecified:
      1) EPV>=10 budget from the full development cohort.
      2) Rank by outer-fold RF-RFE selection frequency.
      3) Require frequency >=60% whenever possible.
      4) Full-development deterministic RF-RFE importance is only a tie-breaker.
      5) Do not fill unused EPV capacity with unstable variables.
    """
    y = np.asarray(y, dtype=int)
    budget = task_feature_budget(y)
    full_selected, full_rank, selector_params, full_path = select_stable_rf_rfe_features(
        train_df=train_df,
        y=y,
        numeric=numeric,
        categorical=categorical,
        k=budget["effective_df_budget"],
        seed=RANDOM_STATE + 888,
    )
    full_path.to_csv(OUTDIR / "full_development_RF_RFE_elimination_path.csv", index=False)
    save_json(selector_params, OUTDIR / "full_development_RF_RFE_selector_details.json")

    full_imp = dict(zip(full_rank["feature"], full_rank["rf_rfe_grouped_MDI_importance"]))
    full_sel = dict(zip(full_rank["feature"], full_rank["selected"]))
    freq_map = dict(zip(frequency_table["feature"], frequency_table["selection_frequency"]))
    rank_map = dict(zip(frequency_table["feature"], frequency_table["mean_selected_rank"]))
    all_features = list(numeric) + list(categorical)

    ordered = sorted(
        all_features,
        key=lambda f: (
            -float(freq_map.get(f, 0.0)),
            float(rank_map.get(f, 999.0)) if np.isfinite(rank_map.get(f, np.nan)) else 999.0,
            -int(bool(full_sel.get(f, False))),
            -float(full_imp.get(f, -np.inf)) if np.isfinite(full_imp.get(f, np.nan)) else np.inf,
            _priority_rank(f),
            f,
        ),
    )

    selected = []
    used_groups = set()
    used_df = 0
    for f in ordered:
        stable = float(freq_map.get(f, 0.0)) >= FINAL_STABILITY_SELECTION_FREQUENCY
        if not stable:
            continue
        cost = raw_feature_df_cost(train_df, f, numeric)
        group = feature_group(f)
        if (group is None or group not in used_groups) and used_df + cost <= budget["effective_df_budget"]:
            selected.append(f)
            used_df += cost
            if group is not None:
                used_groups.add(group)

    fallback_features = []
    if len(selected) < MIN_FINAL_LOCKED_RAW_FEATURES:
        for f in ordered:
            if f in selected:
                continue
            cost = raw_feature_df_cost(train_df, f, numeric)
            group = feature_group(f)
            if (group is None or group not in used_groups) and used_df + cost <= budget["effective_df_budget"]:
                selected.append(f)
                fallback_features.append(f)
                used_df += cost
                if group is not None:
                    used_groups.add(group)
            if len(selected) >= MIN_FINAL_LOCKED_RAW_FEATURES:
                break

    if not selected:
        raise RuntimeError("No feature could be locked under the RF-RFE stability and EPV rules.")

    rows = []
    for f in ordered:
        rows.append({
            "feature": f,
            "feature_label": FEATURE_LABELS.get(f, f),
            "selected_final": f in selected,
            "selection_frequency": float(freq_map.get(f, 0.0)),
            "mean_selected_rank": rank_map.get(f, np.nan),
            "full_development_RF_RFE_selected": bool(full_sel.get(f, False)),
            "full_development_RF_RFE_grouped_MDI_importance": float(full_imp.get(f, np.nan)),
            "effective_df_cost": raw_feature_df_cost(train_df, f, numeric),
            "stability_threshold_met": float(freq_map.get(f, 0.0)) >= FINAL_STABILITY_SELECTION_FREQUENCY,
            "fallback_due_to_too_few_stable_features": f in fallback_features,
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTDIR / "final_RF_RFE_feature_selection_manifest.csv", index=False)
    manifest.to_csv(OUTDIR / "final_feature_selection_manifest.csv", index=False)
    final_only = manifest[manifest["selected_final"] == True].copy()
    final_only["locked_model_feature_order"] = range(1, len(final_only) + 1)
    final_only.to_csv(OUTDIR / "final_selected_features.csv", index=False)

    audit = dict(budget)
    audit.update({
        "selector_method": "nested_stable_random_forest_recursive_feature_elimination",
        "final_stability_frequency_threshold": FINAL_STABILITY_SELECTION_FREQUENCY,
        "selected_raw_feature_n": int(len(selected)),
        "selected_effective_df": int(used_df),
        "selected_features": selected,
        "fallback_features_below_stability_threshold": fallback_features,
        "full_development_RF_RFE_selected_features": full_selected,
        "full_development_RF_selector_details": selector_params,
    })
    save_json(audit, OUTDIR / "final_feature_budget_audit.json")
    return selected, final_only, audit


def fit_final_tuned_model(
    model_name: str,
    train_df: pd.DataFrame,
    y,
    selected_features: Sequence[str],
    numeric: Sequence[str],
    categorical: Sequence[str],
    seed: int,
) -> Tuple[object, Dict[str, object], float]:
    search = build_tuned_search(
        model_name=model_name,
        train_df=train_df,
        y=y,
        selected_features=selected_features,
        all_numeric=numeric,
        all_categorical=categorical,
        seed=seed,
    )
    search.fit(train_df[list(selected_features)], np.asarray(y, dtype=int))
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


# =============================================================================
# 12. PERFORMANCE, THRESHOLD, PLOTS, AND INTERPRETABILITY
# =============================================================================

def choose_threshold(y_true, probability, policy: str = THRESHOLD_POLICY) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if policy == "fixed_0.50":
        return 0.50
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    if policy == "youden":
        return float(thresholds[int(np.argmax(tpr - fpr))])
    if policy == "sensitivity_0.80":
        idx = np.where(tpr >= 0.80)[0]
        if len(idx):
            specificity = 1 - fpr[idx]
            return float(thresholds[idx[int(np.argmax(specificity))]])
        return float(thresholds[int(np.argmax(tpr))])
    raise ValueError(f"Unknown threshold policy={policy}")


def bootstrap_threshold_metric_ci(
    y_true, probability, threshold: float, metric: str, n_bootstrap: Optional[int] = None, seed: int = RANDOM_STATE
) -> Tuple[float, float]:
    if n_bootstrap is None:
        n_bootstrap = N_BOOTSTRAP
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(y_true), len(y_true))
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        pred = (p[idx] >= threshold).astype(int)
        if metric == "F1":
            value = f1_score(yb, pred, zero_division=0)
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(yb, pred)
        else:
            raise ValueError(f"Unsupported threshold metric={metric}")
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return np.nan, np.nan
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def binary_metrics(y_true, probability, threshold: float, dataset: str, model: str) -> Dict[str, object]:
    """Discrimination, calibration and threshold metrics with uncertainty."""
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc, auc_low, auc_high = bootstrap_auc_ci(
        y_true, p, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE + len(dataset)
    )
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    sens_ci = exact_proportion_ci(tp, tp + fn) if tp + fn else (np.nan, np.nan)
    spec_ci = exact_proportion_ci(tn, tn + fp) if tn + fp else (np.nan, np.nan)
    ppv_ci = exact_proportion_ci(tp, tp + fp) if tp + fp else (np.nan, np.nan)
    npv_ci = exact_proportion_ci(tn, tn + fn) if tn + fn else (np.nan, np.nan)
    cal_ci = bootstrap_calibration_ci(
        y_true, p, n_bootstrap=N_CALIBRATION_BOOTSTRAP, seed=RANDOM_STATE + 33 + len(dataset)
    )
    accuracy_ci = exact_proportion_ci(int(np.sum(pred == y_true)), len(y_true))
    balanced_ci = bootstrap_threshold_metric_ci(
        y_true, p, threshold, "balanced_accuracy", seed=RANDOM_STATE + 101 + len(dataset)
    )
    f1_ci = bootstrap_threshold_metric_ci(
        y_true, p, threshold, "F1", seed=RANDOM_STATE + 202 + len(dataset)
    )
    return {
        "dataset": dataset, "model": model, "n": len(y_true),
        "MPR_n": int(y_true.sum()), "non_MPR_n": int((1 - y_true).sum()),
        "AUC": auc, "AUC_95CI_low": auc_low, "AUC_95CI_high": auc_high,
        "PR_AUC": safe_average_precision(y_true, p),
        "Brier": float(brier_score_loss(y_true, p)),
        "Brier_null": null_brier_score(y_true),
        "Brier_skill": brier_skill_score(y_true, p),
        "log_loss": float(log_loss(y_true, p)),
        "ECE_10_bins": expected_calibration_error(y_true, p, 10),
        "threshold": float(threshold),
        "sensitivity": sens, "sensitivity_95CI_low": sens_ci[0], "sensitivity_95CI_high": sens_ci[1],
        "specificity": spec, "specificity_95CI_low": spec_ci[0], "specificity_95CI_high": spec_ci[1],
        "PPV": ppv, "PPV_95CI_low": ppv_ci[0], "PPV_95CI_high": ppv_ci[1],
        "NPV": npv, "NPV_95CI_low": npv_ci[0], "NPV_95CI_high": npv_ci[1],
        "accuracy": float(accuracy_score(y_true, pred)),
        "accuracy_95CI_low": accuracy_ci[0],
        "accuracy_95CI_high": accuracy_ci[1],
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "balanced_accuracy_95CI_low": balanced_ci[0],
        "balanced_accuracy_95CI_high": balanced_ci[1],
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "F1_95CI_low": f1_ci[0],
        "F1_95CI_high": f1_ci[1],
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "mean_predicted_probability": float(np.mean(p)),
        "observed_prevalence": float(np.mean(y_true)),
        **cal_ci,
    }


def apply_manuscript_style() -> None:
    """Nature-skill compact journal style with editable vector text."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.2,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.8,
        "axes.edgecolor": COLOR_DARK,
        "axes.labelcolor": COLOR_DARK,
        "xtick.color": COLOR_DARK,
        "ytick.color": COLOR_DARK,
        "text.color": COLOR_DARK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def save_manuscript_figure(fig, stem: str) -> None:
    """Export editable SVG/PDF plus 600-dpi PNG/TIFF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in FIGURE_FORMATS:
        path = FIGURE_DIR / f"{stem}.{fmt}"
        kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
            "edgecolor": "none",
            "pad_inches": 0.035,
        }
        if fmt in {"png", "tiff", "tif"}:
            kwargs["dpi"] = FIGURE_DPI
        fig.savefig(path, **kwargs)

def _clean_axis(ax, *, grid_x: bool = False, grid_y: bool = False) -> None:
    """Open axes with only very light guides when analytically useful."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_NEUTRAL_DARK)
    ax.spines["bottom"].set_color(COLOR_NEUTRAL_DARK)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=2.5, width=0.7, pad=2,
                   color=COLOR_NEUTRAL_DARK, labelcolor=COLOR_DARK)
    ax.set_axisbelow(True)
    if grid_x:
        ax.grid(axis="x", color=COLOR_PALE, linewidth=0.55, zorder=0)
    if grid_y:
        ax.grid(axis="y", color=COLOR_PALE, linewidth=0.55, zorder=0)

def _panel_title(ax, letter: str, title: str) -> None:
    """Compact Nature-style lower-case panel label plus short title."""
    ax.text(-0.015, 1.055, letter.lower(), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.2, fontweight="bold",
            color=COLOR_DARK, clip_on=False)
    ax.text(0.065, 1.055, title, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.7, fontweight="semibold",
            color=COLOR_DARK, clip_on=False)

def plot_roc_curves(y_true, probability_map: Dict[str, np.ndarray], title: str, stem: str) -> None:
    """Supplementary all-model ROC using a unified low-saturation method family."""
    apply_manuscript_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    models = list(probability_map)
    family = [
        "#484878", "#5F6795", "#7884B4", "#929CC7", "#AAB4D7",
        "#B4C0E4", "#8C7F9B", "#A998AF", "#C0B1C5",
    ]
    for i, model in enumerate(models):
        p = probability_map[model]
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = safe_auc(y_true, p)
        ax.plot(fpr, tpr, lw=1.35, color=family[i % len(family)], alpha=0.92,
                label=f"{model} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle=(0, (2.5, 2.5)), linewidth=0.75,
            color=COLOR_NEUTRAL_LIGHT)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 − specificity",
           ylabel="Sensitivity", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6,
              borderaxespad=0.2, labelspacing=0.35)
    _clean_axis(ax)
    fig.tight_layout(pad=0.8)
    save_manuscript_figure(fig, stem)
    plt.close(fig)

def plot_model_auc_comparison(table: pd.DataFrame, primary_model: str, stem: str) -> None:
    """Development model comparison aligned with the prespecified selection rule.

    Point = mean repeat-level OOF AUROC.
    Line  = observed min-to-max AUROC across outer-CV repeats.
    """
    apply_manuscript_style()
    ordered = table.sort_values(
        ["mean_repeat_OOF_AUC", "averaged_repeated_OOF_Brier"],
        ascending=[True, False],
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.0, max(3.0, 0.34 * len(ordered) + 0.95)))
    for i, row in ordered.iterrows():
        primary = str(row["model"]) == str(primary_model)
        line_color = COLOR_PRIMARY if primary else COLOR_NEUTRAL_LIGHT
        marker_color = COLOR_PRIMARY if primary else COLOR_NEUTRAL
        ax.plot([row["min_repeat_OOF_AUC"], row["max_repeat_OOF_AUC"]], [i, i],
                color=line_color, lw=1.45 if primary else 0.75,
                solid_capstyle="round")
        ax.scatter(row["mean_repeat_OOF_AUC"], i, s=27 if primary else 14,
                   color=marker_color, zorder=3, edgecolor="white", linewidth=0.25)
        if primary:
            ax.text(row["mean_repeat_OOF_AUC"] + 0.008, i,
                    f'{row["mean_repeat_OOF_AUC"]:.3f}', va="center", ha="left",
                    fontsize=6.6, fontweight="semibold", color=COLOR_PRIMARY)
    ax.axvline(0.5, linestyle=(0, (2.5, 2.5)), linewidth=0.7,
               color=COLOR_NEUTRAL_LIGHT)
    ax.set_yticks(np.arange(len(ordered)), ordered["model"])
    ax.set_xlabel("Mean repeat-level OOF AUROC  (range across repeats)")
    ax.set_title("Development model comparison", loc="left", fontweight="semibold")
    _clean_axis(ax, grid_x=True)
    fig.tight_layout(pad=0.8)
    save_manuscript_figure(fig, stem)
    plt.close(fig)

def calibration_bin_table(y_true, p, n_bins: int) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    n_bins = max(3, min(int(n_bins), len(y_true)))
    q = pd.qcut(pd.Series(p), q=n_bins, labels=False, duplicates="drop")
    rows = []
    for bin_id in sorted(pd.Series(q).dropna().unique()):
        mask = np.asarray(q == bin_id)
        n_bin = int(mask.sum())
        events = int(y_true[mask].sum())
        low, high = exact_proportion_ci(events, n_bin)
        rows.append({
            "bin": int(bin_id), "n": n_bin,
            "mean_predicted": float(np.mean(p[mask])),
            "observed": float(np.mean(y_true[mask])),
            "observed_95CI_low": low, "observed_95CI_high": high,
        })
    return pd.DataFrame(rows)


def plot_final_calibration(y_true, p, title: str, metrics: Dict[str, object], stem: str) -> None:
    """Calibration using signal-family points and softer uncertainty bars."""
    apply_manuscript_style()
    n_bins = 5 if len(y_true) <= 80 else 8
    bins = calibration_bin_table(y_true, p, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(4.55, 4.1))
    ax.plot([0, 1], [0, 1], linestyle=(0, (2.5, 2.5)), linewidth=0.75,
            color=COLOR_NEUTRAL_LIGHT)
    if not bins.empty:
        yerr = np.vstack([bins["observed"] - bins["observed_95CI_low"],
                          bins["observed_95CI_high"] - bins["observed"]])
        ax.errorbar(bins["mean_predicted"], bins["observed"], yerr=yerr,
                    fmt="o-", lw=1.15, ms=3.6, capsize=2.0,
                    color=COLOR_PRIMARY, ecolor=COLOR_SOFT, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.3)
    ax.text(0.055, 0.945,
            f"Intercept {metrics['calibration_intercept']:.2f}\n"
            f"Slope {metrics['calibration_slope']:.2f}\n"
            f"Brier {metrics['Brier']:.3f}\n"
            f"n = {len(y_true)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.7,
            linespacing=1.35, color=COLOR_DARK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted MPR probability",
           ylabel="Observed MPR proportion", title=title)
    ax.set_aspect("equal", adjustable="box")
    _clean_axis(ax)
    fig.tight_layout(pad=0.8)
    save_manuscript_figure(fig, stem)
    plt.close(fig)

def extract_linear_effects(
    fitted_pipeline,
    selected_features: Sequence[str],
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> pd.DataFrame:
    """Descriptive penalized/logistic coefficients; not inferential effect estimates."""
    model = fitted_pipeline.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        return pd.DataFrame()
    pre = fitted_pipeline.named_steps["preprocessor"]
    names = [str(x) for x in pre.get_feature_names_out()]
    coef = np.asarray(model.coef_).ravel()
    numeric_selected = [f for f in selected_features if f in numeric]
    categorical_selected = [f for f in selected_features if f in categorical]
    rows = []
    for name, beta in zip(names, coef):
        raw = transformed_to_raw_feature(name, numeric_selected, categorical_selected)
        if raw in numeric_selected:
            interpretation = "OR per 1-SD increase after preprocessing"
        else:
            interpretation = "OR for encoded category versus encoder reference"
        rows.append({
            "transformed_feature": name,
            "raw_feature": raw,
            "raw_feature_label": FEATURE_LABELS.get(raw, raw),
            "model_coefficient": float(beta),
            "descriptive_OR": float(np.exp(beta)),
            "interpretation": interpretation,
            "note": "Penalized/model coefficient; descriptive only, not an unpenalized inferential odds ratio.",
        })
    return pd.DataFrame(rows)


def external_raw_permutation_importance(
    fitted_pipeline,
    valid_df: pd.DataFrame,
    y_valid,
    selected_features: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Exploratory external permutation importance; never used for development."""
    baseline = safe_auc(
        y_valid, pipeline_positive_probability(fitted_pipeline, valid_df[list(selected_features)])
    )
    rng = np.random.default_rng(seed)
    rows = []
    for feature in selected_features:
        drops = []
        for _ in range(PERMUTATION_REPEATS_EXTERNAL):
            perturbed = valid_df[list(selected_features)].copy()
            perturbed[feature] = rng.permutation(perturbed[feature].to_numpy())
            auc_perm = safe_auc(y_valid, pipeline_positive_probability(fitted_pipeline, perturbed))
            if np.isfinite(auc_perm):
                drops.append(baseline - auc_perm)
        arr = np.asarray(drops, dtype=float)
        rows.append({
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "external_baseline_AUC": baseline,
            "mean_external_AUC_drop_when_permuted": float(np.mean(arr)) if len(arr) else np.nan,
            "sd_external_AUC_drop_when_permuted": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
            "note": "Exploratory diagnostic only; external outcomes were never used for feature/model selection.",
        })
    return pd.DataFrame(rows).sort_values("mean_external_AUC_drop_when_permuted", ascending=False)


# =============================================================================
# 13. MAIN
# =============================================================================

def plot_feature_stability(frequency: pd.DataFrame, final_features: Sequence[str], stem: str) -> None:
    """RF-RFE stability using one signal family and neutral subordinate features."""
    apply_manuscript_style()
    table = (frequency.sort_values(["selection_frequency", "mean_selected_rank"],
                                   ascending=[False, True])
             .head(MAIN_FIGURE_MAX_FEATURES)
             .sort_values("selection_frequency", ascending=True)
             .reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(5.0, max(3.1, 0.31 * len(table) + 1.0)))
    for i, row in table.iterrows():
        selected = row["feature"] in final_features
        line_color = COLOR_PRIMARY if selected else COLOR_NEUTRAL_LIGHT
        marker_color = COLOR_PRIMARY if selected else COLOR_NEUTRAL
        ax.hlines(i, 0, row["selection_frequency"], color=line_color,
                  lw=1.35 if selected else 0.7)
        ax.scatter(row["selection_frequency"], i, s=25 if selected else 12,
                   color=marker_color, zorder=3, edgecolor="white", linewidth=0.25)
    ax.axvline(FINAL_STABILITY_SELECTION_FREQUENCY, linestyle=(0, (3, 2)),
               lw=1.15, color=COLOR_ACCENT_DARK)
    ax.text(FINAL_STABILITY_SELECTION_FREQUENCY + 0.012, len(table) - 0.20,
            f"stability lock {FINAL_STABILITY_SELECTION_FREQUENCY:.0%}",
            fontsize=6.0, color=COLOR_ACCENT_DARK, va="top", ha="left")
    ax.set_yticks(np.arange(len(table)), table["feature_label"])
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Outer-fold RF-RFE selection frequency")
    ax.set_title("RF-RFE feature stability", loc="left", fontweight="semibold")
    _clean_axis(ax, grid_x=True)
    fig.tight_layout(pad=0.8)
    save_manuscript_figure(fig, stem)
    plt.close(fig)

def plot_primary_external_roc(y_true, p, model_name: str, stem: str) -> None:
    """Locked external ROC using the same signal family as the complete figure."""
    apply_manuscript_style()
    auc, low, high = bootstrap_auc_ci(y_true, p, n_bootstrap=N_BOOTSTRAP,
                                      seed=RANDOM_STATE + 501)
    fpr, tpr, _ = roc_curve(y_true, p)
    fig, ax = plt.subplots(figsize=(4.55, 4.1))
    ax.plot(fpr, tpr, lw=1.8, color=COLOR_PRIMARY)
    ax.plot([0, 1], [0, 1], linestyle=(0, (2.5, 2.5)), lw=0.75,
            color=COLOR_NEUTRAL_LIGHT)
    ax.text(0.055, 0.945,
            f"AUROC {auc:.3f}\n95% CI {low:.3f}–{high:.3f}\nn = {len(y_true)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
            linespacing=1.35, color=COLOR_DARK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 − specificity",
           ylabel="Sensitivity", title=f"Locked external discrimination — {model_name}")
    ax.set_aspect("equal", adjustable="box")
    _clean_axis(ax)
    fig.tight_layout(pad=0.8)
    save_manuscript_figure(fig, stem)
    plt.close(fig)

def make_figure2_ml_benchmark(
    oof_table: pd.DataFrame,
    frequency: pd.DataFrame,
    final_features: Sequence[str],
    primary_model: str,
    y_valid,
    primary_ext,
    ext_metrics: Dict[str, object],
    stem: str = "Figure2_conventional_ML_benchmark",
) -> None:
    """Nature-skill optimized four-panel manuscript Figure 2.

    a: development-only algorithm selection evidence.
    b: feature-selection stability evidence.
    c: independent external discrimination.
    d: independent external calibration.
    """
    apply_manuscript_style()
    fig = plt.figure(figsize=(7.20, 6.15))  # ~183 mm wide
    gs = fig.add_gridspec(
        2, 2,
        left=0.115, right=0.985, bottom=0.085, top=0.955,
        width_ratios=[1.00, 1.08], height_ratios=[1.00, 1.02],
        wspace=0.44, hspace=0.47,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # a — development model comparison
    ordered = oof_table.sort_values(
        ["mean_repeat_OOF_AUC", "averaged_repeated_OOF_Brier"],
        ascending=[True, False],
    ).reset_index(drop=True)
    for i, row in ordered.iterrows():
        primary = str(row["model"]) == str(primary_model)
        line_color = COLOR_PRIMARY if primary else COLOR_NEUTRAL_LIGHT
        marker_color = COLOR_PRIMARY if primary else COLOR_NEUTRAL
        axA.plot([row["min_repeat_OOF_AUC"], row["max_repeat_OOF_AUC"]], [i, i],
                 color=line_color, lw=1.35 if primary else 0.72,
                 solid_capstyle="round")
        axA.scatter(row["mean_repeat_OOF_AUC"], i, s=23 if primary else 12,
                    color=marker_color, zorder=3, edgecolor="white", linewidth=0.25)
        if primary:
            axA.text(row["mean_repeat_OOF_AUC"] + 0.008, i,
                     f'{row["mean_repeat_OOF_AUC"]:.3f}', va="center", ha="left",
                     fontsize=6.35, fontweight="semibold", color=COLOR_PRIMARY)
    axA.axvline(0.5, linestyle=(0, (2.5, 2.5)), lw=0.7,
                color=COLOR_NEUTRAL_LIGHT)
    axA.set_yticks(np.arange(len(ordered)), ordered["model"])
    axA.set_xlabel("Mean repeat-level OOF AUROC\n(range across repeats)")
    _panel_title(axA, "a", "Development model comparison")
    _clean_axis(axA, grid_x=True)

    # b — feature stability
    stab = (frequency.sort_values(["selection_frequency", "mean_selected_rank"],
                                  ascending=[False, True])
            .head(MAIN_FIGURE_MAX_FEATURES)
            .sort_values("selection_frequency", ascending=True)
            .reset_index(drop=True))
    for i, row in stab.iterrows():
        selected = row["feature"] in final_features
        line_color = COLOR_PRIMARY if selected else COLOR_NEUTRAL_LIGHT
        marker_color = COLOR_PRIMARY if selected else COLOR_NEUTRAL
        axB.hlines(i, 0, row["selection_frequency"], color=line_color,
                   lw=1.30 if selected else 0.65)
        axB.scatter(row["selection_frequency"], i, s=21 if selected else 10,
                    color=marker_color, zorder=3, edgecolor="white", linewidth=0.22)
    axB.axvline(FINAL_STABILITY_SELECTION_FREQUENCY, linestyle=(0, (3, 2)),
                lw=1.10, color=COLOR_ACCENT_DARK)
    axB.text(FINAL_STABILITY_SELECTION_FREQUENCY + 0.012, len(stab) - 0.12,
             f"stability lock {FINAL_STABILITY_SELECTION_FREQUENCY:.0%}",
             fontsize=5.9, color=COLOR_ACCENT_DARK, va="top", ha="left")
    axB.set_yticks(np.arange(len(stab)), stab["feature_label"])
    axB.set_xlim(0, 1.04)
    axB.set_xlabel("Outer-fold selection frequency")
    _panel_title(axB, "b", "RF-RFE feature stability")
    _clean_axis(axB, grid_x=True)

    # c — external discrimination
    auc, low, high = bootstrap_auc_ci(y_valid, primary_ext,
                                      n_bootstrap=N_BOOTSTRAP,
                                      seed=RANDOM_STATE + 501)
    fpr, tpr, _ = roc_curve(y_valid, primary_ext)
    axC.plot(fpr, tpr, lw=1.75, color=COLOR_PRIMARY)
    axC.plot([0, 1], [0, 1], linestyle=(0, (2.5, 2.5)), lw=0.7,
             color=COLOR_NEUTRAL_LIGHT)
    axC.text(0.055, 0.945,
             f"AUROC {auc:.3f}\n95% CI {low:.3f}–{high:.3f}\nn = {len(y_valid)}",
             transform=axC.transAxes, va="top", ha="left", fontsize=6.35,
             linespacing=1.35, color=COLOR_DARK,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.4))
    axC.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 − specificity", ylabel="Sensitivity")
    axC.set_aspect("equal", adjustable="box")
    _panel_title(axC, "c", "Locked external discrimination")
    _clean_axis(axC)

    # d — external calibration
    bins = calibration_bin_table(y_valid, primary_ext,
                                 n_bins=5 if len(y_valid) <= 80 else 8)
    axD.plot([0, 1], [0, 1], linestyle=(0, (2.5, 2.5)), lw=0.7,
             color=COLOR_NEUTRAL_LIGHT)
    if not bins.empty:
        yerr = np.vstack([bins["observed"] - bins["observed_95CI_low"],
                          bins["observed_95CI_high"] - bins["observed"]])
        axD.errorbar(bins["mean_predicted"], bins["observed"], yerr=yerr,
                     fmt="o-", lw=1.08, ms=3.3, capsize=1.8,
                     color=COLOR_PRIMARY, ecolor=COLOR_SOFT, elinewidth=0.82,
                     markeredgecolor="white", markeredgewidth=0.25)
    axD.text(0.055, 0.945,
             f"Intercept {ext_metrics['calibration_intercept']:.2f}\n"
             f"Slope {ext_metrics['calibration_slope']:.2f}\n"
             f"Brier {ext_metrics['Brier']:.3f}\n"
             f"n = {len(y_valid)}",
             transform=axD.transAxes, va="top", ha="left", fontsize=6.2,
             linespacing=1.35, color=COLOR_DARK,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.4))
    axD.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted MPR probability",
            ylabel="Observed MPR proportion")
    axD.set_aspect("equal", adjustable="box")
    _panel_title(axD, "d", "Locked external calibration")
    _clean_axis(axD)

    save_manuscript_figure(fig, stem)
    plt.close(fig)

def main() -> None:
    start = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    apply_manuscript_style()
    save_json(software_versions(), OUTDIR / "software_versions.json")

    method_text = f"""MPR >=90% prediction: prespecified analysis workflow

Outcome: pathological necrosis >= {MPR_CUTOFF:.0%}.
Development center: {TRAIN_CENTER}. Locked external validation center: {VALID_CENTER}.
Prediction profile: {PREDICTION_PROFILE}.

Primary sequence:
1. Prespecified temporal/leakage exclusions are applied before modelling.
2. Within EVERY outer-training fold, perform outcome-independent candidate QC and redundancy pruning.
3. Within that same outer-training fold, tune the RF selector by inner CV and run grouped-MDI RF-RFE under a conservative effective-df budget motivated by EPV>={EPV_MIN_PER_PARAMETER}.
4. Tune each candidate prediction algorithm by inner CV and predict only the untouched outer-validation fold.
5. Aggregate repeated OOF performance; select the primary algorithm using development data only.
6. Lock final features from outer-fold RF-RFE selection frequency >= {FINAL_STABILITY_SELECTION_FREQUENCY:.0%}, subject to the full-development df budget.
7. Derive the operating threshold from the locked primary model's averaged development OOF probabilities only.
8. Refit on the full development cohort and evaluate the locked external center. External outcomes never influence feature selection, hyperparameter tuning, threshold selection, or primary-model selection.

Important reporting note: the df budget is a conservative anti-overfitting design constraint motivated by the EPV heuristic; it is not asserted to be an exact effective-degrees-of-freedom theory for every nonlinear learner.
"""
    (OUTDIR / "FEATURE_SELECTION_METHOD.txt").write_text(method_text, encoding="utf-8")
    pd.DataFrame([
        {"feature": x, "reason": "hard_excluded_postoperative_followup"}
        for x in sorted(POSTOPERATIVE_FOLLOWUP_RAW_COLUMNS | POSTOPERATIVE_DERIVED_COLUMNS)
    ]).to_csv(OUTDIR / "hard_excluded_postoperative_followup_features.csv", index=False)

    print("Loading data, mapping aliases, engineering preoperative predictors, and defining MPR >=90%...")
    source, analysis, train, valid = load_binary_mpr_data()
    y_train = train["mpr_binary"].astype(int).to_numpy()
    y_valid = valid["mpr_binary"].astype(int).to_numpy()
    print("Development outcome counts:", {"non_MPR": int((y_train == 0).sum()), "MPR": int((y_train == 1).sum())})
    print("External outcome counts:", {"non_MPR": int((y_valid == 0).sum()), "MPR": int((y_valid == 1).sum())})

    # Full-development X-only screening is for the final locked model and audits.
    final_numeric, final_categorical, quality = validate_candidate_features(
        train,
        numeric_candidates=NUMERIC_CANDIDATES,
        categorical_candidates=CATEGORICAL_CANDIDATES,
        output_path=OUTDIR / "candidate_feature_quality_control.csv",
        context="full_development_final_model",
    )
    final_numeric, final_categorical, redundancy = prune_candidate_redundancy(
        train,
        final_numeric,
        final_categorical,
        output_path=OUTDIR / "candidate_redundancy_pruning_audit.csv",
        context="full_development_final_model",
    )
    print("Final-model eligible numeric features:", final_numeric)
    print("Final-model eligible categorical features:", final_categorical)
    run_binary_data_integrity_audit(analysis, train, valid, final_numeric, final_categorical)

    full_budget = task_feature_budget(y_train)
    pd.DataFrame([full_budget]).to_csv(OUTDIR / "EPV10_feature_budget_audit.csv", index=False)
    print(f"Conservative full-development effective-df budget: {full_budget['effective_df_budget']}")

    # Strict nested performance estimation starts from the full prespecified candidate pool;
    # data-dependent X-only QC/redundancy filtering is repeated inside each outer-training fold.
    print("\nPreparing strict repeated outer folds with fold-local screening and nested RF-RFE...")
    fold_cache, selection_history = prepare_outer_fold_cache(
        train_df=train,
        y=y_train,
        numeric=NUMERIC_CANDIDATES,
        categorical=CATEGORICAL_CANDIDATES,
    )
    frequency = build_selection_frequency(selection_history)

    print("\nRunning nested hyperparameter optimisation for candidate models...")
    model_results = {}
    summary_rows = []
    tuning_tables = []
    repeat_tables = []
    for model_name in MODEL_NAMES:
        model_start = time.time()
        print(f"  {model_name} ...", flush=True)
        result = run_repeated_nested_oof_for_model(
            model_name=model_name,
            train_df=train,
            y=y_train,
            numeric=NUMERIC_CANDIDATES,
            categorical=CATEGORICAL_CANDIDATES,
            fold_cache=fold_cache,
        )
        model_results[model_name] = result
        summary_rows.append(summarize_oof_result(result, y_train))
        tuning_tables.append(result["tuning_history"])
        repeat_tables.append(result["repeat_metrics"])
        print(f"    mean repeated OOF AUC={summary_rows[-1]['mean_repeat_OOF_AUC']:.3f}; elapsed={(time.time() - model_start)/60:.1f} min")

    oof_table = pd.DataFrame(summary_rows).sort_values("mean_repeat_OOF_AUC", ascending=False).reset_index(drop=True)
    oof_table.to_csv(OUTDIR / "development_nested_OOF_model_comparison.csv", index=False)
    pd.concat(tuning_tables, ignore_index=True).to_csv(OUTDIR / "nested_hyperparameter_tuning_history.csv", index=False)
    pd.concat(repeat_tables, ignore_index=True).to_csv(OUTDIR / "repeat_level_OOF_performance.csv", index=False)

    primary_model, selection_reason = select_primary_model(oof_table)
    print(f"\nLOCKED PRIMARY MODEL (development only): {primary_model}")
    print(selection_reason)

    final_features, final_feature_table, feature_audit = lock_final_features(
        train_df=train,
        y=y_train,
        numeric=final_numeric,
        categorical=final_categorical,
        frequency_table=frequency,
    )
    print("\nFINAL STABILITY-LOCKED FEATURES:")
    for feature in final_features:
        freq = float(frequency.loc[frequency["feature"] == feature, "selection_frequency"].iloc[0]) if (frequency["feature"] == feature).any() else 0.0
        print(f"  {feature}: frequency={freq:.1%}; df={raw_feature_df_cost(train, feature, final_numeric)}")

    primary_oof = np.asarray(model_results[primary_model]["averaged_oof"], dtype=float)
    locked_threshold = choose_threshold(y_train, primary_oof, THRESHOLD_POLICY)

    external_rows = []
    final_param_rows = []
    external_probabilities = {}
    fitted_models = {}
    for model_name in MODEL_NAMES:
        fitted, best_params, inner_best = fit_final_tuned_model(
            model_name=model_name,
            train_df=train,
            y=y_train,
            selected_features=final_features,
            numeric=final_numeric,
            categorical=final_categorical,
            seed=RANDOM_STATE + 700000 + MODEL_COMPLEXITY_RANK.get(model_name, 0),
        )
        fitted_models[model_name] = fitted
        p_ext = pipeline_positive_probability(fitted, valid[final_features])
        external_probabilities[model_name] = p_ext
        auc, low, high = bootstrap_auc_ci(
            y_valid, p_ext, n_bootstrap=N_BOOTSTRAP,
            seed=RANDOM_STATE + 500 + MODEL_COMPLEXITY_RANK.get(model_name, 0),
        )
        external_rows.append({
            "model": model_name,
            "selected_primary_by_development_only": model_name == primary_model,
            "external_AUC": auc,
            "external_AUC_95CI_low": low,
            "external_AUC_95CI_high": high,
            "external_PR_AUC": safe_average_precision(y_valid, p_ext),
            "external_Brier": float(brier_score_loss(y_valid, p_ext)),
            "external_Brier_skill": brier_skill_score(y_valid, p_ext),
            "external_results_role": "descriptive" if model_name != primary_model else "locked_primary_external_validation",
        })
        final_param_rows.append({
            "model": model_name,
            "inner_CV_best_AUC_full_development": inner_best,
            "best_params_json": json.dumps(best_params, ensure_ascii=False, default=_json_default),
            "final_features": "|".join(final_features),
            "final_effective_df": selected_feature_df_cost(train, final_features, final_numeric),
        })

    external_table = pd.DataFrame(external_rows).sort_values("external_AUC", ascending=False)
    external_table.to_csv(OUTDIR / "external_validation_all_models_descriptive.csv", index=False)
    pd.DataFrame(final_param_rows).to_csv(OUTDIR / "final_full_development_tuned_hyperparameters.csv", index=False)
    combined = oof_table.merge(external_table, on="model", how="left")
    combined["PRIMARY_MODEL_LOCKED_FROM_DEVELOPMENT"] = combined["model"] == primary_model
    combined.to_csv(OUTDIR / "PRIMARY_REPORT_OOF_AND_EXTERNAL_AUC.csv", index=False)

    primary_fitted = fitted_models[primary_model]
    primary_ext = external_probabilities[primary_model]
    dev_metrics = binary_metrics(y_train, primary_oof, locked_threshold, "development_averaged_repeated_OOF", primary_model)
    ext_metrics = binary_metrics(y_valid, primary_ext, locked_threshold, "external_locked_validation", primary_model)
    pd.DataFrame([dev_metrics, ext_metrics]).to_csv(OUTDIR / "locked_primary_model_detailed_performance.csv", index=False)

    # Patient-level predictions for downstream RFS analyses. Development predictions are OOF;
    # external predictions come from the locked final model.
    dev_pred = train.copy()
    dev_pred["dataset"] = "development"
    dev_pred["prediction_origin"] = "averaged_repeated_OOF"
    dev_pred["predicted_probability_MPR"] = primary_oof
    dev_pred["predicted_MPR_at_locked_threshold"] = (primary_oof >= locked_threshold).astype(int)
    ext_pred = valid.copy()
    ext_pred["dataset"] = "external"
    ext_pred["prediction_origin"] = "locked_final_model"
    ext_pred["predicted_probability_MPR"] = primary_ext
    ext_pred["predicted_MPR_at_locked_threshold"] = (primary_ext >= locked_threshold).astype(int)
    pd.concat([dev_pred, ext_pred], ignore_index=True, sort=False).to_csv(OUTDIR / "predictions_MPR90_for_RFS.csv", index=False)

    # Interpretability is explicitly descriptive/exploratory.
    linear_effects = extract_linear_effects(primary_fitted, final_features, final_numeric, final_categorical)
    if len(linear_effects):
        linear_effects.to_csv(OUTDIR / "locked_primary_model_logistic_coefficients_DESCRIPTIVE.csv", index=False)
    ext_perm = external_raw_permutation_importance(primary_fitted, valid, y_valid, final_features, seed=RANDOM_STATE + 9876)
    ext_perm.to_csv(OUTDIR / "locked_primary_model_external_permutation_importance_EXPLORATORY.csv", index=False)

    joblib.dump({
        "model_name": primary_model,
        "model": primary_fitted,
        "features": final_features,
        "numeric_features": [f for f in final_features if f in final_numeric],
        "categorical_features": [f for f in final_features if f in final_categorical],
        "MPR_cutoff": MPR_CUTOFF,
        "probability_threshold": locked_threshold,
        "threshold_policy": THRESHOLD_POLICY,
        "train_center": TRAIN_CENTER,
        "valid_center": VALID_CENTER,
        "prediction_profile": PREDICTION_PROFILE,
        "random_state": RANDOM_STATE,
    }, OUTDIR / "locked_primary_MPR90_model.joblib")

    # Publication-grade plots: vector + high-resolution raster.
    plot_model_auc_comparison(oof_table, primary_model, "Fig2A_development_model_comparison")
    plot_feature_stability(frequency, final_features, "Fig2B_RF_RFE_feature_stability")
    plot_primary_external_roc(y_valid, primary_ext, primary_model, "Fig2C_locked_external_ROC")
    plot_final_calibration(y_train, primary_oof, f"Development OOF calibration — {primary_model}", dev_metrics, "Supplement_development_OOF_calibration")
    plot_final_calibration(y_valid, primary_ext, f"External calibration — {primary_model}", ext_metrics, "Fig2D_locked_external_calibration")
    plot_roc_curves(y_valid, external_probabilities, "External validation — all candidate models (descriptive)", "Supplement_external_ROC_all_models")
    make_figure2_ml_benchmark(oof_table, frequency, final_features, primary_model, y_valid, primary_ext, ext_metrics)

    best_oof_row = oof_table.loc[oof_table["model"] == primary_model].iloc[0]
    best_ext_row = external_table.loc[external_table["model"] == primary_model].iloc[0]
    readme_lines = [
        "MPR >=90% BINARY PREDICTION — LOCKED PRIMARY SUMMARY",
        "===================================================",
        f"Development center: {TRAIN_CENTER}",
        f"External validation center: {VALID_CENTER}",
        f"Outcome: pathological necrosis >= {MPR_CUTOFF:.0%}",
        f"Prediction profile: {PREDICTION_PROFILE}",
        f"Strict outer-fold X-only screening: {STRICT_OUTER_FOLD_SCREENING}",
        f"Conservative df budget basis: {FEATURE_BUDGET_BASIS}; EPV heuristic={EPV_MIN_PER_PARAMETER}",
        f"Final effective-df budget: {feature_audit['effective_df_budget']}",
        f"Locked primary model: {primary_model}",
        f"Development mean repeated OOF AUC: {best_oof_row['mean_repeat_OOF_AUC']:.3f}",
        f"Development averaged repeated OOF AUC: {best_oof_row['averaged_repeated_OOF_AUC']:.3f} ({best_oof_row['averaged_repeated_OOF_AUC_95CI_low']:.3f}–{best_oof_row['averaged_repeated_OOF_AUC_95CI_high']:.3f})",
        f"External locked AUC: {best_ext_row['external_AUC']:.3f} ({best_ext_row['external_AUC_95CI_low']:.3f}–{best_ext_row['external_AUC_95CI_high']:.3f})",
        f"Locked threshold policy: {THRESHOLD_POLICY}; threshold={locked_threshold:.6f}",
        "",
        "FINAL FEATURES:",
    ]
    for feature in final_features:
        freq = float(frequency.loc[frequency["feature"] == feature, "selection_frequency"].iloc[0]) if (frequency["feature"] == feature).any() else 0.0
        readme_lines.append(f"- {feature} ({FEATURE_LABELS.get(feature, feature)}); selection_frequency={freq:.1%}")
    readme_lines += [
        "", "PRIMARY MODEL SELECTION RULE:", selection_reason, "",
        "INTERPRETATION BOUNDARIES:",
        "- External performances of non-primary models are descriptive and cannot alter the locked primary model.",
        "- External permutation importance is exploratory only.",
        "- Development threshold-based metrics reuse the development OOF probabilities from which the threshold was selected; external threshold metrics are the confirmatory operating-point evaluation.",
        "- The EPV-inspired df budget is a conservative complexity constraint, not a universal ML degrees-of-freedom theorem.",
        "- RFS/status were not used to include/exclude patients from the MPR prediction cohort.",
    ]
    (OUTDIR / "PRIMARY_RESULTS_READ_ME.txt").write_text("\n".join(readme_lines), encoding="utf-8")

    manifest = {
        "input_xlsx": INPUT_XLSX,
        "input_sheet": INPUT_SHEET,
        "train_center": TRAIN_CENTER,
        "valid_center": VALID_CENTER,
        "MPR_cutoff": MPR_CUTOFF,
        "prediction_profile": PREDICTION_PROFILE,
        "random_state": RANDOM_STATE,
        "outer_splits": N_OUTER_SPLITS,
        "outer_repeats": N_OUTER_REPEATS,
        "inner_splits": N_INNER_SPLITS,
        "strict_outer_fold_screening": STRICT_OUTER_FOLD_SCREENING,
        "EPV_min_per_parameter": EPV_MIN_PER_PARAMETER,
        "feature_budget_basis": FEATURE_BUDGET_BASIS,
        "candidate_models": MODEL_NAMES,
        "locked_primary_model": primary_model,
        "locked_threshold": locked_threshold,
        "threshold_policy": THRESHOLD_POLICY,
        "final_features": final_features,
        "model_selection_reason": selection_reason,
        "external_nonprimary_models_role": "descriptive_only",
        "external_permutation_importance_role": "exploratory_only",
        "cuda_used": CUDA_AVAILABLE,
        "software_versions": software_versions(),
        "input_sha256": file_sha256(Path(INPUT_XLSX)) if Path(INPUT_XLSX).exists() else None,
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "runtime_minutes": (time.time() - start) / 60,
    }
    save_json(manifest, OUTDIR / "analysis_manifest.json")

    print(f"\nOutputs saved to: {OUTDIR.resolve()}")
    print("Main table: PRIMARY_REPORT_OOF_AND_EXTERNAL_AUC.csv")
    print("Primary performance: locked_primary_model_detailed_performance.csv")
    print("Final features: final_selected_features.csv")
    print("Main Figure 2: figures/Figure2_conventional_ML_benchmark.[png|pdf|svg]")
    print(f"Total runtime: {(time.time() - start)/60:.1f} min")


if __name__ == "__main__":
    main()

from __future__ import annotations

import inspect
import json
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils import check_random_state


# -------------------------------------------------------------------
# walkability_predictive_spatial_ml.py
# Spatially Validated Prediction of Perceived Walkability and Voluntary Walking in Kigali
# Using Participatory Micro-Audit Data
# -------------------------------------------------------------------


# -----------------------------
# Config
# -----------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default


def _env_list_str(name: str, default: str) -> List[str]:
    s = _env(name, default).strip()
    return [x.strip() for x in s.split(",") if x.strip()]


def _env_list_int(name: str, default: str) -> List[int]:
    out: List[int] = []
    for x in _env_list_str(name, default):
        try:
            out.append(int(x))
        except Exception:
            pass
    return out


def _env_list_float(name: str, default: str) -> List[float]:
    out: List[float] = []
    for x in _env_list_str(name, default):
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


DATA_PATH = _env("WALKABILITY_DATA_PATH", "kigali_walkability_clean_wide.csv")
OUT_DIR = _env("WALKABILITY_OUT_DIR", "./outputs")

FIG_DIR = os.path.join(OUT_DIR, "figures")
MAP_DIR = os.path.join(OUT_DIR, "maps")
PAPER_DIR = os.path.join(OUT_DIR, "paper")
PAPER_TABLES_DIR = os.path.join(PAPER_DIR, "tables")

RANDOM_STATE = _env_int("RANDOM_STATE", 42)

N_OUTER_SPLITS = _env_int("N_OUTER_SPLITS", 5)
N_INNER_SPLITS = _env_int("N_INNER_SPLITS", 4)
N_ITER_TUNE = _env_int("N_ITER_TUNE", 30)

GRID_SIZES_M = _env_list_int("GRID_SIZES_M", "250,500,1000")

PRIMARY_SPLITS = _env_list_str(
    "PRIMARY_SPLITS",
    "polygon_block,leave_one_polygon_out,leave_one_area_out,grid_500m",
)
SECONDARY_SPLITS = _env_list_str(
    "SECONDARY_SPLITS",
    "respondent_group,random",
)

CAL_METHOD = _env("CAL_METHOD", "sigmoid").strip().lower()
PROB_CALIB_FRAC = _env_float("PROB_CALIB_FRAC", 0.15)
CONFORMAL_CALIB_FRAC = _env_float("CONFORMAL_CALIB_FRAC", 0.20)
CONFORMAL_ALPHA = _env_float("CONFORMAL_ALPHA", 0.10)

TARGETING_TOP_FRACS = _env_list_float("TARGETING_TOP_FRACS", "0.01,0.05,0.10")
MAX_POINTS_MAP = _env_int("MAX_POINTS_MAP", 8000)

ABLATIONS = _env_list_str("ABLATIONS", "full,no_location,location_only")
MODEL_CANDIDATES = _env_list_str(
    "MODEL_CANDIDATES",
    "logreg_en,hgb,random_forest,extra_trees,gradient_boosting",
)

CLUSTER_COL = _env("CLUSTER_COL", "respondent_id")
LAT_COL = _env("LAT_COL", "observation_lat")
LON_COL = _env("LON_COL", "observation_lon")
STUDY_AREA_COL = _env("STUDY_AREA_COL", "study_area_id")

Y_PERCEPTION = _env("Y_PERCEPTION", "perception_code")
Y_CHOICE = _env("Y_CHOICE", "walk_choice")

PLACE_ID_COL = _env("PLACE_ID_COL", "").strip()
POLYGON_PATH = _env("KIGALI_POLYGON_PATH", "").strip()
POLYGON_ID_COL = _env("KIGALI_POLYGON_ID_COL", "poly_id").strip()
DECISION_UNIT_COL = _env("DECISION_UNIT_COL", "").strip()

ORACLE_REGISTRY_PATH = os.path.join(OUT_DIR, "oracle_best_models.csv")
ORACLE_MANIFEST_PATH = os.path.join(OUT_DIR, "oracle_winner_manifest.json")


# -----------------------------
# IO helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    alt = os.path.join("/mnt/data", os.path.basename(path))
    if os.path.exists(alt):
        return alt
    return path


def safe_json_dumps(d: Dict[str, Any]) -> str:
    def _conv(x: Any) -> Any:
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, pd.Series):
            return x.tolist()
        return x

    return json.dumps({k: _conv(v) for k, v in d.items()}, ensure_ascii=False)


# -----------------------------
# General safety helpers
# -----------------------------

def unique_non_nan_classes(y: Sequence[Any]) -> np.ndarray:
    return np.sort(pd.Series(np.asarray(y)).dropna().astype(int).unique())


def has_at_least_two_classes(y: Sequence[Any]) -> bool:
    return len(unique_non_nan_classes(y)) >= 2


def safe_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def safe_calibration_curve(y_true: np.ndarray, p_hat: np.ndarray, n_bins: int = 10):
    y_true = np.asarray(y_true, dtype=int)
    p_hat = np.asarray(p_hat, dtype=float)
    if len(y_true) == 0 or not np.isfinite(p_hat).any():
        return np.array([]), np.array([])
    try:
        return calibration_curve(y_true, p_hat, n_bins=n_bins, strategy="quantile")
    except Exception:
        try:
            return calibration_curve(
                y_true,
                p_hat,
                n_bins=min(5, max(2, len(y_true))),
                strategy="uniform",
            )
        except Exception:
            return np.array([]), np.array([])


# -----------------------------
# CV utilities + leakage checks
# -----------------------------

def get_stratified_group_kfold(n_splits: int, random_state: int):
    try:
        from sklearn.model_selection import StratifiedGroupKFold  # type: ignore

        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    except Exception:
        return None


def make_splits_random(y: np.ndarray, n_splits: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    if len(y) < 2:
        return []
    counts = pd.Series(y).value_counts()
    safe_splits = int(min(n_splits, max(2, counts.min()))) if not counts.empty else int(min(n_splits, len(y)))
    safe_splits = min(safe_splits, len(y))
    X_dummy = np.zeros((len(y), 1))
    if counts.empty or counts.min() < 2 or safe_splits < 2:
        kf = KFold(
            n_splits=min(max(2, min(n_splits, len(y))), len(y)),
            shuffle=True,
            random_state=random_state,
        )
        return [(tr, te) for tr, te in kf.split(X_dummy)]
    skf = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=random_state)
    return [(tr, te) for tr, te in skf.split(X_dummy, y)]


def make_splits_group_stratified(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_groups = pd.Series(groups).dropna().nunique()
    if len(y) < 2 or n_groups < 2:
        return []
    safe_splits = int(min(n_splits, n_groups))
    safe_splits = max(2, safe_splits) if n_groups >= 2 else safe_splits
    X_dummy = np.zeros((len(y), 1))
    sgkf = get_stratified_group_kfold(n_splits=safe_splits, random_state=random_state)
    if sgkf is not None:
        try:
            return [(tr, te) for tr, te in sgkf.split(X_dummy, y, groups)]
        except Exception:
            pass
    gkf = GroupKFold(n_splits=safe_splits)
    return [(tr, te) for tr, te in gkf.split(X_dummy, y, groups)]


def make_splits_leave_one_group_out(
    groups: np.ndarray,
    min_test: int = 30,
    min_train: int = 80,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique = pd.unique(pd.Series(groups).dropna())
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for g in unique:
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        if len(te) < min_test or len(tr) < min_train:
            continue
        splits.append((tr, te))
    return splits


def leakage_check_no_group_overlap(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    name: str,
) -> None:
    for i, (tr, te) in enumerate(splits, start=1):
        gtr = set(pd.Series(groups[tr]).dropna().astype(str).unique().tolist())
        gte = set(pd.Series(groups[te]).dropna().astype(str).unique().tolist())
        inter = gtr.intersection(gte)
        if inter:
            raise RuntimeError(
                f"[LEAKAGE] Split '{name}' fold {i} has group overlap (first few): {list(sorted(inter))[:5]}"
            )

# -----------------------------
# Spatial utilities (meter grids)
# -----------------------------

def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    is_south = lat < 0
    return (32700 if is_south else 32600) + zone


def project_lonlat_to_xy_m(df: pd.DataFrame, lon_col: str, lat_col: str) -> Optional[pd.DataFrame]:
    if lon_col not in df.columns or lat_col not in df.columns:
        return None

    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    ok = lon.notna() & lat.notna()
    if ok.sum() == 0:
        return None

    try:
        from pyproj import Transformer  # type: ignore
    except Exception:
        return None

    lon0 = float(lon[ok].mean())
    lat0 = float(lat[ok].mean())
    epsg = _utm_epsg_from_lonlat(lon0, lat0)

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x = np.full(len(df), np.nan, dtype=float)
    y = np.full(len(df), np.nan, dtype=float)
    xx, yy = transformer.transform(lon[ok].to_numpy(), lat[ok].to_numpy())
    x[ok.to_numpy()] = xx
    y[ok.to_numpy()] = yy

    out = pd.DataFrame({"_x_m": x, "_y_m": y}, index=df.index)
    out.attrs["utm_epsg"] = epsg
    return out


def make_meter_grid_ids(df_xy: pd.DataFrame, cell_m: int) -> pd.Series:
    x = pd.to_numeric(df_xy["_x_m"], errors="coerce")
    y = pd.to_numeric(df_xy["_y_m"], errors="coerce")
    a = np.floor(x / float(cell_m)).astype("Int64")
    b = np.floor(y / float(cell_m)).astype("Int64")
    return (a.astype(str) + "_" + b.astype(str)).astype("category")


# -----------------------------
# Polygon grouping (optional)
# -----------------------------

def try_assign_polygon_groups(
    df: pd.DataFrame,
    polygon_path: str,
    lat_col: str,
    lon_col: str,
    polygon_id_col: str,
) -> Optional[pd.Series]:
    if not polygon_path:
        return None
    polygon_path = resolve_path(polygon_path)
    if not os.path.exists(polygon_path):
        print(f"[WARN] Polygon file not found: {polygon_path}")
        return None
    if lat_col not in df.columns or lon_col not in df.columns:
        print("[WARN] No lat/lon columns; cannot assign polygons.")
        return None

    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import Point  # type: ignore
    except Exception as e:
        print(f"[WARN] geopandas/shapely not available; skipping polygon holdout. ({e})")
        return None

    d = df[[lat_col, lon_col]].copy()
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    ok = d[lat_col].notna() & d[lon_col].notna()
    if ok.sum() == 0:
        return None

    gdf_pts = gpd.GeoDataFrame(
        d.loc[ok].copy(),
        geometry=[Point(xy) for xy in zip(d.loc[ok, lon_col], d.loc[ok, lat_col])],
        crs="EPSG:4326",
    )
    polys = gpd.read_file(polygon_path)
    if polys.empty or "geometry" not in polys.columns:
        print("[WARN] Polygon file has no geometries.")
        return None

    if polygon_id_col not in polys.columns:
        polys = polys.copy()
        polys[polygon_id_col] = np.arange(len(polys)).astype(int).astype(str)

    if polys.crs is None:
        polys = polys.set_crs("EPSG:4326")
    if polys.crs != gdf_pts.crs:
        polys = polys.to_crs(gdf_pts.crs)

    try:
        joined = gpd.sjoin(gdf_pts, polys[[polygon_id_col, "geometry"]], how="left", predicate="within")
    except TypeError:
        joined = gpd.sjoin(gdf_pts, polys[[polygon_id_col, "geometry"]], how="left", op="within")

    out = pd.Series(index=df.index, dtype="object")
    out.loc[joined.index] = joined[polygon_id_col].astype(str).values
    return out


# -----------------------------
# Data + feature inference
# -----------------------------

def load_data(path: str) -> pd.DataFrame:
    path = resolve_path(path)
    df = pd.read_csv(path)

    if Y_PERCEPTION not in df.columns and "perception" in df.columns:
        df[Y_PERCEPTION] = df["perception"].map({"Good": 0, "Concern": 1, "Problem": 2})
    if Y_CHOICE not in df.columns and "walk_decision" in df.columns:
        df[Y_CHOICE] = (df["walk_decision"].astype(str).str.lower() == "choice").astype(int)

    return df


def infer_feature_blocks(df: pd.DataFrame) -> Dict[str, List[str]]:
    issue_cols = [c for c in df.columns if c.startswith("issue_")]
    issue_bin: List[str] = []
    for c in issue_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            vals = set(pd.to_numeric(df[c], errors="coerce").dropna().unique().tolist())
            if vals.issubset({0, 1}):
                issue_bin.append(c)

    context_cat_candidates = [
        "respondent_gender",
        "respondent_mobility",
        "respondent_age_group",
        "walk_purpose",
        "walk_company",
        "walk_familiarity",
        "weather_main",
        "start_weekday",
        STUDY_AREA_COL,
    ]
    context_num_candidates = [
        "temperature_c",
        "start_hour",
        "start_month",
        "start_day",
    ]

    context_cat = [c for c in context_cat_candidates if c in df.columns]
    context_num = [c for c in context_num_candidates if c in df.columns]
    location_cols = [c for c in [LAT_COL, LON_COL, STUDY_AREA_COL] if c in df.columns]

    def dedupe(seq: List[str]) -> List[str]:
        return list(dict.fromkeys(seq))

    issue_cols = dedupe(issue_bin)
    context_cols = dedupe([c for c in (context_cat + context_num) if c not in location_cols])
    full_predictors = dedupe(issue_cols + context_cols + location_cols)

    return {
        "issue_cols": issue_cols,
        "context_cols": context_cols,
        "location_cols": location_cols,
        "full_predictors": full_predictors,
    }


def build_ablation_predictors(blocks: Dict[str, List[str]]) -> Dict[str, List[str]]:
    full = blocks["full_predictors"]
    location = blocks["location_cols"]
    issue = blocks["issue_cols"]
    no_location = [c for c in full if c not in location]
    location_only = location[:] if location else full[:]
    issue_only = issue[:] if issue else full[:]
    no_issue = [c for c in full if c not in issue]
    context_only = blocks["context_cols"][:] if blocks["context_cols"] else full[:]

    abls = {
        "full": full,
        "no_location": no_location,
        "location_only": location_only,
        "issue_only": issue_only,
        "no_issue": no_issue,
        "context_only": context_only,
    }
    return {k: v for k, v in abls.items() if k in ABLATIONS}


# -----------------------------
# Preprocessing + model specs
# -----------------------------

def make_onehot(sparse: bool) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def build_preprocessor(cat_cols: List[str], num_cols: List[str], sparse_ohe: bool, scale_num: bool) -> ColumnTransformer:
    transformers = []
    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("onehot", make_onehot(sparse=sparse_ohe)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if num_cols:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_num:
            steps.append(("scaler", StandardScaler(with_mean=False)))
        transformers.append(("num", Pipeline(steps=steps), num_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _to_dense_if_needed(X):
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(X):
            return X.toarray()
    except Exception:
        pass
    return X


def get_feature_names(pre: ColumnTransformer) -> List[str]:
    try:
        return [str(x) for x in pre.get_feature_names_out()]
    except Exception:
        names: List[str] = []
        for name, trans, cols in getattr(pre, "transformers_", []):
            if name == "remainder":
                continue
            if isinstance(cols, (list, tuple)):
                for c in cols:
                    names.append(str(c))
            else:
                names.append(str(cols))
        return names


def parse_original_feature_from_ohe_name(feature_name: str) -> str:
    s = str(feature_name)
    if "__" in s:
        left, right = s.split("__", 1)
        if left == "num":
            return right
        if left == "cat":
            bits = right.split("_")
            if len(bits) <= 1:
                return right
            return "_".join(bits[:-1]) if len(bits) > 1 else right
        return right
    return s


class OrdinalThresholdClassifier(ClassifierMixin, BaseEstimator):
    """
    Ordinal classifier via cumulative binary threshold models:
      P(Y > c), c = 0, ..., K-2

    Important implementation detail:
    this wrapper is *parameter-transparent* to scikit-learn so that
    Pipeline.set_params(model__C=..., model__max_depth=..., etc.) works
    exactly as if the base estimator were exposed directly. This is needed
    for nested tuning with ordinal models.
    """

    _estimator_type = "classifier"

    def __init__(self, base_estimator: BaseEstimator):
        self.base_estimator = base_estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        try:
            tags.estimator_type = "classifier"
        except Exception:
            pass
        return tags

    def get_params(self, deep: bool = True):
        # Critical for sklearn.clone(): when deep=False, only constructor
        # parameters may be returned. Exposing nested transparent parameters
        # here makes clone() try to pass them into __init__().
        params = {"base_estimator": self.base_estimator}
        if not deep:
            return params
        if hasattr(self.base_estimator, "get_params"):
            base_params = self.base_estimator.get_params(deep=True)
            # Expose transparent names so Pipeline.set_params(model__C=...)
            # works on the wrapper.
            # base_estimator__* aliases here because sklearn.clone() may
            # propagate them back through constructor kwargs.
            for k, v in base_params.items():
                params[k] = v
        return params

    def set_params(self, **params):
        if not params:
            return self

        base_estimator = params.pop("base_estimator", None)
        if base_estimator is not None:
            self.base_estimator = base_estimator

        if not params:
            return self

        base_params = {}
        direct_params = {}
        base_valid = set()
        if hasattr(self.base_estimator, "get_params"):
            base_valid = set(self.base_estimator.get_params(deep=True).keys())

        for k, v in params.items():
            if k.startswith("base_estimator__"):
                base_params[k.split("base_estimator__", 1)[1]] = v
            elif k in base_valid:
                base_params[k] = v
            else:
                direct_params[k] = v

        if base_params:
            self.base_estimator.set_params(**base_params)
        for k, v in direct_params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.classes_ = np.sort(pd.Series(y).dropna().unique())
        self.threshold_models_: List[BaseEstimator] = []
        self.constant_class_: Optional[int] = None

        if len(self.classes_) <= 1:
            self.constant_class_ = int(self.classes_[0]) if len(self.classes_) == 1 else 0
            self.thresholds_ = []
            return self

        self.thresholds_ = self.classes_[:-1].tolist()

        for thr in self.thresholds_:
            y_bin = (y > int(thr)).astype(int)
            est = clone(self.base_estimator)
            uniq = np.unique(y_bin)
            if len(uniq) < 2:
                est = DummyClassifier(strategy="constant", constant=int(uniq[0]))
            est.fit(X, y_bin)
            self.threshold_models_.append(est)

        return self

    def predict_proba(self, X):
        n = len(X)
        K = len(self.classes_)

        if self.constant_class_ is not None:
            proba = np.zeros((n, K), dtype=float)
            idx = int(np.where(self.classes_ == int(self.constant_class_))[0][0])
            proba[:, idx] = 1.0
            return proba

        cum = np.zeros((n, K - 1), dtype=float)
        for j, est in enumerate(self.threshold_models_):
            p = est.predict_proba(X)
            if p.ndim == 1:
                p1 = p
            else:
                cls = getattr(est, "classes_", np.array([0, 1]))
                cls = np.asarray(cls).astype(int)
                if len(cls) == 1:
                    p1 = np.ones(n, dtype=float) if int(cls[0]) == 1 else np.zeros(n, dtype=float)
                else:
                    pos_idx = int(np.where(cls == 1)[0][0])
                    p1 = p[:, pos_idx]
            cum[:, j] = np.clip(p1, 1e-8, 1 - 1e-8)

        # enforce monotone non-increasing cumulative probabilities
        for j in range(1, cum.shape[1]):
            cum[:, j] = np.minimum(cum[:, j - 1], cum[:, j])

        proba = np.zeros((n, K), dtype=float)
        proba[:, 0] = 1.0 - cum[:, 0]
        for j in range(1, K - 1):
            proba[:, j] = cum[:, j - 1] - cum[:, j]
        proba[:, K - 1] = cum[:, K - 2]

        proba = np.clip(proba, 1e-12, 1.0)
        s = proba.sum(axis=1, keepdims=True)
        s[s <= 0] = 1.0
        return proba / s

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


@dataclass
class ModelSpec:
    name: str
    builder: Any
    param_distributions: List[Dict[str, Any]]
    needs_dense: bool
    sparse_ohe: bool
    scale_num: bool
    ordinal_ok: bool = True


def make_model_specs(random_state: int) -> List[ModelSpec]:
    def logreg_builder():
        return LogisticRegression(
            solver="saga",
            max_iter=12000,
            tol=1e-3,
            n_jobs=None,
            random_state=random_state,
        )

    def hgb_builder():
        return HistGradientBoostingClassifier(
            random_state=random_state,
            early_stopping=True,
        )

    def rf_builder():
        return RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        )

    def et_builder():
        return ExtraTreesClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
        )

    def gb_builder():
        return GradientBoostingClassifier(
            random_state=random_state,
        )

    logreg_params = [
        {"model__C": np.logspace(-3, 2, 20), "model__penalty": ["l2"], "model__class_weight": [None, "balanced"]},
        {"model__C": np.logspace(-3, 2, 20), "model__penalty": ["l1"], "model__class_weight": [None, "balanced"]},
        {
            "model__C": np.logspace(-3, 2, 20),
            "model__penalty": ["elasticnet"],
            "model__l1_ratio": np.linspace(0.0, 1.0, 6),
            "model__class_weight": [None, "balanced"],
        },
    ]

    hgb_params = [
        {
            "model__learning_rate": np.logspace(math.log10(0.01), math.log10(0.2), 12),
            "model__max_depth": [2, 3, 4, 5],
            "model__max_iter": [200, 400, 700],
            "model__min_samples_leaf": [10, 20, 50, 100],
            "model__l2_regularization": np.logspace(-6, -1, 10),
            "model__max_bins": [64, 128, 255],
        }
    ]

    rf_params = [
        {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 5, 10, 20],
            "model__max_features": ["sqrt", 0.5, 0.75],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        }
    ]

    et_params = [
        {
            "model__n_estimators": [400, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 5, 10, 20],
            "model__max_features": ["sqrt", 0.5, 0.75],
            "model__class_weight": [None, "balanced"],
        }
    ]

    gb_params = [
        {
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__n_estimators": [100, 200, 400],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_leaf": [5, 10, 20],
        }
    ]

    specs = [
        ModelSpec("logreg_en", logreg_builder, logreg_params, False, True, True, True),
        ModelSpec("hgb", hgb_builder, hgb_params, True, True, False, True),
        ModelSpec("random_forest", rf_builder, rf_params, False, True, False, True),
        ModelSpec("extra_trees", et_builder, et_params, False, True, False, True),
        ModelSpec("gradient_boosting", gb_builder, gb_params, True, True, False, True),
    ]
    return [s for s in specs if s.name in MODEL_CANDIDATES]


def make_pipeline(
    cat_cols: List[str],
    num_cols: List[str],
    spec: ModelSpec,
    is_multiclass: bool = False,
    ordinal: bool = False,
) -> Pipeline:
    pre = build_preprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
        sparse_ohe=spec.sparse_ohe,
        scale_num=spec.scale_num,
    )

    steps: List[Tuple[str, Any]] = [("pre", pre)]
    if spec.needs_dense:
        steps.append(("to_dense", FunctionTransformer(_to_dense_if_needed, accept_sparse=True)))

    model_obj = spec.builder()
    if ordinal:
        model_obj = OrdinalThresholdClassifier(model_obj)

    steps.append(("model", model_obj))
    return Pipeline(steps=steps)


def fit_pipeline_safe(pipeline: Pipeline, X_fit: pd.DataFrame, y_fit: np.ndarray) -> Tuple[Pipeline, str]:
    model = clone(pipeline)
    classes = unique_non_nan_classes(y_fit)

    if len(classes) == 0:
        raise ValueError("No non-missing class labels in training fold.")

    if len(classes) == 1:
        only_class = int(classes[0])
        dummy_pipe = clone(pipeline)
        dummy_pipe.set_params(model=DummyClassifier(strategy="constant", constant=only_class))
        dummy_pipe.fit(X_fit, y_fit)
        return dummy_pipe, "dummy_constant"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_fit, y_fit)
        return model, "native"
    except ValueError as e:
        msg = str(e).lower()
        if ("at least 2 classes" not in msg) and ("class" not in msg):
            raise
        only_class = int(classes[0])
        dummy_pipe = clone(pipeline)
        dummy_pipe.set_params(model=DummyClassifier(strategy="constant", constant=only_class))
        dummy_pipe.fit(X_fit, y_fit)
        return dummy_pipe, "dummy_constant"


# -----------------------------
# Calibration utilities (robust to sklearn version + one-class partitions)
# -----------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def estimator_classes(estimator: BaseEstimator, n_fallback: Optional[int] = None) -> np.ndarray:
    try:
        return np.asarray(estimator.classes_, dtype=int)
    except Exception:
        try:
            return np.asarray(estimator.named_steps["model"].classes_, dtype=int)
        except Exception:
            if n_fallback is None:
                return np.asarray([], dtype=int)
            return np.arange(n_fallback, dtype=int)


class IdentityProbabilityCalibrator(BaseEstimator):
    def __init__(self, fitted_estimator: BaseEstimator):
        self.fitted_estimator = fitted_estimator
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X=None, y=None):
        self.classes_ = estimator_classes(self.fitted_estimator)
        return self

    def predict_proba(self, X):
        return self.fitted_estimator.predict_proba(X)


class PrefitProbabilityCalibrator(BaseEstimator):
    """
    Robust fallback prefit calibrator.
    - Binary: sigmoid / isotonic on p(class=1)
    - Multiclass: OvR per class, renormalized
    - One-class or class-absent calibration targets fall back to identity for that task
    """

    def __init__(self, fitted_estimator: BaseEstimator, method: str = "sigmoid"):
        self.fitted_estimator = fitted_estimator
        self.method = method
        self.classes_: Optional[np.ndarray] = None
        self._calibrators: Optional[List[Tuple[str, Any]]] = None

    def _fit_binary_map(self, p: np.ndarray, y: np.ndarray) -> Tuple[str, Any]:
        y = np.asarray(y, dtype=int)
        p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
        if len(y) == 0 or len(np.unique(y)) < 2:
            return ("identity", None)

        method = (self.method or "sigmoid").lower().strip()
        if method == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression

                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(p, y.astype(float))
                return ("isotonic", ir)
            except Exception:
                method = "sigmoid"

        z = _logit(p).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=2000)
        lr.fit(z, y)
        return ("sigmoid", lr)

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        proba = self.fitted_estimator.predict_proba(X)
        K = proba.shape[1]
        self.classes_ = estimator_classes(self.fitted_estimator, n_fallback=K)
        calibrators: List[Tuple[str, Any]] = []

        if K == 2:
            kind, obj = self._fit_binary_map(proba[:, 1], y)
            calibrators.append((kind, obj))
            self._calibrators = calibrators
            return self

        for k in range(K):
            yk = (y == int(self.classes_[k])).astype(int)
            kind, obj = self._fit_binary_map(proba[:, k], yk)
            calibrators.append((kind, obj))
        self._calibrators = calibrators
        return self

    def predict_proba(self, X):
        proba = np.clip(self.fitted_estimator.predict_proba(X).astype(float), 1e-8, 1 - 1e-8)
        K = proba.shape[1]
        if self._calibrators is None:
            return proba

        if K == 2:
            kind, cal = self._calibrators[0]
            p = proba[:, 1]
            if kind == "identity":
                p1 = p
            elif kind == "sigmoid":
                z = _logit(p).reshape(-1, 1)
                p1 = cal.predict_proba(z)[:, 1]
            else:
                p1 = cal.predict(p)
            p1 = np.clip(p1, 1e-8, 1 - 1e-8)
            return np.column_stack([1 - p1, p1])

        out = np.zeros_like(proba)
        for k in range(K):
            kind, cal = self._calibrators[k]
            pk = proba[:, k]
            if kind == "identity":
                out[:, k] = pk
            elif kind == "sigmoid":
                z = _logit(pk).reshape(-1, 1)
                out[:, k] = cal.predict_proba(z)[:, 1]
            else:
                out[:, k] = cal.predict(pk)
        out = np.clip(out, 1e-12, 1.0)
        s = out.sum(axis=1, keepdims=True)
        s[s <= 0] = 1.0
        return out / s


def _contains_ordinal_threshold_estimator(estimator: BaseEstimator) -> bool:
    if isinstance(estimator, OrdinalThresholdClassifier):
        return True
    try:
        if hasattr(estimator, "named_steps"):
            for step in estimator.named_steps.values():
                if isinstance(step, OrdinalThresholdClassifier):
                    return True
    except Exception:
        pass
    return False


def calibrate_prefit(
    fitted_estimator: BaseEstimator,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    method: str,
) -> BaseEstimator:
    method = (method or "sigmoid").lower().strip()
    if method not in ("sigmoid", "isotonic"):
        method = "sigmoid"

    y_cal = np.asarray(y_cal, dtype=int)
    if len(y_cal) == 0 or len(np.unique(y_cal)) < 2:
        return IdentityProbabilityCalibrator(fitted_estimator).fit(X_cal, y_cal)

    # Custom ordinal wrapper is safest with the custom fallback calibrator.
    if _contains_ordinal_threshold_estimator(fitted_estimator):
        return PrefitProbabilityCalibrator(fitted_estimator, method=method).fit(X_cal, y_cal)

    try:
        from sklearn.frozen import FrozenEstimator  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = CalibratedClassifierCV(
                estimator=FrozenEstimator(fitted_estimator),
                method=method,
            )
            cal.fit(X_cal, y_cal)
            return cal
    except Exception:
        pass

    try:
        sig = inspect.signature(CalibratedClassifierCV)
        kwargs: Dict[str, Any] = {"method": method, "cv": "prefit"}
        if "estimator" in sig.parameters:
            kwargs["estimator"] = fitted_estimator
        elif "base_estimator" in sig.parameters:
            kwargs["base_estimator"] = fitted_estimator
        else:
            raise RuntimeError("Unsupported CalibratedClassifierCV signature")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = CalibratedClassifierCV(**kwargs)
            cal.fit(X_cal, y_cal)
            return cal
    except Exception:
        return PrefitProbabilityCalibrator(fitted_estimator, method=method).fit(X_cal, y_cal)


def expected_calibration_error(y_true: np.ndarray, p_hat: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_hat, dtype=float), 1e-8, 1 - 1e-8)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y[m]))
        conf = float(np.mean(p[m]))
        ece += float(np.mean(m)) * abs(acc - conf)
    return float(ece)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    y = y_true.astype(int)
    Y = np.zeros((len(y), n_classes), dtype=float)
    Y[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def multiclass_ece_macro(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    y = y_true.astype(int)
    K = proba.shape[1]
    eces = []
    for k in range(K):
        yk = (y == k).astype(int)
        eces.append(expected_calibration_error(yk, proba[:, k], n_bins=n_bins))
    return float(np.mean(eces)) if eces else float("nan")


def calibration_slope_intercept_logistic(y_true: np.ndarray, p_hat: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_hat, dtype=float)
    ok = np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) < 3 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    z = _logit(p).reshape(-1, 1)
    try:
        lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=2000)
        lr.fit(z, y)
        return float(lr.intercept_[0]), float(lr.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


# -----------------------------
# Conformal prediction (split conformal)
# -----------------------------

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(scores)
    n = len(s)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(s[k])


def conformal_sets_from_proba(proba: np.ndarray, qhat: float) -> List[List[int]]:
    thr = 1.0 - float(qhat)
    sets: List[List[int]] = []
    for i in range(proba.shape[0]):
        ks = np.where(proba[i, :] >= thr)[0].tolist()
        if len(ks) == 0:
            ks = [int(np.argmax(proba[i, :]))]
        sets.append(ks)
    return sets


# -----------------------------
# Nested tuning utilities
# -----------------------------

def _align_proba_classes(proba: np.ndarray, classes_: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if len(classes_) == 0:
        if proba.shape[1] == n_classes:
            return proba.astype(float)
        if proba.shape[1] == 1 and n_classes == 2:
            out[:, 1] = proba[:, 0]
            out[:, 0] = 1.0 - out[:, 1]
        return out
    for j, c in enumerate(classes_):
        ci = int(c)
        if 0 <= ci < n_classes and j < proba.shape[1]:
            out[:, ci] = proba[:, j]
    if n_classes == 2 and proba.shape[1] == 1 and len(classes_) == 1:
        c0 = int(classes_[0])
        if c0 == 0:
            out[:, 1] = 1.0 - out[:, 0]
        elif c0 == 1:
            out[:, 0] = 1.0 - out[:, 1]
    return out


def _sample_param_grid(
    rng: np.random.RandomState,
    grids: List[Dict[str, Any]],
    n_iter: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if len(grids) == 0:
        return out
    for _ in range(n_iter):
        g = grids[int(rng.randint(0, len(grids)))]
        params: Dict[str, Any] = {}
        for k, v in g.items():
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                params[k] = v[int(rng.randint(0, len(v)))]
            else:
                params[k] = v
        out.append(params)
    return out


def _inner_cv_score(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    is_multiclass: bool,
    n_classes: int,
) -> float:
    scores: List[float] = []
    for tr, te in splits:
        m = clone(pipeline)
        try:
            m, _ = fit_pipeline_safe(m, X.iloc[tr], y[tr])
            proba = m.predict_proba(X.iloc[te])
            cls = estimator_classes(m, n_fallback=proba.shape[1])
            proba = _align_proba_classes(proba, cls, n_classes=n_classes)
            if is_multiclass:
                ll = log_loss(y[te], proba, labels=list(range(n_classes)))
            else:
                ll = log_loss(y[te], proba, labels=[0, 1])
            scores.append(-float(ll))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else float("-inf")


def tune_pipeline_nested(
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    inner_splits: List[Tuple[np.ndarray, np.ndarray]],
    param_candidates: List[Dict[str, Any]],
    is_multiclass: bool,
    n_classes: int,
) -> Tuple[Pipeline, Dict[str, Any], float]:
    best_score = float("-inf")
    best_params: Dict[str, Any] = {}
    best_model: Optional[Pipeline] = None

    if len(param_candidates) == 0:
        best_model = clone(base_pipeline)
        best_score = _inner_cv_score(
            best_model,
            X_train,
            y_train,
            inner_splits,
            is_multiclass=is_multiclass,
            n_classes=n_classes,
        )
        return best_model, best_params, best_score

    for params in param_candidates:
        m = clone(base_pipeline)
        m.set_params(**params)
        score = _inner_cv_score(
            m,
            X_train,
            y_train,
            inner_splits,
            is_multiclass=is_multiclass,
            n_classes=n_classes,
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_model = m

    if best_model is None:
        best_model = clone(base_pipeline)
        best_score = _inner_cv_score(
            best_model,
            X_train,
            y_train,
            inner_splits,
            is_multiclass=is_multiclass,
            n_classes=n_classes,
        )

    return best_model, best_params, best_score


# -----------------------------
# Disjoint splitting inside outer-train (fit / prob-cal / conformal-cal)
# -----------------------------

def _threeway_index_split(
    idx: np.ndarray,
    prob_calib_frac: float,
    conformal_frac: float,
    rng: np.random.RandomState,
    min_each: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = rng.permutation(np.asarray(idx, dtype=int))
    n = len(perm)
    if n <= 3:
        return perm, np.array([], dtype=int), np.array([], dtype=int)

    n_prob = max(1, int(round(prob_calib_frac * n)))
    n_conf = max(1, int(round(conformal_frac * n)))
    if n_prob + n_conf >= n:
        n_prob = max(1, min(n_prob, n - 2))
        n_conf = max(1, min(n_conf, n - n_prob - 1))

    if n >= 3 * min_each:
        n_prob = max(min_each, n_prob)
        n_conf = max(min_each, n_conf)
        if n_prob + n_conf >= n - min_each:
            n_prob = min(n_prob, max(1, n - min_each - 1))
            n_conf = min(n_conf, max(1, n - n_prob - min_each))

    prob_idx = perm[:n_prob]
    conf_idx = perm[n_prob : n_prob + n_conf]
    fit_idx = perm[n_prob + n_conf :]
    return fit_idx, prob_idx, conf_idx


def split_train_threeway_by_groups(
    train_idx: np.ndarray,
    groups: np.ndarray,
    y_train_full: np.ndarray,
    prob_calib_frac: float,
    conformal_frac: float,
    rng: np.random.RandomState,
    min_each: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = pd.Series(groups[train_idx]).astype(str).fillna("MISSING")
    uniq = g.unique().tolist()
    rng.shuffle(uniq)

    n = len(uniq)
    if n >= 3:
        n_prob = max(1, int(round(prob_calib_frac * n)))
        n_conf = max(1, int(round(conformal_frac * n)))
        if n_prob + n_conf >= n:
            n_prob = max(1, min(n_prob, n - 2))
            n_conf = max(1, min(n_conf, n - n_prob - 1))

        g_prob = set(uniq[:n_prob])
        g_conf = set(uniq[n_prob : n_prob + n_conf])
        g_fit = set(uniq[n_prob + n_conf :])

        fit_idx = train_idx[g.isin(g_fit).to_numpy()]
        prob_idx = train_idx[g.isin(g_prob).to_numpy()]
        conf_idx = train_idx[g.isin(g_conf).to_numpy()]

        if len(fit_idx) >= 1 and len(prob_idx) >= 1 and len(conf_idx) >= 1:
            return fit_idx, prob_idx, conf_idx

    return _threeway_index_split(
        np.asarray(train_idx, dtype=int),
        prob_calib_frac=prob_calib_frac,
        conformal_frac=conformal_frac,
        rng=rng,
        min_each=min_each,
    )


def select_inner_groups_for_protocol(
    df: pd.DataFrame,
    split_name: str,
    split_groups: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if split_groups is not None:
        return np.asarray(split_groups)

    if split_name.startswith("grid_") and split_name in df.columns:
        return df[split_name].astype(str).to_numpy()
    if split_name in {"polygon_block", "leave_one_polygon_out"} and "_poly_id" in df.columns:
        return df["_poly_id"].astype(str).to_numpy()
    if split_name == "leave_one_area_out" and STUDY_AREA_COL in df.columns:
        return df[STUDY_AREA_COL].astype(str).to_numpy()
    if split_name == "place_group" and PLACE_ID_COL and PLACE_ID_COL in df.columns:
        return df[PLACE_ID_COL].astype(str).to_numpy()
    if split_name == "respondent_group" and CLUSTER_COL in df.columns:
        return df[CLUSTER_COL].astype(str).to_numpy()

    if PLACE_ID_COL and PLACE_ID_COL in df.columns and df[PLACE_ID_COL].notna().any():
        return df[PLACE_ID_COL].astype(str).to_numpy()
    if "_poly_id" in df.columns and df["_poly_id"].notna().any():
        return df["_poly_id"].astype(str).to_numpy()
    if "_grid_250m" in df.columns and df["_grid_250m"].notna().any():
        return df["_grid_250m"].astype(str).to_numpy()
    if CLUSTER_COL in df.columns and df[CLUSTER_COL].notna().any():
        return df[CLUSTER_COL].astype(str).to_numpy()

    return None


# -----------------------------
# Decision-unit targeting
# -----------------------------

def unit_targeting_metrics(
    unit_df: pd.DataFrame,
    y_col: str,
    risk_col: str,
    top_frac: float,
    weight_col: str = "n_points",
) -> Dict[str, float]:
    d = unit_df.dropna(subset=[risk_col, y_col]).copy()
    if d.empty:
        return {
            "top_frac": top_frac,
            "n_units": 0,
            "precision_at_k": np.nan,
            "recall_at_k": np.nan,
            "lift_at_k": np.nan,
            "base_rate": np.nan,
        }

    w = pd.to_numeric(d[weight_col], errors="coerce").fillna(1.0).to_numpy()
    y = pd.to_numeric(d[y_col], errors="coerce").fillna(0.0).to_numpy()
    r = pd.to_numeric(d[risk_col], errors="coerce").fillna(0.0).to_numpy()

    n_units = len(d)
    k = max(1, int(np.ceil(top_frac * n_units)))
    order = np.argsort(-r)
    sel = np.zeros(n_units, dtype=bool)
    sel[order[:k]] = True

    base = float(np.sum(w * y) / np.sum(w)) if np.sum(w) > 0 else np.nan
    prec = float(np.sum(w[sel] * y[sel]) / np.sum(w[sel])) if np.sum(w[sel]) > 0 else np.nan
    rec = float(np.sum(w[sel] * y[sel]) / np.sum(w * y)) if np.sum(w * y) > 0 else np.nan
    lift = float(prec / base) if (base is not None and base > 0) else np.nan

    return {
        "top_frac": float(top_frac),
        "n_units": int(n_units),
        "k_units": int(k),
        "base_rate": base,
        "precision_at_k": prec,
        "recall_at_k": rec,
        "lift_at_k": lift,
    }


def make_unit_table(
    df_points: pd.DataFrame,
    unit_col: str,
    y_bin: np.ndarray,
    risk: np.ndarray,
    baseline_issue_count: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    d = df_points.copy()
    d["_y"] = y_bin.astype(int)
    d["_risk"] = np.asarray(risk, dtype=float)

    agg: Dict[str, Any] = {"_y": "mean", "_risk": "mean", "row_id": "count"}
    if baseline_issue_count is not None:
        d["_issue"] = baseline_issue_count.astype(float)
        agg["_issue"] = "mean"

    g = d.groupby(unit_col, dropna=False).agg(agg).rename(columns={"row_id": "n_points"})
    g = g.reset_index().rename(columns={"_y": "y_rate", "_risk": "risk_mean", "_issue": "issue_mean"})
    return g


# -----------------------------
# Plotting (reliability)
# -----------------------------

def plot_reliability_binary(y_true: np.ndarray, p_hat: np.ndarray, title: str, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    frac_pos, mean_pred = safe_calibration_curve(y_true, p_hat, n_bins=10)
    if len(mean_pred) == 0:
        return
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()


def plot_reliability_ovr(y_true: np.ndarray, proba: np.ndarray, focus_class: int, title: str, out_png: str) -> None:
    y_bin = (y_true.astype(int) == int(focus_class)).astype(int)
    p = proba[:, int(focus_class)]
    plot_reliability_binary(y_bin, p, title, out_png)


# -----------------------------
# Maps (OOF mosaic + deployment-only full fit)
# -----------------------------

def export_risk_map_folium(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    risk_col: str,
    unc_col: Optional[str],
    out_html: str,
    title: str,
    max_points: int = 8000,
    banner: Optional[str] = None,
) -> None:
    try:
        import folium  # type: ignore
    except Exception:
        print("[WARN] folium not installed; skipping map export.")
        return

    cols = [lat_col, lon_col, risk_col] + ([unc_col] if unc_col else [])
    d = df[cols].copy()
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d[risk_col] = pd.to_numeric(d[risk_col], errors="coerce")
    if unc_col:
        d[unc_col] = pd.to_numeric(d[unc_col], errors="coerce")
    d = d.dropna(subset=[lat_col, lon_col, risk_col])
    if d.empty:
        return
    if len(d) > max_points:
        d = d.sample(n=max_points, random_state=RANDOM_STATE)

    lat0 = float(d[lat_col].mean())
    lon0 = float(d[lon_col].mean())
    m = folium.Map(location=[lat0, lon0], zoom_start=12, control_scale=True)

    if banner:
        folium.Marker(
            [lat0, lon0],
            popup=folium.Popup(banner, max_width=450),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    qs = d[risk_col].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()

    def _color(v: float) -> str:
        if v <= qs[0.05]:
            return "#2c7bb6"
        if v <= qs[0.25]:
            return "#abd9e9"
        if v <= qs[0.5]:
            return "#ffffbf"
        if v <= qs[0.75]:
            return "#fdae61"
        return "#d7191c"

    for _, row in d.iterrows():
        risk = float(row[risk_col])
        txt = f"{title}<br>risk={risk:.3f}"
        if unc_col:
            txt += f"<br>unc={float(row[unc_col]):.3f}"
        folium.CircleMarker(
            location=[float(row[lat_col]), float(row[lon_col])],
            radius=3,
            color=_color(risk),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(txt, max_width=250),
        ).add_to(m)

    ensure_dir(os.path.dirname(out_html))
    m.save(out_html)


# -----------------------------
# Permutation importance (fold-wise + stability)
# -----------------------------

def permutation_importance_by_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    cat_cols: List[str],
    num_cols: List[str],
    spec: ModelSpec,
    scoring: str,
    outer_groups: Optional[np.ndarray] = None,
    inner_groups: Optional[np.ndarray] = None,
    is_multiclass: bool = False,
    ordinal: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = check_random_state(RANDOM_STATE)
    n_classes = int(len(np.unique(y))) if is_multiclass else 2

    for f, (tr, te) in enumerate(outer_splits, start=1):
        tr = np.asarray(tr, dtype=int)
        te = np.asarray(te, dtype=int)
        X_tr = X.iloc[tr]
        y_tr = y[tr]

        if inner_groups is not None:
            g_tr = inner_groups[tr]
            inner_splits = make_splits_group_stratified(
                y_tr,
                g_tr,
                n_splits=min(N_INNER_SPLITS, max(2, pd.Series(g_tr).nunique())),
                random_state=RANDOM_STATE + f,
            )
        else:
            inner_splits = make_splits_random(
                y_tr,
                n_splits=min(N_INNER_SPLITS, max(2, len(np.unique(y_tr)))),
                random_state=RANDOM_STATE + f,
            )

        base_pipe = make_pipeline(cat_cols, num_cols, spec, is_multiclass=is_multiclass, ordinal=ordinal)
        candidates = _sample_param_grid(rng, spec.param_distributions, n_iter=max(10, min(N_ITER_TUNE, 20)))

        tuned_pipe, _, _ = tune_pipeline_nested(
            base_pipeline=base_pipe,
            X_train=X_tr,
            y_train=y_tr,
            inner_splits=inner_splits,
            param_candidates=candidates,
            is_multiclass=is_multiclass,
            n_classes=n_classes,
        )

        try:
            tuned_fit, fit_mode = fit_pipeline_safe(clone(tuned_pipe), X_tr, y_tr)
        except Exception:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pi = permutation_importance(
                    tuned_fit,
                    X.iloc[te],
                    y[te],
                    scoring=scoring,
                    n_repeats=8,
                    random_state=RANDOM_STATE + f,
                )
            except Exception:
                continue

        feat_names = list(X.columns)
        imp = pi.importances_mean
        for j in range(min(len(imp), len(feat_names))):
            rows.append(
                {
                    "fold": f,
                    "fit_mode": fit_mode,
                    "feature_name": feat_names[j],
                    "orig_feature": feat_names[j],
                    "importance_mean": float(imp[j]),
                }
            )
    return pd.DataFrame(rows)


def summarize_importance_stability(df_imp: pd.DataFrame) -> pd.DataFrame:
    if df_imp.empty:
        return df_imp
    g = (
        df_imp.groupby("orig_feature", as_index=False)["importance_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g.columns = ["orig_feature", "importance_mean", "importance_std", "n_folds"]
    return g.sort_values("importance_mean", ascending=False)


# -----------------------------
# Split protocol assembly
# -----------------------------

def build_split_protocols(df: pd.DataFrame, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
    protocols: Dict[str, Dict[str, Any]] = {}

    protocols["random"] = {
        "splits": make_splits_random(y, n_splits=N_OUTER_SPLITS, random_state=RANDOM_STATE),
        "groups": None,
        "primary": False,
    }

    if CLUSTER_COL in df.columns and df[CLUSTER_COL].notna().any():
        g = df[CLUSTER_COL].astype(str).to_numpy()
        splits = make_splits_group_stratified(y, g, n_splits=N_OUTER_SPLITS, random_state=RANDOM_STATE)
        leakage_check_no_group_overlap(splits, g, "respondent_group")
        protocols["respondent_group"] = {"splits": splits, "groups": g, "primary": False}

    if PLACE_ID_COL and PLACE_ID_COL in df.columns and df[PLACE_ID_COL].notna().any():
        g = df[PLACE_ID_COL].astype(str).to_numpy()
        splits = make_splits_group_stratified(y, g, n_splits=N_OUTER_SPLITS, random_state=RANDOM_STATE)
        leakage_check_no_group_overlap(splits, g, "place_group")
        protocols["place_group"] = {"splits": splits, "groups": g, "primary": True}

    if "_x_m" in df.columns and df["_x_m"].notna().any():
        for cell in GRID_SIZES_M:
            col = f"_grid_{cell}m"
            if col not in df.columns:
                continue
            g = df[col].astype(str).to_numpy()
            splits = make_splits_group_stratified(y, g, n_splits=N_OUTER_SPLITS, random_state=RANDOM_STATE)
            leakage_check_no_group_overlap(splits, g, col)
            protocols[f"grid_{cell}m"] = {"splits": splits, "groups": g, "primary": True}

    if "_poly_id" in df.columns and df["_poly_id"].notna().any():
        g = df["_poly_id"].astype(str).to_numpy()
        splits = make_splits_group_stratified(y, g, n_splits=N_OUTER_SPLITS, random_state=RANDOM_STATE)
        leakage_check_no_group_overlap(splits, g, "polygon_block")
        protocols["polygon_block"] = {"splits": splits, "groups": g, "primary": True}

        lo = make_splits_leave_one_group_out(g)
        if len(lo) >= 3:
            leakage_check_no_group_overlap(lo, g, "leave_one_polygon_out")
            protocols["leave_one_polygon_out"] = {"splits": lo, "groups": g, "primary": True}

    if STUDY_AREA_COL in df.columns and df[STUDY_AREA_COL].notna().any():
        g = df[STUDY_AREA_COL].astype(str).to_numpy()
        lo = make_splits_leave_one_group_out(g)
        if len(lo) >= 3:
            leakage_check_no_group_overlap(lo, g, "leave_one_area_out")
            protocols["leave_one_area_out"] = {"splits": lo, "groups": g, "primary": True}

    for k in list(protocols.keys()):
        protocols[k]["primary"] = k in PRIMARY_SPLITS

    return protocols


# -----------------------------
# Oracle helpers
# -----------------------------

def build_oracle_registry(perf_df: pd.DataFrame) -> pd.DataFrame:
    if perf_df.empty:
        return pd.DataFrame()

    d = perf_df.copy()
    d = d[~d["model"].astype(str).str.startswith("dummy_")]
    if "ablation" not in d.columns:
        d["ablation"] = d["outcome"].astype(str).map(lambda s: s.split("__", 1)[1] if "__" in s else "full")

    rows: List[Dict[str, Any]] = []
    for (outcome, split, ablation), g in d.groupby(["outcome", "split", "ablation"], dropna=False):
        g = g.copy()
        g["_ll"] = pd.to_numeric(g.get("log_loss"), errors="coerce")
        g["_f1"] = pd.to_numeric(g.get("macro_f1"), errors="coerce")
        g = g.sort_values(["_ll", "_f1"], ascending=[True, False])
        best = g.iloc[0].to_dict()
        rows.append(
            {
                "outcome": outcome,
                "split": split,
                "ablation": ablation,
                "oracle_model": best.get("model"),
                "oracle_log_loss": best.get("log_loss"),
                "oracle_macro_f1": best.get("macro_f1"),
                "oracle_accuracy": best.get("accuracy"),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Paper-facing reporting layer
# -----------------------------

def _metric_priority_cols(is_multiclass: bool) -> List[str]:
    if is_multiclass:
        return ["log_loss", "macro_f1", "brier_mc", "accuracy", "kappa_quadratic", "mae_ordinal"]
    return ["log_loss", "avg_precision", "auc", "brier", "macro_f1", "accuracy"]


def _pick_best_model(
    perf_df: pd.DataFrame,
    outcome: str,
    split: str,
    ablation: str,
    model_names: List[str],
    is_multiclass: bool,
) -> Optional[pd.Series]:
    key = f"{outcome}__{ablation}"
    d = perf_df[(perf_df["outcome"] == key) & (perf_df["split"] == split) & (perf_df["model"].isin(model_names))].copy()
    if d.empty:
        return None
    d["_ll"] = pd.to_numeric(d.get("log_loss"), errors="coerce")
    d["_f1"] = pd.to_numeric(d.get("macro_f1"), errors="coerce")
    d = d.sort_values(["_ll", "_f1"], ascending=[True, False])
    return d.iloc[0]


def generate_paper_report(
    perf_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    targeting_pooled_df: pd.DataFrame,
    targeting_by_fold_df: pd.DataFrame,
    conformal_df: pd.DataFrame,
    out_dir: str,
    fig_dir: str,
) -> None:
    ensure_dir(PAPER_DIR)
    ensure_dir(PAPER_TABLES_DIR)

    perf_df = perf_df.copy()
    fold_df = fold_df.copy()

    tuned_models = sorted([
        m for m in perf_df["model"].unique().tolist()
        if (not str(m).startswith("dummy_")) and (str(m) != "oracle_winner")
    ])

    outcomes_base: List[str] = []
    for o in perf_df["outcome"].astype(str).unique():
        if "__" in o:
            outcomes_base.append(o.split("__", 1)[0])
    outcomes_base = sorted(list(set(outcomes_base)))

    rows: List[Dict[str, Any]] = []
    for base in outcomes_base:
        is_multiclass = base == Y_PERCEPTION
        for split in PRIMARY_SPLITS:
            best = _pick_best_model(perf_df, base, split, "full", tuned_models, is_multiclass=is_multiclass)
            if best is None:
                continue
            row = best.to_dict()
            row["outcome_base"] = base
            row["ablation"] = "full"
            row["primary_split"] = True
            rows.append(row)
    table1 = pd.DataFrame(rows)
    if not table1.empty:
        keep_cols = ["outcome_base", "outcome", "split", "model", "n_eval"]
        for c in _metric_priority_cols(is_multiclass=True):
            if c in table1.columns:
                keep_cols.append(c)
                if f"{c}_ci_lo" in table1.columns and f"{c}_ci_hi" in table1.columns:
                    keep_cols.extend([f"{c}_ci_lo", f"{c}_ci_hi"])
        for c in [
            "auc",
            "avg_precision",
            "brier",
            "ece",
            "cal_intercept",
            "cal_slope",
            "ece_focus",
            "ece_macro",
            "brier_mc",
            "kappa_quadratic",
            "mae_ordinal",
        ]:
            if c in table1.columns and c not in keep_cols:
                keep_cols.append(c)
                if f"{c}_ci_lo" in table1.columns and f"{c}_ci_hi" in table1.columns:
                    keep_cols.extend([f"{c}_ci_lo", f"{c}_ci_hi"])
        keep_cols = [c for c in keep_cols if c in table1.columns]
        table1[keep_cols].to_csv(os.path.join(PAPER_TABLES_DIR, "Table1_primary_performance.csv"), index=False)

    rows2: List[Dict[str, Any]] = []
    for base in outcomes_base:
        is_multiclass = base == Y_PERCEPTION
        for split in PRIMARY_SPLITS:
            for ab in ["full", "no_location", "location_only"]:
                best = _pick_best_model(perf_df, base, split, ab, tuned_models, is_multiclass=is_multiclass)
                if best is None:
                    continue
                r = best.to_dict()
                r["outcome_base"] = base
                r["ablation"] = ab
                rows2.append(r)
    table2 = pd.DataFrame(rows2)
    if not table2.empty:
        keep = ["outcome_base", "outcome", "split", "ablation", "model", "n_eval"]
        for c in ["log_loss", "macro_f1", "accuracy", "brier", "brier_mc", "auc", "avg_precision", "ece", "ece_macro", "ece_focus", "kappa_quadratic", "mae_ordinal"]:
            if c in table2.columns:
                keep.append(c)
                if f"{c}_ci_lo" in table2.columns and f"{c}_ci_hi" in table2.columns:
                    keep.extend([f"{c}_ci_lo", f"{c}_ci_hi"])
        keep = [c for c in keep if c in table2.columns]
        table2[keep].to_csv(os.path.join(PAPER_TABLES_DIR, "Table2_leakage_sensitivity.csv"), index=False)

    if not targeting_by_fold_df.empty:
        d = targeting_by_fold_df[targeting_by_fold_df["split"].isin(PRIMARY_SPLITS)].copy()
        group_cols = ["outcome", "split", "model", "unit_col", "baseline", "top_frac"]
        metric_cols = ["precision_at_k", "recall_at_k", "lift_at_k", "base_rate"]
        agg_rows: List[Dict[str, Any]] = []
        for keys, g in d.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, keys))
            for mc in metric_cols:
                vals = pd.to_numeric(g[mc], errors="coerce").to_numpy()
                row[mc] = float(np.nanmean(vals))
                v = vals[np.isfinite(vals)]
                if len(v) >= 2:
                    row[f"{mc}_ci_lo"] = float(np.quantile(v, 0.025))
                    row[f"{mc}_ci_hi"] = float(np.quantile(v, 0.975))
                else:
                    row[f"{mc}_ci_lo"] = np.nan
                    row[f"{mc}_ci_hi"] = np.nan
            row["n_folds"] = int(g["fold"].nunique())
            agg_rows.append(row)
        pd.DataFrame(agg_rows).to_csv(os.path.join(PAPER_TABLES_DIR, "Table3_targeting_primary.csv"), index=False)

    if not conformal_df.empty:
        c = conformal_df[conformal_df["split"].isin(PRIMARY_SPLITS)].copy()
        group_cols = ["outcome", "split", "model"]
        agg_rows = []
        for keys, g in c.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, keys))
            for mc in ["coverage", "avg_set_size", "median_set_size", "p95_set_size"]:
                vals = pd.to_numeric(g[mc], errors="coerce").to_numpy()
                row[mc] = float(np.nanmean(vals))
                v = vals[np.isfinite(vals)]
                if len(v) >= 2:
                    row[f"{mc}_ci_lo"] = float(np.quantile(v, 0.025))
                    row[f"{mc}_ci_hi"] = float(np.quantile(v, 0.975))
                else:
                    row[f"{mc}_ci_lo"] = np.nan
                    row[f"{mc}_ci_hi"] = np.nan
            row["n_folds"] = int(g["fold"].nunique())
            agg_rows.append(row)
        pd.DataFrame(agg_rows).to_csv(os.path.join(PAPER_TABLES_DIR, "Table4_conformal_primary.csv"), index=False)

    perf_df.to_csv(os.path.join(PAPER_TABLES_DIR, "Appendix_all_models.csv"), index=False)

    fig_rows: List[Dict[str, Any]] = []
    if os.path.isdir(fig_dir):
        for fn in sorted(os.listdir(fig_dir)):
            if fn.startswith("reliability_") and fn.endswith(".png"):
                fig_rows.append(
                    {
                        "figure_file": os.path.join(fig_dir, fn),
                        "type": "reliability_curve",
                        "note": "OOF reliability curve (primary evidence when split is primary spatial holdout).",
                    }
                )
    pd.DataFrame(fig_rows).to_csv(os.path.join(PAPER_DIR, "figures_manifest.csv"), index=False)

    report_lines = [
        "# Reporting summary\n",
        "## What is considered primary evidence\n",
        "- Primary evidence is restricted to **spatially structured holdouts** (polygon-block, leave-one-area/polygon-out, and meter-grid blocks).\n",
        "- Random CV and respondent-group CV are treated as **secondary/diagnostic** and should not be used for generalization claims.\n",
        "\n## Key tables generated\n",
        "- Table1_primary_performance.csv: best tuned model per primary split (ablation=full)\n",
        "- Table2_leakage_sensitivity.csv: full vs no_location vs location_only (memorization sensitivity)\n",
        "- Table3_targeting_primary.csv: decision-unit targeting vs baselines with fold uncertainty\n",
        "- Table4_conformal_primary.csv: conformal coverage + set size by primary split\n",
        "- Appendix_all_models.csv: all splits/models/ablations\n",
        "\n## How to use \n",
        "- Headline performance: Table 1 (primary splits only).\n",
        "- Leakage/memorization argument: Table 2 (location_only is an upper bound; no_location is the conservative scientific model).\n",
        "- Decision analysis: Table 3 (unit-level targeting, compare against issue_count + random).\n",
        "- Uncertainty under spatial shift: Table 4 (coverage stability across held-out areas).\n",
    ]
    with open(os.path.join(PAPER_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


# -----------------------------
# Outer CV runner (nested tuning + strict calibration + conformal)
# -----------------------------

def run_nested_outer_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    row_ids: np.ndarray,
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    outer_groups: Optional[np.ndarray],
    inner_groups: Optional[np.ndarray],
    cat_cols: List[str],
    num_cols: List[str],
    spec: ModelSpec,
    is_multiclass: bool,
    outcome_name: str,
    split_name: str,
    focus_class: int,
    rng: np.random.RandomState,
    ordinal: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_classes = int(len(np.unique(y))) if is_multiclass else 2
    n = len(y)

    oof_proba = np.full((n, n_classes), np.nan, dtype=float)
    oof_fold = np.full(n, np.nan, dtype=float)

    fold_rows: List[Dict[str, Any]] = []
    cal_bins_rows: List[pd.DataFrame] = []
    conf_rows: List[Dict[str, Any]] = []

    base_pipe = make_pipeline(cat_cols, num_cols, spec, is_multiclass=is_multiclass, ordinal=ordinal)

    for f, (tr, te) in enumerate(outer_splits, start=1):
        tr = np.asarray(tr, dtype=int)
        te = np.asarray(te, dtype=int)

        X_tr = X.iloc[tr]
        y_tr = y[tr]
        X_te = X.iloc[te]
        y_te = y[te]

        if inner_groups is not None:
            inner_g = inner_groups[tr]
            try:
                n_inner = min(N_INNER_SPLITS, max(2, pd.Series(inner_g).dropna().nunique()))
                inner_splits = make_splits_group_stratified(
                    y_tr,
                    inner_g,
                    n_splits=n_inner,
                    random_state=RANDOM_STATE + f,
                )
            except Exception:
                inner_splits = make_splits_random(
                    y_tr,
                    n_splits=min(N_INNER_SPLITS, max(2, len(np.unique(y_tr)))),
                    random_state=RANDOM_STATE + f,
                )
        else:
            inner_splits = make_splits_random(
                y_tr,
                n_splits=min(N_INNER_SPLITS, max(2, len(np.unique(y_tr)))),
                random_state=RANDOM_STATE + f,
            )

        candidates = _sample_param_grid(rng, spec.param_distributions, n_iter=N_ITER_TUNE)
        tuned_pipe, tuned_params, inner_score = tune_pipeline_nested(
            base_pipeline=base_pipe,
            X_train=X_tr,
            y_train=y_tr,
            inner_splits=inner_splits,
            param_candidates=candidates,
            is_multiclass=is_multiclass,
            n_classes=n_classes,
        )

        if inner_groups is not None:
            split_groups = inner_groups
        elif outer_groups is not None:
            split_groups = outer_groups
        else:
            split_groups = np.array([str(i) for i in range(len(y))], dtype=object)

        fit_idx, prob_cal_idx, conf_cal_idx = split_train_threeway_by_groups(
            train_idx=tr,
            groups=split_groups,
            y_train_full=y,
            prob_calib_frac=PROB_CALIB_FRAC,
            conformal_frac=CONFORMAL_CALIB_FRAC,
            rng=rng,
            min_each=40 if len(tr) > 250 else 25,
        )

        if len(fit_idx) == 0:
            fit_idx = tr
        if len(prob_cal_idx) == 0:
            prob_cal_idx = fit_idx
        if len(conf_cal_idx) == 0:
            conf_cal_idx = prob_cal_idx

        fitted_uncal, fit_mode = fit_pipeline_safe(clone(tuned_pipe), X.iloc[fit_idx], y[fit_idx])

        method = CAL_METHOD
        if method == "isotonic" and len(prob_cal_idx) < 200:
            method = "sigmoid"
        fitted_cal = calibrate_prefit(fitted_uncal, X.iloc[prob_cal_idx], y[prob_cal_idx], method=method)

        proba_uncal = fitted_uncal.predict_proba(X_te)
        proba_cal = fitted_cal.predict_proba(X_te)

        cls_uncal = estimator_classes(fitted_uncal, n_fallback=proba_uncal.shape[1])
        cls_cal = estimator_classes(fitted_cal, n_fallback=proba_cal.shape[1])

        proba_uncal = _align_proba_classes(proba_uncal, cls_uncal, n_classes=n_classes)
        proba_cal = _align_proba_classes(proba_cal, cls_cal, n_classes=n_classes)

        oof_proba[te, :] = proba_cal
        oof_fold[te] = float(f)

        try:
            if is_multiclass:
                y_bin = (y_te.astype(int) == int(focus_class)).astype(int)
                p = proba_cal[:, int(focus_class)]
            else:
                y_bin = y_te.astype(int)
                p = proba_cal[:, 1]
            frac_pos, mean_pred = safe_calibration_curve(y_bin, p, n_bins=10)
            if len(mean_pred) > 0:
                cal_bins_rows.append(
                    pd.DataFrame(
                        {
                            "outcome": outcome_name,
                            "split": split_name,
                            "model": spec.name,
                            "fold": f,
                            "bin_mean_pred": mean_pred,
                            "bin_frac_pos": frac_pos,
                            "n_test": len(te),
                        }
                    )
                )
        except Exception:
            pass

        try:
            proba_conf = fitted_cal.predict_proba(X.iloc[conf_cal_idx])
            proba_conf = _align_proba_classes(proba_conf, cls_cal, n_classes=n_classes)
            y_conf = y[conf_cal_idx].astype(int)
            valid_rows = (y_conf >= 0) & (y_conf < n_classes)
            y_conf = y_conf[valid_rows]
            proba_conf = proba_conf[valid_rows]
            scores = 1.0 - proba_conf[np.arange(len(y_conf)), y_conf] if len(y_conf) else np.array([], dtype=float)
            qhat = conformal_quantile(scores, alpha=CONFORMAL_ALPHA)
            sets = conformal_sets_from_proba(proba_cal, qhat=qhat)
            covered = np.array([int(y_te[i] in sets[i]) for i in range(len(te))], dtype=int)
            set_sizes = np.array([len(s) for s in sets], dtype=int)
            conf_rows.append(
                {
                    "outcome": outcome_name,
                    "split": split_name,
                    "model": spec.name,
                    "fold": f,
                    "n_test": int(len(te)),
                    "qhat": float(qhat),
                    "coverage": float(np.mean(covered)) if len(covered) else np.nan,
                    "avg_set_size": float(np.mean(set_sizes)) if len(set_sizes) else np.nan,
                    "median_set_size": float(np.median(set_sizes)) if len(set_sizes) else np.nan,
                    "p95_set_size": float(np.quantile(set_sizes, 0.95)) if len(set_sizes) else np.nan,
                }
            )
        except Exception:
            pass

        met: Dict[str, float] = {}
        if is_multiclass:
            pred = np.argmax(proba_cal, axis=1)
            met["accuracy"] = float(accuracy_score(y_te, pred))
            met["macro_f1"] = safe_macro_f1(y_te, pred)
            met["log_loss"] = float(log_loss(y_te, proba_cal, labels=list(range(n_classes))))
            met["brier_mc"] = float(multiclass_brier(y_te, proba_cal, n_classes=n_classes))
            met["kappa_quadratic"] = float(cohen_kappa_score(y_te, pred, weights="quadratic"))
            met["ece_macro"] = float(multiclass_ece_macro(y_te, proba_cal))
            met["ece_focus"] = float(expected_calibration_error((y_te == focus_class).astype(int), proba_cal[:, focus_class]))
            met["mae_ordinal"] = float(mean_absolute_error(y_te, pred))
        else:
            p1 = proba_cal[:, 1]
            pred = (p1 >= 0.5).astype(int)
            met["accuracy"] = float(accuracy_score(y_te, pred))
            met["macro_f1"] = safe_macro_f1(y_te, pred)
            met["log_loss"] = float(log_loss(y_te, proba_cal, labels=[0, 1]))
            met["brier"] = float(brier_score_loss(y_te, p1))
            met["ece"] = float(expected_calibration_error(y_te, p1))
            if len(np.unique(y_te)) == 2:
                met["auc"] = float(roc_auc_score(y_te, p1))
                met["avg_precision"] = float(average_precision_score(y_te, p1))
            else:
                met["auc"] = float("nan")
                met["avg_precision"] = float("nan")
            ci, cs = calibration_slope_intercept_logistic(y_te, p1)
            met["cal_intercept"] = float(ci)
            met["cal_slope"] = float(cs)

        fold_rows.append(
            {
                "outcome": outcome_name,
                "split": split_name,
                "model": spec.name,
                "fold": f,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "fit_mode": fit_mode,
                "n_fit": int(len(fit_idx)),
                "n_prob_cal": int(len(prob_cal_idx)),
                "n_conf_cal": int(len(conf_cal_idx)),
                "inner_score_neg_logloss": float(inner_score),
                "tuned_params": safe_json_dumps(tuned_params),
                **met,
            }
        )

    oof_df = pd.DataFrame(
        {
            "row_id": row_ids.astype(int),
            "outcome": outcome_name,
            "split": split_name,
            "model": spec.name,
            "fold": oof_fold,
            "y_true": y.astype(int),
        }
    )
    if is_multiclass:
        for k in range(n_classes):
            oof_df[f"p_class_{k}"] = oof_proba[:, k]
        ok = np.isfinite(oof_proba).all(axis=1)
        oof_df["y_pred"] = np.nan
        oof_df.loc[ok, "y_pred"] = np.argmax(oof_proba[ok], axis=1)
    else:
        oof_df["p1"] = oof_proba[:, 1]
        ok = np.isfinite(oof_proba).all(axis=1)
        oof_df["y_pred"] = np.nan
        oof_df.loc[ok, "y_pred"] = (oof_proba[ok, 1] >= 0.5).astype(int)

    fold_df = pd.DataFrame(fold_rows)
    cal_bins_df = pd.concat(cal_bins_rows, ignore_index=True) if cal_bins_rows else pd.DataFrame()
    conf_df = pd.DataFrame(conf_rows) if conf_rows else pd.DataFrame()
    return fold_df, oof_df, cal_bins_df, conf_df


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(FIG_DIR)
    ensure_dir(MAP_DIR)
    ensure_dir(PAPER_DIR)
    ensure_dir(PAPER_TABLES_DIR)

    df = load_data(DATA_PATH).copy().reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=int)

    df_xy = project_lonlat_to_xy_m(df, lon_col=LON_COL, lat_col=LAT_COL)
    utm_epsg = None
    if df_xy is not None:
        df = pd.concat([df, df_xy], axis=1)
        utm_epsg = df_xy.attrs.get("utm_epsg", None)
        for cell in GRID_SIZES_M:
            df[f"_grid_{cell}m"] = make_meter_grid_ids(df, cell_m=cell).astype(str)
        print(f"[INFO] Projected lon/lat to meters (UTM EPSG:{utm_epsg}); created grids: {GRID_SIZES_M}")
    else:
        print("[WARN] Could not project to meters (pyproj missing or no lon/lat). Meter-grid splits unavailable.")

    if POLYGON_PATH:
        poly = try_assign_polygon_groups(df, POLYGON_PATH, LAT_COL, LON_COL, POLYGON_ID_COL)
        if poly is not None:
            df["_poly_id"] = poly.astype(str)
            print(f"[INFO] Assigned polygons from {resolve_path(POLYGON_PATH)} into df['_poly_id']")
        else:
            print("[WARN] Polygon assignment failed; polygon holdouts unavailable.")

    blocks = infer_feature_blocks(df)
    ablations = build_ablation_predictors(blocks)
    specs = make_model_specs(RANDOM_STATE)

    if blocks["issue_cols"]:
        df["_issue_count"] = df[blocks["issue_cols"]].fillna(0).astype(float).sum(axis=1)
    else:
        df["_issue_count"] = 0.0

    manifest = {
        "data_path": resolve_path(DATA_PATH),
        "out_dir": os.path.abspath(OUT_DIR),
        "random_state": RANDOM_STATE,
        "n_outer_splits": N_OUTER_SPLITS,
        "n_inner_splits": N_INNER_SPLITS,
        "n_iter_tune": N_ITER_TUNE,
        "grid_sizes_m": GRID_SIZES_M,
        "utm_epsg": utm_epsg,
        "primary_splits": PRIMARY_SPLITS,
        "secondary_splits": SECONDARY_SPLITS,
        "cal_method": CAL_METHOD,
        "prob_calib_frac": PROB_CALIB_FRAC,
        "conformal_calib_frac": CONFORMAL_CALIB_FRAC,
        "conformal_alpha": CONFORMAL_ALPHA,
        "polygon_path": resolve_path(POLYGON_PATH) if POLYGON_PATH else "",
        "polygon_id_col": POLYGON_ID_COL,
        "decision_unit_col": DECISION_UNIT_COL,
        "ablations": list(ablations.keys()),
        "model_candidates": MODEL_CANDIDATES,
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import sklearn

        manifest["sklearn_version"] = sklearn.__version__
        manifest["numpy_version"] = np.__version__
        manifest["pandas_version"] = pd.__version__
    except Exception:
        pass

    with open(os.path.join(OUT_DIR, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Loaded data: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"[INFO] Ablations: {list(ablations.keys())}")
    print(f"[INFO] Candidate models: {[s.name for s in specs]}")
    print(f"[INFO] Primary splits: {PRIMARY_SPLITS}")

    rng = check_random_state(RANDOM_STATE)

    perf_rows: List[Dict[str, Any]] = []
    fold_metrics_all: List[pd.DataFrame] = []
    oof_all: List[pd.DataFrame] = []
    cal_bins_all: List[pd.DataFrame] = []
    conformal_all: List[pd.DataFrame] = []
    targeting_pooled_all: List[pd.DataFrame] = []
    targeting_by_fold_all: List[pd.DataFrame] = []
    perm_by_fold_all: List[pd.DataFrame] = []
    perm_stability_all: List[pd.DataFrame] = []

    outcomes = [
        (Y_PERCEPTION, True, 2, True),
        (Y_CHOICE, False, 1, False),
    ]

    for y_col, is_multiclass, focus_class, ordinal in outcomes:
        if y_col not in df.columns:
            continue

        d0 = df.dropna(subset=[y_col]).copy().reset_index(drop=True)
        y = d0[y_col].astype(int).to_numpy()
        row_ids = d0["row_id"].to_numpy()

        split_protocols = build_split_protocols(d0, y=y)

        for abl_name, predictors in ablations.items():
            use_cols = [c for c in predictors if c in d0.columns]
            if len(use_cols) < 3:
                continue

            X = d0[use_cols].copy()
            cat_cols = [c for c in use_cols if not pd.api.types.is_numeric_dtype(d0[c])]
            num_cols = [c for c in use_cols if pd.api.types.is_numeric_dtype(d0[c])]
            outcome_name = f"{y_col}__{abl_name}"

            for split_name, spec_split in split_protocols.items():
                if (split_name not in PRIMARY_SPLITS) and (split_name not in SECONDARY_SPLITS):
                    continue

                outer_splits = spec_split["splits"]
                outer_groups = spec_split["groups"]
                inner_group_vec = select_inner_groups_for_protocol(d0, split_name, outer_groups)

                for dummy_strat in ["most_frequent", "stratified"]:
                    dummy_name = f"dummy_{dummy_strat}"
                    n_classes = int(len(np.unique(y))) if is_multiclass else 2
                    fold_vals: List[Dict[str, Any]] = []

                    for f, (tr, te) in enumerate(outer_splits, start=1):
                        dclf = DummyClassifier(strategy=dummy_strat, random_state=RANDOM_STATE)
                        dclf.fit(X.iloc[tr], y[tr])
                        p = dclf.predict_proba(X.iloc[te])
                        cls = estimator_classes(dclf, n_fallback=p.shape[1])
                        p = _align_proba_classes(p, cls, n_classes=n_classes)

                        if is_multiclass:
                            pred = np.argmax(p, axis=1)
                            fold_vals.append(
                                {
                                    "outcome": outcome_name,
                                    "split": split_name,
                                    "model": dummy_name,
                                    "ablation": abl_name,
                                    "fold": f,
                                    "n_test": len(te),
                                    "accuracy": float(accuracy_score(y[te], pred)),
                                    "macro_f1": safe_macro_f1(y[te], pred),
                                    "log_loss": float(log_loss(y[te], p, labels=list(range(n_classes)))),
                                    "kappa_quadratic": float(cohen_kappa_score(y[te], pred, weights="quadratic")),
                                    "mae_ordinal": float(mean_absolute_error(y[te], pred)),
                                }
                            )
                        else:
                            p1 = p[:, 1]
                            pred = (p1 >= 0.5).astype(int)
                            row: Dict[str, Any] = {
                                "outcome": outcome_name,
                                "split": split_name,
                                "model": dummy_name,
                                "ablation": abl_name,
                                "fold": f,
                                "n_test": len(te),
                                "accuracy": float(accuracy_score(y[te], pred)),
                                "macro_f1": safe_macro_f1(y[te], pred),
                                "log_loss": float(log_loss(y[te], p, labels=[0, 1])),
                                "brier": float(brier_score_loss(y[te], p1)),
                            }
                            if len(np.unique(y[te])) == 2:
                                row["auc"] = float(roc_auc_score(y[te], p1))
                                row["avg_precision"] = float(average_precision_score(y[te], p1))
                            fold_vals.append(row)

                    fold_df_dummy = pd.DataFrame(fold_vals)
                    if not fold_df_dummy.empty:
                        fold_metrics_all.append(fold_df_dummy)
                        summary = {
                            "outcome": outcome_name,
                            "split": split_name,
                            "model": dummy_name,
                            "ablation": abl_name,
                            "n_eval": int(len(y)),
                        }
                        met_cols = [c for c in fold_df_dummy.columns if c not in ("outcome", "split", "model", "ablation", "fold", "n_test")]
                        for c in met_cols:
                            summary[c] = float(np.nanmean(pd.to_numeric(fold_df_dummy[c], errors="coerce").to_numpy()))
                        perf_rows.append(summary)

                for model_spec in specs:
                    fold_df, oof_df, cal_bins_df, conf_df = run_nested_outer_cv(
                        X=X,
                        y=y,
                        row_ids=row_ids,
                        outer_splits=outer_splits,
                        outer_groups=outer_groups,
                        inner_groups=inner_group_vec,
                        cat_cols=cat_cols,
                        num_cols=num_cols,
                        spec=model_spec,
                        is_multiclass=is_multiclass,
                        outcome_name=outcome_name,
                        split_name=split_name,
                        focus_class=focus_class,
                        rng=rng,
                        ordinal=ordinal,
                    )

                    if not fold_df.empty:
                        fold_metrics_all.append(fold_df)
                        summary = {
                            "outcome": outcome_name,
                            "split": split_name,
                            "model": model_spec.name,
                            "ablation": abl_name,
                            "n_eval": int(len(y)),
                        }
                        met_cols = [
                            c
                            for c in fold_df.columns
                            if c not in (
                                "outcome", "split", "model", "fold", "n_train", "n_test",
                                "tuned_params", "fit_mode", "n_fit", "n_prob_cal", "n_conf_cal"
                            )
                        ]
                        for c in met_cols:
                            summary[c] = float(np.nanmean(pd.to_numeric(fold_df[c], errors="coerce").to_numpy()))
                        perf_rows.append(summary)

                    oof_all.append(oof_df)
                    if not cal_bins_df.empty:
                        cal_bins_all.append(cal_bins_df)
                    if not conf_df.empty:
                        conformal_all.append(conf_df)

                    if split_name in PRIMARY_SPLITS:
                        try:
                            ok = pd.to_numeric(oof_df["y_pred"], errors="coerce").notna().to_numpy()
                            if ok.sum() > 0:
                                if is_multiclass:
                                    prob_cols = [c for c in oof_df.columns if c.startswith("p_class_")]
                                    proba_ok = oof_df.loc[ok, prob_cols].to_numpy(dtype=float)
                                    y_ok = oof_df.loc[ok, "y_true"].astype(int).to_numpy()
                                    out_png = os.path.join(FIG_DIR, f"reliability_{outcome_name}_{split_name}_{model_spec.name}.png")
                                    plot_reliability_ovr(
                                        y_true=y_ok,
                                        proba=proba_ok,
                                        focus_class=focus_class,
                                        title=f"{outcome_name} | {split_name} | {model_spec.name} (OOF OVR class {focus_class})",
                                        out_png=out_png,
                                    )
                                else:
                                    y_ok = oof_df.loc[ok, "y_true"].astype(int).to_numpy()
                                    p1_ok = pd.to_numeric(oof_df.loc[ok, "p1"], errors="coerce").to_numpy()
                                    out_png = os.path.join(FIG_DIR, f"reliability_{outcome_name}_{split_name}_{model_spec.name}.png")
                                    plot_reliability_binary(
                                        y_true=y_ok,
                                        p_hat=p1_ok,
                                        title=f"{outcome_name} | {split_name} | {model_spec.name} (OOF)",
                                        out_png=out_png,
                                    )
                        except Exception:
                            pass

                    unit_col = None
                    if DECISION_UNIT_COL and DECISION_UNIT_COL in d0.columns:
                        unit_col = DECISION_UNIT_COL
                    elif "_poly_id" in d0.columns and d0["_poly_id"].notna().any():
                        unit_col = "_poly_id"
                    elif "_grid_500m" in d0.columns and d0["_grid_500m"].notna().any():
                        unit_col = "_grid_500m"
                    elif "_grid_1000m" in d0.columns and d0["_grid_1000m"].notna().any():
                        unit_col = "_grid_1000m"

                    if unit_col is not None:
                        if is_multiclass:
                            risk = pd.to_numeric(oof_df.get(f"p_class_{focus_class}"), errors="coerce").to_numpy()
                            y_bin = (oof_df["y_true"].astype(int).to_numpy() == focus_class).astype(int)
                        else:
                            risk = pd.to_numeric(oof_df.get("p1"), errors="coerce").to_numpy()
                            y_bin = oof_df["y_true"].astype(int).to_numpy()

                        issue_base = pd.to_numeric(d0["_issue_count"], errors="coerce").fillna(0.0).to_numpy()
                        point_df = d0[["row_id", unit_col]].copy()
                        point_df["fold"] = pd.to_numeric(oof_df["fold"], errors="coerce").to_numpy()

                        unit_tbl = make_unit_table(
                            point_df[["row_id", unit_col]],
                            unit_col,
                            y_bin=y_bin,
                            risk=risk,
                            baseline_issue_count=issue_base,
                        )
                        unit_tbl["risk_random"] = rng.random(size=len(unit_tbl))

                        for frac in TARGETING_TOP_FRACS:
                            m_model = unit_targeting_metrics(unit_tbl, y_col="y_rate", risk_col="risk_mean", top_frac=frac)
                            m_model.update({
                                "outcome": outcome_name,
                                "split": split_name,
                                "model": model_spec.name,
                                "ablation": abl_name,
                                "unit_col": unit_col,
                                "baseline": "model",
                            })
                            targeting_pooled_all.append(pd.DataFrame([m_model]))

                            if "issue_mean" in unit_tbl.columns:
                                m_issue = unit_targeting_metrics(unit_tbl, y_col="y_rate", risk_col="issue_mean", top_frac=frac)
                                m_issue.update({
                                    "outcome": outcome_name,
                                    "split": split_name,
                                    "model": model_spec.name,
                                    "ablation": abl_name,
                                    "unit_col": unit_col,
                                    "baseline": "issue_count",
                                })
                                targeting_pooled_all.append(pd.DataFrame([m_issue]))

                            m_rand = unit_targeting_metrics(unit_tbl, y_col="y_rate", risk_col="risk_random", top_frac=frac)
                            m_rand.update({
                                "outcome": outcome_name,
                                "split": split_name,
                                "model": model_spec.name,
                                "ablation": abl_name,
                                "unit_col": unit_col,
                                "baseline": "random",
                            })
                            targeting_pooled_all.append(pd.DataFrame([m_rand]))

                        fold_vals = []
                        for fold_id in sorted(point_df["fold"].dropna().unique().tolist()):
                            fold_id = int(fold_id)
                            msk = (point_df["fold"] == fold_id).to_numpy()
                            if msk.sum() < 20:
                                continue
                            unit_tbl_f = make_unit_table(
                                point_df.loc[msk, ["row_id", unit_col]],
                                unit_col,
                                y_bin=y_bin[msk],
                                risk=risk[msk],
                                baseline_issue_count=issue_base[msk],
                            )
                            unit_tbl_f["risk_random"] = rng.random(size=len(unit_tbl_f))
                            for frac in TARGETING_TOP_FRACS:
                                mm = unit_targeting_metrics(unit_tbl_f, y_col="y_rate", risk_col="risk_mean", top_frac=frac)
                                mm.update({
                                    "outcome": outcome_name,
                                    "split": split_name,
                                    "model": model_spec.name,
                                    "ablation": abl_name,
                                    "unit_col": unit_col,
                                    "baseline": "model",
                                    "fold": fold_id,
                                })
                                fold_vals.append(mm)
                                if "issue_mean" in unit_tbl_f.columns:
                                    mi = unit_targeting_metrics(unit_tbl_f, y_col="y_rate", risk_col="issue_mean", top_frac=frac)
                                    mi.update({
                                        "outcome": outcome_name,
                                        "split": split_name,
                                        "model": model_spec.name,
                                        "ablation": abl_name,
                                        "unit_col": unit_col,
                                        "baseline": "issue_count",
                                        "fold": fold_id,
                                    })
                                    fold_vals.append(mi)
                                mr = unit_targeting_metrics(unit_tbl_f, y_col="y_rate", risk_col="risk_random", top_frac=frac)
                                mr.update({
                                    "outcome": outcome_name,
                                    "split": split_name,
                                    "model": model_spec.name,
                                    "ablation": abl_name,
                                    "unit_col": unit_col,
                                    "baseline": "random",
                                    "fold": fold_id,
                                })
                                fold_vals.append(mr)
                        if fold_vals:
                            targeting_by_fold_all.append(pd.DataFrame(fold_vals))

                    if split_name in PRIMARY_SPLITS:
                        try:
                            imp_df = permutation_importance_by_fold(
                                X=X,
                                y=y,
                                outer_splits=outer_splits,
                                cat_cols=cat_cols,
                                num_cols=num_cols,
                                spec=model_spec,
                                scoring="neg_log_loss",
                                outer_groups=outer_groups,
                                inner_groups=inner_group_vec,
                                is_multiclass=is_multiclass,
                                ordinal=ordinal,
                            )
                            if not imp_df.empty:
                                imp_df.insert(0, "outcome", outcome_name)
                                imp_df.insert(1, "split", split_name)
                                imp_df.insert(2, "model", model_spec.name)
                                imp_df.insert(3, "ablation", abl_name)
                                perm_by_fold_all.append(imp_df)

                                stab = summarize_importance_stability(imp_df)
                                stab.insert(0, "outcome", outcome_name)
                                stab.insert(1, "split", split_name)
                                stab.insert(2, "model", model_spec.name)
                                stab.insert(3, "ablation", abl_name)
                                perm_stability_all.append(stab)
                        except Exception:
                            pass

    perf_df = pd.DataFrame(perf_rows)
    if not perf_df.empty:
        perf_df.to_csv(os.path.join(OUT_DIR, "performance_summary.csv"), index=False)

    oracle_df = build_oracle_registry(perf_df)
    if not oracle_df.empty:
        oracle_df.to_csv(ORACLE_REGISTRY_PATH, index=False)
        with open(ORACLE_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(json.loads(oracle_df.to_json(orient="records")), f, indent=2)

    if fold_metrics_all:
        pd.concat(fold_metrics_all, ignore_index=True).to_csv(os.path.join(OUT_DIR, "fold_metrics_all.csv"), index=False)

    if oof_all:
        oof_cat = pd.concat(oof_all, ignore_index=True)
        oof_cat.to_csv(os.path.join(OUT_DIR, "oof_predictions_all.csv"), index=False)

    if cal_bins_all:
        pd.concat(cal_bins_all, ignore_index=True).to_csv(os.path.join(OUT_DIR, "calibration_bins_all.csv"), index=False)

    conformal_df = pd.concat(conformal_all, ignore_index=True) if conformal_all else pd.DataFrame()
    if not conformal_df.empty:
        conformal_df.to_csv(os.path.join(OUT_DIR, "conformal_by_fold.csv"), index=False)

    targeting_pooled_df = pd.concat(targeting_pooled_all, ignore_index=True) if targeting_pooled_all else pd.DataFrame()
    if not targeting_pooled_df.empty:
        targeting_pooled_df.to_csv(os.path.join(OUT_DIR, "targeting_unit_pooled.csv"), index=False)

    targeting_by_fold_df = pd.concat(targeting_by_fold_all, ignore_index=True) if targeting_by_fold_all else pd.DataFrame()
    if not targeting_by_fold_df.empty:
        targeting_by_fold_df.to_csv(os.path.join(OUT_DIR, "targeting_unit_by_fold.csv"), index=False)

    if perm_by_fold_all:
        pd.concat(perm_by_fold_all, ignore_index=True).to_csv(os.path.join(OUT_DIR, "perm_importance_by_fold.csv"), index=False)

    if perm_stability_all:
        pd.concat(perm_stability_all, ignore_index=True).to_csv(os.path.join(OUT_DIR, "perm_importance_stability.csv"), index=False)

    if LAT_COL in df.columns and LON_COL in df.columns and oof_all:
        oof_all_df = pd.concat(oof_all, ignore_index=True)

        def _entropy(P: np.ndarray) -> np.ndarray:
            eps = 1e-12
            P = np.clip(P, eps, 1.0)
            return -np.sum(P * np.log(P), axis=1)

        oracle_map = oracle_df.copy() if not oracle_df.empty else pd.DataFrame()

        for y_col, is_multiclass, focus_class, ordinal in outcomes:
            for ab in ["full", "no_location"]:
                outcome_prefix = f"{y_col}__{ab}"
                cand = oof_all_df[oof_all_df["outcome"].astype(str) == outcome_prefix]
                cand = cand[cand["split"].isin(PRIMARY_SPLITS)] if not cand.empty else cand
                if cand.empty:
                    continue

                model_pick = None
                split_pick = None
                if not oracle_map.empty:
                    d_or = oracle_map[
                        (oracle_map["outcome"].astype(str) == outcome_prefix) &
                        (oracle_map["ablation"].astype(str) == ab) &
                        (oracle_map["split"].astype(str).isin(PRIMARY_SPLITS))
                    ]
                    if not d_or.empty:
                        split_pick = str(d_or.iloc[0]["split"])
                        model_pick = str(d_or.iloc[0]["oracle_model"])

                if model_pick is None:
                    if (cand["model"] == "hgb").any():
                        cand = cand[cand["model"] == "hgb"]
                    split_pick = str(cand["split"].iloc[0])
                    model_pick = str(cand["model"].iloc[0])

                df_join = df.merge(
                    cand[(cand["split"] == split_pick) & (cand["model"] == model_pick)][
                        ["row_id"] + [c for c in cand.columns if c.startswith("p_class_") or c == "p1"]
                    ],
                    on="row_id",
                    how="inner",
                )
                if df_join.empty:
                    continue

                if is_multiclass:
                    prob_cols = [c for c in df_join.columns if c.startswith("p_class_")]
                    P = df_join[prob_cols].to_numpy(dtype=float)
                    df_join["entropy"] = _entropy(P)
                    risk_col = f"p_class_{focus_class}"
                    out_html = os.path.join(MAP_DIR, f"oof_map_{y_col}_{ab}_{split_pick}_{model_pick}.html")
                    export_risk_map_folium(
                        df_join,
                        LAT_COL,
                        LON_COL,
                        risk_col,
                        "entropy",
                        out_html,
                        title=f"OOF mosaic: {y_col} | {ab} | Pr(class={focus_class})",
                        max_points=MAX_POINTS_MAP,
                        banner="SCIENTIFIC MAP: out-of-fold mosaic predictions on held-out folds (valid for spatial generalization visualization).",
                    )
                else:
                    p = pd.to_numeric(df_join["p1"], errors="coerce").to_numpy()
                    eps = 1e-12
                    ent = -(np.clip(p, eps, 1.0) * np.log(np.clip(p, eps, 1.0)) + np.clip(1 - p, eps, 1.0) * np.log(np.clip(1 - p, eps, 1.0)))
                    df_join["entropy"] = ent
                    out_html = os.path.join(MAP_DIR, f"oof_map_{y_col}_{ab}_{split_pick}_{model_pick}.html")
                    export_risk_map_folium(
                        df_join,
                        LAT_COL,
                        LON_COL,
                        "p1",
                        "entropy",
                        out_html,
                        title=f"OOF mosaic: {y_col} | {ab} | Pr(1)",
                        max_points=MAX_POINTS_MAP,
                        banner="SCIENTIFIC MAP: out-of-fold mosaic predictions on held-out folds (valid for spatial generalization visualization).",
                    )

        try:
            for y_col, is_multiclass, focus_class, ordinal in outcomes:
                if y_col not in df.columns:
                    continue
                dfit = df.dropna(subset=[y_col]).copy()
                blocks2 = infer_feature_blocks(dfit)
                preds = [c for c in blocks2["full_predictors"] if c in dfit.columns]
                if len(preds) < 3:
                    continue
                Xfit = dfit[preds].copy()
                yfit = dfit[y_col].astype(int).to_numpy()
                cat_cols = [c for c in preds if not pd.api.types.is_numeric_dtype(dfit[c])]
                num_cols = [c for c in preds if pd.api.types.is_numeric_dtype(dfit[c])]

                best_model_name = "hgb"
                if not oracle_df.empty:
                    d_or = oracle_df[
                        (oracle_df["outcome"].astype(str) == f"{y_col}__full") &
                        (oracle_df["ablation"].astype(str) == "full")
                    ]
                    if not d_or.empty:
                        best_model_name = str(d_or.iloc[0]["oracle_model"])

                spec_lookup = {s.name: s for s in specs}
                chosen_spec = spec_lookup.get(best_model_name, spec_lookup.get("hgb", specs[0]))

                pipe = make_pipeline(cat_cols, num_cols, chosen_spec, is_multiclass=is_multiclass, ordinal=ordinal)
                pipe, _ = fit_pipeline_safe(pipe, Xfit, yfit)
                proba = pipe.predict_proba(Xfit)
                K = int(len(np.unique(yfit))) if is_multiclass else 2
                cls = estimator_classes(pipe, n_fallback=proba.shape[1])
                proba = _align_proba_classes(proba, cls, n_classes=K)

                if is_multiclass:
                    for k in range(K):
                        dfit[f"p_class_{k}"] = proba[:, k]
                    dfit["entropy"] = _entropy(proba)
                    out_html = os.path.join(MAP_DIR, f"fullfit_map_DEPLOYMENT_ONLY_{y_col}_{best_model_name}.html")
                    export_risk_map_folium(
                        dfit,
                        LAT_COL,
                        LON_COL,
                        f"p_class_{focus_class}",
                        "entropy",
                        out_html,
                        title=f"FULL-FIT (deployment only): {y_col} Pr(class={focus_class})",
                        max_points=MAX_POINTS_MAP,
                        banner="DEPLOYMENT-ONLY: trained on full dataset and predicted on the same points. NOT valid for generalization evidence.",
                    )
                else:
                    dfit["p1"] = proba[:, 1]
                    p = np.clip(dfit["p1"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
                    dfit["entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
                    out_html = os.path.join(MAP_DIR, f"fullfit_map_DEPLOYMENT_ONLY_{y_col}_{best_model_name}.html")
                    export_risk_map_folium(
                        dfit,
                        LAT_COL,
                        LON_COL,
                        "p1",
                        "entropy",
                        out_html,
                        title=f"FULL-FIT (deployment only): {y_col} Pr(1)",
                        max_points=MAX_POINTS_MAP,
                        banner="DEPLOYMENT-ONLY: trained on full dataset and predicted on the same points. NOT valid for generalization evidence.",
                    )
        except Exception:
            pass

    try:
        generate_paper_report(
            perf_df=perf_df if not perf_df.empty else pd.DataFrame(),
            fold_df=pd.read_csv(os.path.join(OUT_DIR, "fold_metrics_all.csv")) if os.path.exists(os.path.join(OUT_DIR, "fold_metrics_all.csv")) else pd.DataFrame(),
            targeting_pooled_df=targeting_pooled_df,
            targeting_by_fold_df=targeting_by_fold_df,
            conformal_df=conformal_df,
            out_dir=OUT_DIR,
            fig_dir=FIG_DIR,
        )
    except Exception as e:
        print(f"[WARN] Paper report generation failed: {e}")

    print("[DONE] Spatial predictive pipeline complete.")
    print(f"  Outputs:  {os.path.abspath(OUT_DIR)}")
    print(f"  Figures:  {os.path.abspath(FIG_DIR)}")
    print(f"  Maps:     {os.path.abspath(MAP_DIR)}")
    print(f"  Paper:    {os.path.abspath(PAPER_DIR)}")
    if os.path.exists(ORACLE_REGISTRY_PATH):
        print(f"  Oracle:   {os.path.abspath(ORACLE_REGISTRY_PATH)}")


if __name__ == "__main__":
    main()

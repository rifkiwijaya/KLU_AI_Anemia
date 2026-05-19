"""Feature engineering and preprocessing utilities for anemia analytics."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

AGE_BINS: Sequence[float] = (-np.inf, 19, 25, 30, 35, np.inf)
AGE_LABELS: Sequence[str] = ("<20", "20-25", "26-30", "31-35", ">35")

HB_THRESHOLD: float = 11.0
LILA_THRESHOLD: float = 23.5

NUMERIC_FEATURES: Sequence[str] = (
    "Usia",
    "Pendapatan",
    "Jarak_Faskes",
    "Hb",
    "LILA",
    "Frekuensi_Protein_Hewani",
)

CATEGORICAL_FEATURES: Sequence[str] = (
    "Pendidikan",
    "Trimester",
    "Gravida",
    "Terima_TTD",
    "Kepatuhan_TTD",
    "Kendala_TTD",
    "Konsumsi_Inhibitor",
    "Akses_Air",
    "Cuci_Tangan",
    "Paparan_Asap_Rokok",
    "Desa",
    "Kecamatan",
)


@dataclass(slots=True)
class FeatureMatrix:
    """Container for processed feature matrix and encoders."""

    features: pd.DataFrame
    encoder: Optional[OneHotEncoder]
    scaler: Optional[StandardScaler | MinMaxScaler]


def assign_age_group(age_series: pd.Series) -> pd.Series:
    """Compute categorical age group bands.

    Args:
        age_series: Series containing age values in years.

    Returns:
        Series of age group labels aligned with the input index.
    """
    return pd.cut(
        age_series,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
        right=True,
    )


def categorize_hb_status(hb_series: pd.Series, threshold: float = HB_THRESHOLD) -> pd.Series:
    """Categorize hemoglobin measurements.

    Args:
        hb_series: Series containing Hb values in g/dL.
        threshold: Cut-off for anemia classification.

    Returns:
        Series of categorical anemia labels.
    """
    categories = np.where(hb_series < threshold, "Anemia", "Non-Anemia")
    return pd.Series(categories, index=hb_series.index, dtype="category")


def derive_kek_status(lila_series: pd.Series, threshold: float = LILA_THRESHOLD) -> pd.Series:
    """Infer KEK status based on arm circumference measurements.

    Args:
        lila_series: Series with LILA values in centimeters.
        threshold: Cut-off point for chronic energy deficiency risk.

    Returns:
        Series of KEK risk labels.
    """
    status = np.where(lila_series < threshold, "Berisiko KEK", "Tidak Berisiko")
    return pd.Series(status, index=lila_series.index, dtype="category")


def compute_risk_score(
    dataframe: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Generate composite risk scores using weighted feature contributions.

    Args:
        dataframe: Input dataframe containing relevant features.
        weights: Custom feature weights; defaults calibrated on domain heuristics.

    Returns:
        Normalized risk score between 0 and 100.
    """
    default_weights: Dict[str, float] = {
        "Hb": 0.35,
        "LILA": 0.25,
        "Kepatuhan_TTD": 0.15,
        "Usia": 0.1,
        "Status_KEK": 0.1,
        "Frekuensi_Protein_Hewani": 0.05,
    }
    if weights:
        default_weights.update(weights)

    df = dataframe.copy()
    df["Kepatuhan_TTD"] = pd.to_numeric(df.get("Kepatuhan_TTD", 0), errors="coerce").fillna(0)

    status_kek_source = (
        df["Status_KEK"]
        if "Status_KEK" in df
        else pd.Series(["Tidak Berisiko"] * len(df), index=df.index)
    )
    status_anemia_source = (
        df["Status_Anemia"]
        if "Status_Anemia" in df
        else pd.Series(["Non-Anemia"] * len(df), index=df.index)
    )

    # Convert to binary flags: 1 for risk condition, 0 otherwise
    df["Status_KEK"] = status_kek_source.astype(str).str.strip().str.lower().str.startswith("berisiko").astype(float)
    df["Status_Anemia"] = status_anemia_source.astype(str).str.strip().str.lower().eq("anemia").astype(float)

    score = np.zeros(len(df), dtype=float)

    if "Hb" in df:
        hb_component = np.clip((HB_THRESHOLD - df["Hb"]) / HB_THRESHOLD, 0, 1)
        score += default_weights["Hb"] * hb_component

    if "LILA" in df:
        lila_component = np.clip((LILA_THRESHOLD - df["LILA"]) / LILA_THRESHOLD, 0, 1)
        score += default_weights["LILA"] * lila_component

    score += default_weights.get("Kepatuhan_TTD", 0) * (1 - (df["Kepatuhan_TTD"] / 100.0).clip(0, 1))
    score += default_weights.get("Usia", 0) * np.clip(df.get("Usia", 0) / 45.0, 0, 1)
    score += default_weights.get("Status_KEK", 0) * df["Status_KEK"]

    if "Frekuensi_Protein_Hewani" in df:
        protein_component = np.clip(1 - (df["Frekuensi_Protein_Hewani"] / 7.0), 0, 1)
        score += default_weights.get("Frekuensi_Protein_Hewani", 0) * protein_component

    score += 0.1 * df["Status_Anemia"]

    normalized = np.clip(score / score.max() if score.max() else score, 0, 1)
    return pd.Series((normalized * 100).round(2), index=df.index, name="Risk_Score")


def handle_outliers_iqr(
    dataframe: pd.DataFrame,
    numeric_columns: Iterable[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Cap outliers using the IQR rule.

    Args:
        dataframe: Input dataframe.
        numeric_columns: Columns subjected to outlier capping.
        multiplier: IQR multiplier defining the bounds.

    Returns:
        Dataframe with outliers clipped to whisker boundaries.
    """
    capped = dataframe.copy()
    for column in numeric_columns:
        if column not in capped.columns:
            continue
        series = capped[column].dropna()
        if series.empty:
            continue
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        capped[column] = capped[column].clip(lower=lower_bound, upper=upper_bound)
    return capped


def scale_features(
    dataframe: pd.DataFrame,
    numeric_columns: Iterable[str],
    method: str = "standard",
) -> Tuple[pd.DataFrame, StandardScaler | MinMaxScaler | None]:
    """Scale numeric features using the specified method.

    Args:
        dataframe: Dataframe containing the numeric features.
        numeric_columns: Columns to scale.
        method: Scaling method - "standard" or "minmax".

    Returns:
        Tuple of transformed dataframe and fitted scaler instance.
    """
    valid_columns = [column for column in numeric_columns if column in dataframe.columns]
    if not valid_columns:
        logger.warning("No numeric columns provided for scaling")
        return dataframe.copy(), None

    scaled_df = dataframe.copy()
    if method == "standard":
        scaler: StandardScaler | MinMaxScaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaling method: %s" % method)

    scaled_values = scaler.fit_transform(scaled_df[valid_columns])
    scaled_df[valid_columns] = scaled_values
    return scaled_df, scaler


def encode_categorical(
    dataframe: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> Tuple[pd.DataFrame, Optional[OneHotEncoder]]:
    """Apply one-hot encoding to categorical columns.

    Args:
        dataframe: Input dataframe.
        categorical_columns: Categorical columns to encode.

    Returns:
        Tuple containing transformed dataframe and fitted encoder.
    """
    valid_columns = [column for column in categorical_columns if column in dataframe.columns]
    if not valid_columns:
        logger.info("No categorical columns supplied for encoding")
        return dataframe.copy(), None

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        dtype=float,
    )
    encoded = encoder.fit_transform(dataframe[valid_columns])
    encoded_columns = encoder.get_feature_names_out(valid_columns)
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=dataframe.index)

    remainder = dataframe.drop(columns=valid_columns)
    transformed = pd.concat([remainder, encoded_df], axis=1)
    return transformed, encoder


def create_interaction_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Generate derived interaction features to enhance model performance.

    Args:
        dataframe: Base dataframe.

    Returns:
        Dataframe augmented with additional interaction terms.
    """
    df = dataframe.copy()
    if {"Hb", "Kepatuhan_TTD"}.issubset(df.columns):
        df["Hb_x_Kepatuhan"] = (
            pd.to_numeric(df["Hb"], errors="coerce").fillna(0)
            * pd.to_numeric(df["Kepatuhan_TTD"], errors="coerce").fillna(0)
        )
    if {"LILA", "Frekuensi_Protein_Hewani"}.issubset(df.columns):
        df["LILA_to_Protein"] = (
            pd.to_numeric(df["LILA"], errors="coerce").replace(0, np.nan)
            / pd.to_numeric(df["Frekuensi_Protein_Hewani"], errors="coerce").replace(0, np.nan)
        )
        df["LILA_to_Protein"] = df["LILA_to_Protein"].fillna(df["LILA_to_Protein"].median())
    return df


def build_feature_matrix(
    dataframe: pd.DataFrame,
    numeric_columns: Optional[Sequence[str]] = None,
    categorical_columns: Optional[Sequence[str]] = None,
    scaling: str = "standard",
) -> FeatureMatrix:
    """Produce a machine-learning-ready feature matrix from the raw dataframe.

    Args:
        dataframe: Cleaned input dataframe.
        numeric_columns: Numeric columns to consider for scaling.
        categorical_columns: Categorical columns to encode.
        scaling: Scaling method for numeric features.

    Returns:
        FeatureMatrix object containing transformed features and fitted artifacts.
    """
    numeric_columns = numeric_columns or NUMERIC_FEATURES
    categorical_columns = categorical_columns or CATEGORICAL_FEATURES

    df = dataframe.copy()
    df["Kelompok_Usia"] = assign_age_group(df["Usia"])
    df["Kategori_Hb"] = categorize_hb_status(df["Hb"])

    if "Status_KEK" not in df:
        df["Status_KEK"] = derive_kek_status(df["LILA"])

    df["Risk_Score"] = compute_risk_score(df)
    df = create_interaction_features(df)
    df = handle_outliers_iqr(df, numeric_columns)

    scaled_df, scaler = scale_features(df, numeric_columns, method=scaling)
    features, encoder = encode_categorical(scaled_df, categorical_columns)

    logger.info(
        "Feature matrix created with %d rows and %d columns", features.shape[0], features.shape[1]
    )
    return FeatureMatrix(features=features, encoder=encoder, scaler=scaler)

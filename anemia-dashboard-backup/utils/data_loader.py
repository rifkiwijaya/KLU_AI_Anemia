"""Utilities for loading and validating anemia survey datasets."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import streamlit as st

REQUIRED_COLUMNS: Sequence[str] = (
    "Timestamp",
    "Nama",
    "Usia",
    "Pendidikan",
    "Pendapatan",
    "Jarak_Faskes",
    "Hb",
    "LILA",
    "Status_KEK",
    "Status_Anemia",
    "Trimester",
    "Gravida",
    "Terima_TTD",
    "Kepatuhan_TTD",
    "Kendala_TTD",
    "Frekuensi_Protein_Hewani",
    "Konsumsi_Inhibitor",
    "Akses_Air",
    "Cuci_Tangan",
    "Paparan_Asap_Rokok",
    "Desa",
    "Kecamatan",
)

NUMERIC_COLUMNS: Sequence[str] = (
    "Usia",
    "Pendapatan",
    "Jarak_Faskes",
    "Hb",
    "LILA",
    "Frekuensi_Protein_Hewani",
    "Kepatuhan_TTD",
)

CATEGORICAL_COLUMNS: Sequence[str] = tuple(
    column for column in REQUIRED_COLUMNS if column not in NUMERIC_COLUMNS
)

FREQUENCY_MAP = {
    "tidak pernah": 0.0,
    "1x/bulan": 0.25,
    "1-3x/bulan": 0.5,
    "1x/minggu": 1.0,
    "2-3x/minggu": 2.5,
    "3-4x/minggu": 3.5,
    "4-6x/minggu": 5.0,
    "1x/hari": 7.0,
    "≥1x/hari": 7.0,
    ">=1x/hari": 7.0,
    "setiap hari": 7.0,
}

INCOME_MAP = {
    "<1 juta": 0.75,
    "1-2 juta": 1.5,
    "2-3 juta": 2.5,
    ">3 juta": 3.5,
}

DISTANCE_MAP = {
    "<1 km": 0.5,
    "1-3 km": 2.0,
    ">3 km": 4.0,
}

TTD_COMPLIANCE_MAP = {
    "habis semua": 100.0,
    "masih tersisa beberapa": 70.0,
    "banyak yang belum diminum": 30.0,
    "tidak diminum sama sekali": 0.0,
}


def _validate_columns(dataframe: pd.DataFrame, required: Iterable[str]) -> None:
    """Validate presence of required columns in the loaded dataframe."""

    missing = set(required) - set(dataframe.columns)
    if missing:
        raise ValueError(
            "Dataset missing required columns: %s" % ", ".join(sorted(missing))
        )


def _coerce_numeric_columns(dataframe: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected columns to numeric dtype with coercion."""

    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def _fill_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using simple, column-aware strategies."""

    existing_numeric = [column for column in NUMERIC_COLUMNS if column in dataframe.columns]
    if existing_numeric:
        numeric_df = dataframe[existing_numeric]
        dataframe[numeric_df.columns] = numeric_df.fillna(numeric_df.median(numeric_only=True))

    categorical_defaults = {
        column: "Tidak Diketahui" for column in CATEGORICAL_COLUMNS if column != "Kendala_TTD"
    }

    if "Kendala_TTD" in dataframe:
        dataframe["Kendala_TTD"] = dataframe["Kendala_TTD"].fillna("Lainnya")

    for column, default_value in categorical_defaults.items():
        if column in dataframe:
            dataframe[column] = dataframe[column].fillna(default_value)

    return dataframe


def _pick_first_available(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def _extract_series(dataframe: pd.DataFrame, candidates: Sequence[str], default: object = np.nan) -> pd.Series:
    column = _pick_first_available(dataframe, candidates)
    if column:
        return dataframe[column]
    return pd.Series([default] * len(dataframe), index=dataframe.index)


def _parse_income(value: object) -> float:
    if not isinstance(value, str):
        return np.nan
    normalized = value.strip().lower()
    for pattern, midpoint in INCOME_MAP.items():
        if pattern in normalized:
            return midpoint * 1_000_000
    numeric_tokens = re.findall(r"\d[\d\.,]*", normalized)
    if numeric_tokens:
        parsed_values = []
        for token in numeric_tokens:
            cleaned = token.replace(".", "").replace(",", ".")
            try:
                parsed_values.append(float(cleaned))
            except ValueError:
                continue
        if parsed_values:
            average = sum(parsed_values) / len(parsed_values)
            if "juta" in normalized and average <= 10:
                return average * 1_000_000
            return average
    return np.nan


def _parse_distance(value: object) -> float:
    if not isinstance(value, str):
        return np.nan
    normalized = value.strip().lower()
    normalized = normalized.replace(" km", "km")
    for pattern, numeric in DISTANCE_MAP.items():
        if pattern in normalized:
            return numeric
    digits = re.findall(r"\d+(?:\.\d+)?", normalized)
    if digits:
        return float(digits[0])
    return np.nan


def _parse_trimester(value: object) -> str | np.nan:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    digits = re.findall(r"\d+(?:\.\d+)?", text)
    if not digits:
        return np.nan
    number = float(digits[0])
    if "bulan" in text:
        weeks = number * 4
    else:
        weeks = number
    if weeks <= 0:
        return np.nan
    trimester = int(min(3, max(1, ((weeks - 1) // 13) + 1)))
    return str(trimester)


def _parse_gravida(value: object) -> str | np.nan:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value)
    match = re.search(r"(\d+)", text)
    if match:
        return match.group(1)
    return np.nan


def _map_frequency(value: object) -> float:
    if not isinstance(value, str):
        return np.nan
    normalized = value.strip().lower().replace(">=", "≥")
    return FREQUENCY_MAP.get(normalized, np.nan)


def _compute_frequency_average(dataframe: pd.DataFrame, keyword: str) -> pd.Series:
    columns = [
        column
        for column in dataframe.columns
        if keyword in column.lower() and "sebutkan" not in column.lower()
    ]
    if not columns:
        return pd.Series(np.nan, index=dataframe.index)
    mapped = pd.DataFrame({column: dataframe[column].map(_map_frequency) for column in columns})
    return mapped.mean(axis=1, skipna=True)


def _compute_protein_frequency(dataframe: pd.DataFrame) -> pd.Series:
    return _compute_frequency_average(dataframe, "makanan sumber zat besi hewani")


def _compute_inhibitor_category(dataframe: pd.DataFrame) -> pd.Series:
    avg = _compute_frequency_average(dataframe, "makanan sumber penghambat zat besi")

    def categorize(value: float) -> str:
        if np.isnan(value):
            return "Tidak Diketahui"
        if value >= 4:
            return "Tinggi"
        if value >= 1:
            return "Sedang"
        return "Rendah"

    return avg.apply(categorize)


def _calculate_ttd_compliance(value: object) -> float:
    if isinstance(value, (int, float)) and not np.isnan(value):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        percentage_match = re.search(r"(\d{1,3})%", normalized)
        if percentage_match:
            return float(percentage_match.group(1))
        numeric_match = re.search(r"(\d{1,3})", normalized)
        if numeric_match and "tablet" in normalized:
            return float(numeric_match.group(1))
        if normalized in TTD_COMPLIANCE_MAP:
            return TTD_COMPLIANCE_MAP[normalized]
    return np.nan


def _normalize_ttd_received(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Tidak Diketahui"
    normalized = value.strip().lower()
    return "Tidak" if "tidak" in normalized else "Ya"


def _derive_desa(dataframe: pd.DataFrame) -> pd.Series:
    desa_columns = [column for column in dataframe.columns if column.lower().startswith("pilih desa")]
    if not desa_columns:
        return pd.Series([np.nan] * len(dataframe), index=dataframe.index)
    desa_df = dataframe[desa_columns]
    return desa_df.apply(
        lambda row: next((str(value).strip() for value in row if isinstance(value, str) and value.strip()), np.nan),
        axis=1,
    )


def transform_raw_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform Google Form export into standardized schema."""

    result = pd.DataFrame(index=dataframe.index)

    result["Timestamp"] = dataframe.get("Timestamp")
    result["Nama"] = _extract_series(dataframe, ["Nama Lengkap", "Nama"], default="Tidak Diketahui")
    result["Usia"] = pd.to_numeric(
        _extract_series(dataframe, ["Usia", "Usia ", "Usia (tahun)"]), errors="coerce"
    )
    result["Pendidikan"] = _extract_series(
        dataframe,
        ["Pendidikan terakhir ibu hamil", "Pendidikan terakhir ibu hamil "],
    )

    pendapatan_series = _extract_series(
        dataframe,
        ["Kisaran pendapatan bulanan keluarga", "Kisaran pendapatan bulanan keluarga "],
    )
    result["Pendapatan"] = pendapatan_series.apply(_parse_income)

    jarak_series = _extract_series(
        dataframe,
        [
            "Jarak dari rumah ke tempat layanan kesehatan",
            "Jarak dari rumah ke tempat layanan kesehatan terdekat",
        ],
    )
    result["Jarak_Faskes"] = jarak_series.apply(_parse_distance)

    result["Hb"] = pd.to_numeric(
        _extract_series(dataframe, ["Hasil pemeriksaan Hb (g/dL)", "Hasil pemeriksaan Hb (g/dL) "]),
        errors="coerce",
    )
    result["LILA"] = pd.to_numeric(
        _extract_series(dataframe, ["LILA (Lingkar Lengan Atas)", "LILA (Lingkar Lengan Atas) "]),
        errors="coerce",
    )

    lila_values = result["LILA"]
    kek_status = pd.Series(np.nan, index=result.index, dtype="object")
    mask_lila = lila_values.notna()
    kek_status.loc[mask_lila] = np.where(
        lila_values.loc[mask_lila] < 23.5,
        "Berisiko KEK",
        "Tidak Berisiko",
    )
    result["Status_KEK"] = kek_status

    hb_values = result["Hb"]
    anemia_status = pd.Series(np.nan, index=result.index, dtype="object")
    mask_hb = hb_values.notna()
    anemia_status.loc[mask_hb] = np.where(
        hb_values.loc[mask_hb] < 11,
        "Anemia",
        "Non-Anemia",
    )
    result["Status_Anemia"] = anemia_status

    usia_kehamilan = _extract_series(dataframe, ["Usia kehamilan saat ini"])
    result["Trimester"] = usia_kehamilan.apply(_parse_trimester)

    paritas = _extract_series(
        dataframe,
        ["Paritas (G_P_A_)", "Paritas (G P A)", "Paritas"],
    )
    result["Gravida"] = paritas.apply(_parse_gravida)

    ttd_source = _extract_series(
        dataframe,
        [
            "Apakah ibu menerima Tablet Tambah Darah (TTD) pada kehamilan ini?)",
            "Apakah ibu menerima Tablet Tambah Darah (TTD) pada kehamilan ini?",
            "Apakah ibu menerima Tablet Tambah Darah (TTD) pada kehamilan ini",
        ],
    )
    result["Terima_TTD"] = ttd_source.apply(_normalize_ttd_received)

    ttd_compliance = _extract_series(
        dataframe,
        [
            "Dari tablet yang ibu terima, kira-kira berapa yang sudah diminum? ",
            "Dari tablet yang ibu terima, kira-kira berapa yang sudah diminum?",
        ],
    )
    result["Kepatuhan_TTD"] = ttd_compliance.apply(_calculate_ttd_compliance)

    result["Kendala_TTD"] = _extract_series(
        dataframe,
        [
            "Apa alasan ibu tidak minum / tidak rutin minum TTD? ",
            "Apa alasan ibu tidak minum / tidak rutin minum TTD?",
        ],
        default="",
    )

    result["Frekuensi_Protein_Hewani"] = _compute_protein_frequency(dataframe)
    result["Konsumsi_Inhibitor"] = _compute_inhibitor_category(dataframe)

    result["Akses_Air"] = _extract_series(
        dataframe,
        [
            "Dari manakah sumber air minum yang biasanya ibu konsumsi sehari-hari? ",
            "Dari manakah sumber air minum yang biasanya ibu konsumsi sehari-hari?",
        ],
    )
    result["Cuci_Tangan"] = _extract_series(
        dataframe,
        [
            "Apakah ibu mencuci tangan dengan sabun sebelum makan, menyiapkan makanan, setelah dari toilet, setelah memegang hewan/sampah? ",
            "Apakah ibu mencuci tangan dengan sabun sebelum makan, menyiapkan makanan, setelah dari toilet, setelah memegang hewan/sampah?",
        ],
    )
    result["Paparan_Asap_Rokok"] = _extract_series(
        dataframe,
        [
            "Apakah ada anggota keluarga merokok di dalam rumah? ",
            "Apakah ada anggota keluarga merokok di dalam rumah?",
        ],
    )

    result["Desa"] = _derive_desa(dataframe)
    kecamatan_raw = _extract_series(
        dataframe,
        ["Kecamatan ", "Kecamatan"],
    )
    result["Kecamatan"] = kecamatan_raw.apply(
        lambda value: value.strip() if isinstance(value, str) and value.strip() else np.nan
    )

    reordered = result.reindex(columns=REQUIRED_COLUMNS)
    return reordered


@st.cache_data(show_spinner=False)
def load_excel_dataset(file_path: Path) -> pd.DataFrame:
    """Load and sanitize the anemia survey Excel dataset."""

    logger = logging.getLogger(__name__)
    if not file_path.exists():
        logger.error("Excel file not found at %s", file_path)
        raise FileNotFoundError(f"Excel file not found at {file_path}")

    try:
        raw_dataframe = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")
    except ValueError as exc:
        logger.exception("Invalid Excel file format: %s", exc)
        raise ValueError("Invalid Excel file format or unreadable content") from exc
    except Exception as exc:  # pragma: no cover - safeguard for unexpected issues
        logger.exception("Unexpected error when reading Excel: %s", exc)
        raise

    if raw_dataframe.empty:
        logger.error("Loaded Excel file is empty")
        raise ValueError("Loaded Excel file is empty")

    dataframe = transform_raw_dataset(raw_dataframe)
    _validate_columns(dataframe, REQUIRED_COLUMNS)

    dataframe = _coerce_numeric_columns(dataframe, NUMERIC_COLUMNS)
    dataframe = _fill_missing_values(dataframe)

    for column in CATEGORICAL_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("category")

    dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"], errors="coerce")
    if dataframe["Timestamp"].isna().any():
        logger.warning("Timestamp column contains invalid dates; replacing with earliest valid")
        earliest = dataframe["Timestamp"].dropna().min()
        dataframe["Timestamp"] = dataframe["Timestamp"].fillna(earliest)

    logger.info("Loaded dataset with %d rows and %d columns", dataframe.shape[0], dataframe.shape[1])
    return dataframe.copy()

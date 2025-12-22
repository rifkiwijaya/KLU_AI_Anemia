"""Application configuration for the anemia dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


BASE_DIR: Path = Path(__file__).resolve().parent

# Application metadata
APP_TITLE: str = "Dashboard Pemetaan Risiko Anemia Ibu Hamil"
APP_ICON: str = "🏥"

# Paths
DATA_PATH: Path = BASE_DIR / "data" / "raw"
MODEL_PATH: Path = BASE_DIR / "models"
EXCEL_FILENAME: str = "Kuesioner Ibu Hamil Anemia (Responses) sheet1.xlsx"
DATA_FILE: Path = DATA_PATH / EXCEL_FILENAME

# Color palette
COLOR_SCHEME: Dict[str, str] = {
    "primary": "#1f77b4",
    "secondary": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff7f0e",
    "background": "#f8f9fa",
    "neutral_dark": "#2b2d42",
    "neutral_light": "#f1f3f5",
}

# Map configuration (Kabupaten Lombok Utara)
MAP_CENTER: Tuple[float, float] = (-8.354, 116.277)
MAP_ZOOM: int = 10

# Risk thresholds (percentage)
RISK_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "High": (30.0, float("inf")),
    "Medium": (15.0, 30.0),
    "Low": (0.0, 15.0),
}

# Risk scoring parameters
RISK_SCORING_WEIGHTS: Dict[str, float] = {
    "Hb": 0.35,
    "LILA": 0.25,
    "Kepatuhan_TTD": 0.15,
    "Usia": 0.10,
    "Status_KEK": 0.10,
    "Frekuensi_Protein_Hewani": 0.05,
}

# Machine learning model settings
MODEL_PARAMS: Dict[str, object] = {
    "algorithm": "RandomForest",
    "n_estimators": 250,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
}

CROSS_VALIDATION: Dict[str, int] = {
    "splits": 5,
    "random_state": 42,
}

# Text mining keyword dictionaries
TEXT_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "Lupa": (
        "lupa",
        "sibuk",
        "terlupa",
        "kesibukan",
        "tidak sempat",
        "terburu",
        "capek",
        "lelah",
        "banyak pekerjaan",
    ),
    "Efek Samping": (
        "mual",
        "muntah",
        "pusing",
        "sakit perut",
        "konstipasi",
        "sembelit",
        "tinja hitam",
        "bau",
        "mules",
        "eneg",
    ),
    "Stok Habis": (
        "habis",
        "tidak tersedia",
        "kosong",
        "tidak ada",
        "kehabisan",
        "stok",
        "puskesmas",
        "pustu",
        "gudang",
    ),
    "Lainnya": (),
}

SENTIMENT_LEXICON: Dict[str, int] = {
    "baik": 1,
    "bagus": 1,
    "nyaman": 1,
    "mantap": 1,
    "buruk": -1,
    "mual": -1,
    "muntah": -1,
    "sakit": -1,
    "pusing": -1,
    "lega": 1,
    "sehat": 1,
    "khawatir": -1,
    "cemas": -1,
    "takut": -1,
    "senang": 1,
}

# Export file names
EXPORT_CLUSTER_RESULTS: str = "hasil_cluster_anemia.xlsx"
EXPORT_TTD_ANALYSIS: str = "analisis_kendala_ttd.xlsx"
EXPORT_RISK_BATCH: str = "prediksi_batch_risiko.xlsx"
EXPORT_HIGH_RISK: str = "daftar_risiko_tinggi.xlsx"

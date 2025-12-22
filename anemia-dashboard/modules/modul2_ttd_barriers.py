"""Module 2: Text mining for TTD consumption barriers."""
from __future__ import annotations

import logging
import re
import string
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud

from utils.preprocessing import assign_age_group
from utils.visualization import create_metric_card, apply_styling

logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS: Dict[str, Sequence[str]] = {
    "Lupa": [
        "lupa",
        "sibuk",
        "busy",
        "terlupa",
        "kesibukan",
        "tidak sempat",
        "terburu",
        "capek",
        "lelah",
        "banyak pekerjaan",
    ],
    "Efek Samping": [
        "mual",
        "muntah",
        "pusing",
        "sakit perut",
        "konstipasi",
        "sembelit",
        "tinja hitam",
        "bau",
        "tidak enak",
        "mules",
        "eneg",
    ],
    "Stok Habis": [
        "habis",
        "tidak tersedia",
        "kosong",
        "tidak ada",
        "kehabisan",
        "stok",
        "puskesmas",
        "pustu",
        "gudang",
    ],
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
    "nausea": -1,
    "lega": 1,
    "sehat": 1,
    "khawatir": -1,
    "cemas": -1,
    "takut": -1,
    "senang": 1,
}

COLOR_MAP: Dict[str, str] = {
    "Lupa": "#1C7CBA",
    "Efek Samping": "#E74C3C",
    "Stok Habis": "#F39C12",
    "Lainnya": "#4AA96C",
}

PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)


@dataclass(slots=True)
class TextAnalysisResult:
    """Container for barrier analysis outputs."""

    dataframe: pd.DataFrame
    category_counts: pd.DataFrame
    sentiment_summary: pd.DataFrame
    cooccurrence_matrix: pd.DataFrame


def _load_stopwords() -> List[str]:
    """Load Indonesian stopwords with graceful degradation."""
    try:
        from nltk.corpus import stopwords
        from nltk import download as nltk_download

        try:
            stopwords.words("indonesian")
        except LookupError:
            nltk_download("stopwords")
        return stopwords.words("indonesian")
    except Exception as exc:  # pragma: no cover - fallback path for offline
        logger.warning("Falling back to bundled stopwords due to %s", exc)
        return [
            "dan",
            "yang",
            "di",
            "ke",
            "dari",
            "itu",
            "untuk",
            "dengan",
            "karena",
            "saat",
            "akan",
            "tidak",
            "ada",
            "saya",
            "kami",
            "dia",
        ]


@lru_cache(maxsize=1)
def get_stopword_set() -> set[str]:
    """Return cached stopword set."""
    return set(_load_stopwords())


@lru_cache(maxsize=1)
def get_stemmer():
    """Instantiate and cache Sastrawi stemmer."""
    factory = StemmerFactory()
    return factory.create_stemmer()


def preprocess_text(text: str) -> List[str]:
    """Clean and tokenize Indonesian text.

    Args:
        text: Raw text input from the survey.

    Returns:
        List of normalized tokens.
    """
    if not isinstance(text, str):
        return []
    lower = text.lower().translate(PUNCT_TRANSLATOR)
    tokens = re.split(r"\s+", lower)
    tokens = [token for token in tokens if token]
    stopwords_set = get_stopword_set()
    tokens = [token for token in tokens if token not in stopwords_set]
    stemmer = get_stemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed


def categorize_barrier(tokens: List[str]) -> str:
    """Assign barrier category based on keyword presence."""
    if not tokens:
        return "Lainnya"

    token_set = set(tokens)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in token_set for keyword in keywords):
            return category
    return "Lainnya"


def compute_sentiment(tokens: List[str]) -> str:
    """Approximate sentiment classification using lexicon scoring."""
    if not tokens:
        return "Netral"
    score = sum(SENTIMENT_LEXICON.get(token, 0) for token in tokens)
    if score > 0:
        return "Positif"
    if score < 0:
        return "Negatif"
    return "Netral"


def analyze_barriers(dataframe: pd.DataFrame) -> TextAnalysisResult:
    """Run full NLP pipeline on barrier responses."""
    df = dataframe.copy()
    if "Kendala_TTD" not in df.columns:
        raise ValueError("Kolom 'Kendala_TTD' tidak ditemukan pada dataset.")

    df["Kendala_TTD"] = df["Kendala_TTD"].astype("string").fillna("")
    df["Tokens"] = df["Kendala_TTD"].apply(preprocess_text)
    df["Kategori_Kendala"] = df["Tokens"].apply(categorize_barrier)
    df["Sentimen"] = df["Tokens"].apply(compute_sentiment)

    category_counts = df["Kategori_Kendala"].value_counts().rename_axis("Kategori").reset_index(name="Jumlah")
    sentiment_summary = df.groupby(["Kategori_Kendala", "Sentimen"]).size().reset_index(name="Jumlah")

    cooccurrence = Counter()
    for tokens in df["Tokens"]:
        unique_tokens = sorted(set(tokens))
        for i, token_i in enumerate(unique_tokens):
            for token_j in unique_tokens[i + 1 :]:
                cooccurrence[(token_i, token_j)] += 1

    cooccurrence_df = (
        pd.DataFrame(
            ((token_i, token_j, count) for (token_i, token_j), count in cooccurrence.items()),
            columns=["Token_A", "Token_B", "Frekuensi"],
        )
        .sort_values("Frekuensi", ascending=False)
        .head(30)
    )

    return TextAnalysisResult(
        dataframe=df,
        category_counts=category_counts,
        sentiment_summary=sentiment_summary,
        cooccurrence_matrix=cooccurrence_df,
    )


def generate_wordcloud(tokens_series: pd.Series, color: str) -> Optional[Image.Image]:
    """Produce word cloud image for given tokens."""
    all_tokens = [token for tokens in tokens_series if tokens for token in tokens]
    if not all_tokens:
        return None
    frequencies = Counter(all_tokens)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=None,
        color_func=lambda *args, **kwargs: color,
    ).generate_from_frequencies(frequencies)
    return wordcloud.to_image()


def filter_dataframe(
    dataframe: pd.DataFrame,
    kecamatan: Optional[str],
    age_group: Optional[str],
    trimester: Optional[str],
) -> pd.DataFrame:
    """Filter dataframe based on user selections."""
    filtered = dataframe.copy()
    if kecamatan and kecamatan != "Semua":
        filtered = filtered[filtered["Kecamatan"] == kecamatan]
    if age_group and age_group != "Semua":
        filtered = filtered[filtered["Kelompok_Usia"] == age_group]
    if trimester and trimester != "Semua":
        filtered = filtered[filtered["Trimester"].astype(str) == trimester]
    return filtered


def compute_age_groups(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure age group column exists."""
    df = dataframe.copy()
    if "Kelompok_Usia" not in df.columns and "Usia" in df.columns:
        df["Kelompok_Usia"] = assign_age_group(df["Usia"])
    return df


def barrier_distribution_chart(category_counts: pd.DataFrame) -> go.Figure:
    """Create bar chart for barrier distribution."""
    fig = px.bar(
        category_counts,
        x="Kategori",
        y="Jumlah",
        color="Kategori",
        color_discrete_map=COLOR_MAP,
        labels={"Kategori": "Kategori Kendala", "Jumlah": "Jumlah Responden"},
        title="Distribusi Kendala Konsumsi TTD",
    )
    fig.update_layout(showlegend=False)
    return fig


def breakdown_chart(dataframe: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Generate stacked bar chart for barrier distribution by demographic feature."""
    grouped = dataframe.groupby([column, "Kategori_Kendala"]).size().reset_index(name="Jumlah")
    fig = px.bar(
        grouped,
        x=column,
        y="Jumlah",
        color="Kategori_Kendala",
        barmode="stack",
        color_discrete_map=COLOR_MAP,
        title=title,
    )
    fig.update_layout(xaxis_title=column.replace("_", " "), yaxis_title="Jumlah")
    return fig


def temporal_trend_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Create time series if timestamp available."""
    if "Timestamp" not in dataframe.columns:
        return go.Figure()
    df = dataframe.copy()
    df["Tanggal"] = pd.to_datetime(df["Timestamp"]).dt.to_period("M").dt.to_timestamp()
    trend = df.groupby(["Tanggal", "Kategori_Kendala"]).size().reset_index(name="Jumlah")
    fig = px.line(
        trend,
        x="Tanggal",
        y="Jumlah",
        color="Kategori_Kendala",
        markers=True,
        color_discrete_map=COLOR_MAP,
        title="Tren Kendala dari Waktu ke Waktu",
    )
    return fig


def geographic_heatmap(dataframe: pd.DataFrame) -> go.Figure:
    """Generate heatmap of barrier distribution per kecamatan."""
    pivot = (
        dataframe.groupby(["Kecamatan", "Kategori_Kendala"]).size().reset_index(name="Jumlah")
    )
    pivot_table = pivot.pivot(index="Kecamatan", columns="Kategori_Kendala", values="Jumlah").fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale="YlGnBu",
        )
    )
    fig.update_layout(title="Heatmap Kendala per Kecamatan", xaxis_title="Kategori", yaxis_title="Kecamatan")
    return fig


def download_analyzed_data(dataframe: pd.DataFrame) -> BytesIO:
    """Prepare analyzed dataframe for download."""
    buffer = BytesIO()
    dataframe.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


def dominant_barrier_recommendation(category_counts: pd.DataFrame) -> Tuple[str, List[str]]:
    """Generate actionable recommendations based on top barrier."""
    if category_counts.empty:
        return "Tidak ada data", []
    dominant_row = category_counts.sort_values("Jumlah", ascending=False).iloc[0]
    category = dominant_row["Kategori"]
    if category == "Lupa":
        recommendations = [
            "Implementasi pengingat SMS atau aplikasi untuk jadwal konsumsi TTD.",
            "Libatkan anggota keluarga sebagai pengingat harian.",
            "Distribusikan kalender atau kartu ceklis konsumsi TTD.",
        ]
    elif category == "Efek Samping":
        recommendations = [
            "Berikan konseling tentang cara meminimalkan efek samping (konsumsi setelah makan).",
            "Sediakan materi edukasi tentang variasi waktu konsumsi yang nyaman.",
            "Siapkan hotline bidan untuk konsultasi efek samping yang berat.",
        ]
    elif category == "Stok Habis":
        recommendations = [
            "Perkuat koordinasi rantai pasok untuk memastikan buffer stok minimal 2 minggu.",
            "Jadwalkan distribusi TTD teratur bersama puskesmas/pustu.",
            "Aktifkan pelaporan stok digital agar kekosongan terdeteksi dini.",
        ]
    else:
        recommendations = [
            "Lakukan FGD untuk menggali kendala lain secara lebih rinci.",
            "Perkuat komunikasi dua arah antara tenaga kesehatan dan ibu hamil.",
            "Sesuaikan strategi intervensi berdasarkan kebutuhan lokal.",
        ]
    return category, recommendations


def display_recommendations(category: str, recommendations: Sequence[str]) -> None:
    """Render recommendations in styled boxes."""
    color = COLOR_MAP.get(category, "#34495E")
    st.markdown(
        f"<div style='background:{color}15;border-left:4px solid {color};padding:16px;border-radius:8px;'>"
        f"<strong>Rekomendasi Utama: {category}</strong><ul>"
        + "".join(f"<li>{rec}</li>" for rec in recommendations)
        + "</ul></div>",
        unsafe_allow_html=True,
    )


def render(dataframe: pd.DataFrame) -> None:
    """Render Module 2 Streamlit UI."""
    st.title("Analisis Kendala Konsumsi TTD")
    st.markdown(
        "Modul ini menelusuri kendala utama konsumsi Tablet Tambah Darah (TTD) berdasarkan data survei."
    )
    st.markdown(apply_styling(), unsafe_allow_html=True)

    try:
        dataframe = compute_age_groups(dataframe)
        analysis = analyze_barriers(dataframe)
    except ValueError as exc:
        st.error(str(exc))
        logger.exception("Barrier analysis failed: %s", exc)
        return

    with st.sidebar:
        st.subheader("Filter Data")
        kecamatan_options = ["Semua"] + sorted(dataframe["Kecamatan"].dropna().unique().tolist())
        age_options = ["Semua"] + sorted(dataframe["Kelompok_Usia"].dropna().astype(str).unique().tolist())
        trimester_options = ["Semua"] + sorted(dataframe["Trimester"].dropna().astype(str).unique().tolist())

        selected_kecamatan = st.selectbox("Pilih Kecamatan", options=kecamatan_options)
        selected_age = st.selectbox("Kelompok Usia", options=age_options)
        selected_trimester = st.selectbox("Trimester", options=trimester_options)

    filtered_df = filter_dataframe(
        analysis.dataframe,
        selected_kecamatan,
        selected_age,
        selected_trimester,
    )

    filtered_counts = filtered_df["Kategori_Kendala"].value_counts().rename_axis("Kategori").reset_index(name="Jumlah")
    most_common = (
        filtered_counts.sort_values("Jumlah", ascending=False).iloc[0]["Kategori"]
        if not filtered_counts.empty
        else "-"
    )

    metric_cols = st.columns(3)
    metric_cols[0].markdown(
        create_metric_card(
            label="Total Respons",
            value=str(len(filtered_df)),
            icon="📊",
        ),
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        create_metric_card(
            label="Dominan",
            value=most_common,
            icon="🏷️",
        ),
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        create_metric_card(
            label="Kategori",
            value=str(filtered_counts.shape[0]),
            icon="🧭",
        ),
        unsafe_allow_html=True,
    )

    st.plotly_chart(barrier_distribution_chart(filtered_counts), use_container_width=True)

    tabs = st.tabs([
        "Word Cloud",
        "Sebaran Demografis",
        "Analisis Sentimen",
        "Ko-Occurence",
        "Tren Waktu",
        "Peta Kendala",
    ])

    with tabs[0]:
        wc_cols = st.columns(len(COLOR_MAP))
        for idx, (category, color) in enumerate(COLOR_MAP.items()):
            category_tokens = filtered_df.loc[
                filtered_df["Kategori_Kendala"] == category, "Tokens"
            ]
            image = generate_wordcloud(category_tokens, color=color)
            if image:
                wc_cols[idx % len(wc_cols)].image(image, caption=category, use_container_width=True)
            else:
                wc_cols[idx % len(wc_cols)].info(f"Tidak ada data untuk {category}.")

    with tabs[1]:
        st.plotly_chart(
            breakdown_chart(filtered_df, "Kelompok_Usia", "Sebaran Kendala per Kelompok Usia"),
            use_container_width=True,
        )
        st.plotly_chart(
            breakdown_chart(filtered_df, "Trimester", "Sebaran Kendala per Trimester"),
            use_container_width=True,
        )
        st.plotly_chart(
            breakdown_chart(filtered_df, "Pendidikan", "Sebaran Kendala per Pendidikan"),
            use_container_width=True,
        )

    with tabs[2]:
        sentiment_fig = px.bar(
            filtered_df.groupby(["Kategori_Kendala", "Sentimen"]).size().reset_index(name="Jumlah"),
            x="Kategori_Kendala",
            y="Jumlah",
            color="Sentimen",
            title="Analisis Sentimen per Kategori",
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)

    with tabs[3]:
        st.dataframe(
            analysis.cooccurrence_matrix,
            use_container_width=True,
            hide_index=True,
        )

    with tabs[4]:
        trend_fig = temporal_trend_chart(filtered_df)
        if trend_fig.data:
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("Data timestamp tidak tersedia untuk analisis tren.")

    with tabs[5]:
        st.plotly_chart(geographic_heatmap(filtered_df), use_container_width=True)

    st.subheader("Daftar Respons dan Kategori")
    st.dataframe(
        filtered_df[
            [
                "Nama",
                "Kecamatan",
                "Desa",
                "Trimester",
                "Kelompok_Usia",
                "Kendala_TTD",
                "Kategori_Kendala",
                "Sentimen",
            ]
        ],
        use_container_width=True,
    )

    download_buffer = download_analyzed_data(filtered_df)
    st.download_button(
        label="Unduh Hasil Analisis",
        data=download_buffer,
        file_name="analisis_kendala_ttd.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Rekomendasi Tindak Lanjut")
    dominant_category, recommendations = dominant_barrier_recommendation(filtered_counts)
    display_recommendations(dominant_category, recommendations)

    st.markdown("### Fitur Lanjutan")
    st.info(
        "Segmentasi sentimen dan pelacakan intervensi dapat dikembangkan lebih lanjut untuk "
        "menilai efektivitas program secara berkala."
    )

    st.caption("Analisis kendala TTD membantu merancang strategi peningkatan kepatuhan konsumsi.")

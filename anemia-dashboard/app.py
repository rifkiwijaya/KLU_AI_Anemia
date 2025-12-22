"""Streamlit application entry point for the anemia intelligence dashboard."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from modules import modul1_hotspot_mapping, modul2_ttd_barriers, modul3_risk_segmentation
from utils.data_loader import load_excel_dataset
from utils.preprocessing import assign_age_group
from utils.visualization import (
    apply_styling,
    create_anemia_pie_chart,
    create_metric_card,
)

APP_VERSION = "v0.1.0"
DEFAULT_DATA_FILE = config.DATA_FILE
CUSTOM_CSS_PATH = Path("assets/custom.css")
LOGO_PATH = Path("assets/logo.png")

NAVIGATION_LABELS: Dict[str, str] = {
    "🏠 Dashboard Utama": "home",
    "📍 Modul 1: Peta Hotspot Anemia": "module_1",
    "📋 Modul 2: Analisis Kendala TTD": "module_2",
    "🎯 Modul 3: Segmentasi & Rekomendasi Risiko": "module_3",
    "ℹ️ Tentang / About": "about",
}


def configure_logging(log_level: int = logging.INFO) -> None:
    """Configure global logging."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_logo() -> Optional[bytes]:
    """Load logo bytes when available."""
    if LOGO_PATH.exists():
        try:
            return LOGO_PATH.read_bytes()
        except OSError as exc:
            logging.warning("Unable to read logo: %s", exc)
    return None


@st.cache_data(show_spinner=False)
def load_custom_css(path: str) -> str:
    """Read custom CSS from disk."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.warning("Custom CSS not found at %s", path)
        return ""


def resolve_data_path() -> Path:
    """Determine data file path using config or defaults."""
    try:
        import config  # type: ignore

        data_file = getattr(config, "DATA_FILE", DEFAULT_DATA_FILE)
        return Path(data_file)
    except ImportError:
        pass

    if "data_file" in st.secrets:
        return Path(str(st.secrets["data_file"]))

    return DEFAULT_DATA_FILE


@st.cache_data(show_spinner=False)
def load_dataset(path_str: str) -> pd.DataFrame:
    """Load the survey dataset from disk."""
    return load_excel_dataset(Path(path_str))


def inject_styling() -> None:
    """Apply global styling to the Streamlit app."""
    custom_css = load_custom_css(str(CUSTOM_CSS_PATH))
    if custom_css:
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
    st.markdown(apply_styling(), unsafe_allow_html=True)


def add_footer() -> None:
    """Render footer with credits."""
    st.markdown(
        "<footer class='app-footer'>"
        "Anemia Risk Intelligence Dashboard · "
        f"{APP_VERSION} · Dibangun untuk penelitian KLU · "
        "Kontak: <a href='mailto:prayaadhiganaglobal@gmail.com'>prayaadhiganaglobal@gmail.com</a>"
        "</footer>",
        unsafe_allow_html=True,
    )


def compute_metrics(dataframe: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Compute headline metrics with fallbacks."""
    defaults = {
        "total_respondents": 388.0,
        "anemia_prevalence": 24.5,
        "average_hb": 11.6,
        "ttd_compliance": 45.1,
    }
    if dataframe is None or dataframe.empty:
        return defaults

    metrics = defaults.copy()
    metrics["total_respondents"] = float(len(dataframe))

    if "Status_Anemia" in dataframe.columns:
        anemia_cases = dataframe["Status_Anemia"].astype(str).str.contains("anemia", case=False).sum()
        metrics["anemia_prevalence"] = round((anemia_cases / max(len(dataframe), 1)) * 100, 2)

    if "Hb" in dataframe.columns:
        metrics["average_hb"] = round(float(dataframe["Hb"].mean(skipna=True)), 2)

    if "Kepatuhan_TTD" in dataframe.columns:
        metrics["ttd_compliance"] = round(float(dataframe["Kepatuhan_TTD"].mean(skipna=True)), 2)

    return metrics


def build_anemia_by_age_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Create anemia distribution chart by age group."""
    if "Usia" not in dataframe.columns or "Status_Anemia" not in dataframe.columns:
        return go.Figure()

    age_df = dataframe.copy()
    age_df["Kelompok_Usia"] = assign_age_group(age_df["Usia"]).astype(str)
    grouped = (
        age_df.groupby(["Kelompok_Usia", "Status_Anemia"])
        .size()
        .reset_index(name="Jumlah")
        .sort_values("Kelompok_Usia")
    )
    return px.bar(
        grouped,
        x="Kelompok_Usia",
        y="Jumlah",
        color="Status_Anemia",
        barmode="stack",
        labels={"Kelompok_Usia": "Kelompok Usia", "Jumlah": "Jumlah"},
        title="Distribusi Anemia per Kelompok Usia",
    )


def build_anemia_by_trimester_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Create anemia distribution chart by trimester."""
    if "Trimester" not in dataframe.columns or "Status_Anemia" not in dataframe.columns:
        return go.Figure()

    grouped = (
        dataframe.groupby(["Trimester", "Status_Anemia"])
        .size()
        .reset_index(name="Jumlah")
        .sort_values("Trimester")
    )
    return px.bar(
        grouped,
        x="Trimester",
        y="Jumlah",
        color="Status_Anemia",
        barmode="group",
        labels={"Trimester": "Trimester", "Jumlah": "Jumlah"},
        title="Anemia per Trimester",
    )


def build_ttd_vs_anemia_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Plot TTD compliance grouped by anemia status."""
    if "Status_Anemia" not in dataframe.columns or "Kepatuhan_TTD" not in dataframe.columns:
        return go.Figure()

    grouped = dataframe.groupby("Status_Anemia")["Kepatuhan_TTD"].mean().reset_index()
    fig = px.bar(
        grouped,
        x="Status_Anemia",
        y="Kepatuhan_TTD",
        color="Status_Anemia",
        labels={"Status_Anemia": "Status", "Kepatuhan_TTD": "Rata-rata Kepatuhan (%)"},
        title="Kepatuhan TTD menurut Status Anemia",
    )
    fig.update_yaxes(range=[0, max(100, grouped["Kepatuhan_TTD"].max() * 1.1)])
    return fig


def render_home(dataframe: Optional[pd.DataFrame]) -> None:
    """Render the main dashboard view."""
    metrics = compute_metrics(dataframe)

    st.markdown(
        "<section class='hero'><h2>Anemia Risk Intelligence</h2>"
        "<p>Dashboard ini mendukung pemetaan risiko anemia ibu hamil di Lombok Utara,"
        " menghubungkan analitik spasial, text mining, dan pembelajaran mesin untuk rekomendasi intervensi.</p>"
        "</section>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].markdown(
        create_metric_card("Total Responden", f"{metrics['total_respondents']:.0f}", icon="👩‍🍼"),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        create_metric_card("Prevalensi Anemia", f"{metrics['anemia_prevalence']:.1f}%", icon="🩸"),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        create_metric_card("Rata-rata Hb", f"{metrics['average_hb']:.1f} g/dL", icon="🧪"),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        create_metric_card("Kepatuhan TTD", f"{metrics['ttd_compliance']:.1f}%", icon="💊"),
        unsafe_allow_html=True,
    )

    if dataframe is None or dataframe.empty:
        st.warning("Data belum tersedia. Unggah dataset pada menu batch prediction untuk memulai analisis.")
        return

    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(create_anemia_pie_chart(dataframe), use_container_width=True)
    chart_cols[1].plotly_chart(build_anemia_by_age_chart(dataframe), use_container_width=True)

    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(build_anemia_by_trimester_chart(dataframe), use_container_width=True)
    chart_cols[1].plotly_chart(build_ttd_vs_anemia_chart(dataframe), use_container_width=True)

    st.markdown("### Insight Singkat")
    st.markdown(
        "- Pantau desa dengan prevalensi anemia >30% melalui Modul 1 untuk intervensi cepat.\n"
        "- Identifikasi kendala TTD dominan guna perancangan edukasi sasaran di Modul 2.\n"
        "- Manfaatkan Modul 3 untuk simulasi scenario dan penentuan tindak lanjut individual."
    )

    cta_cols = st.columns(3)
    if cta_cols[0].button("Buka Peta Hotspot", use_container_width=True):
        st.session_state["navigation_pending"] = "📍 Modul 1: Peta Hotspot Anemia"
        st.rerun()
    if cta_cols[1].button("Analisis Kendala TTD", use_container_width=True):
        st.session_state["navigation_pending"] = "📋 Modul 2: Analisis Kendala TTD"
        st.rerun()
    if cta_cols[2].button("Segmentasi Risiko", use_container_width=True):
        st.session_state["navigation_pending"] = "🎯 Modul 3: Segmentasi & Rekomendasi Risiko"
        st.rerun()


def render_about() -> None:
    """Render the about page."""
    st.header("Tentang Proyek")
    st.markdown(
        "Dashboard ini dikembangkan untuk mendukung penelitian anemia ibu hamil di Kabupaten Lombok Utara."
        " Analisis mencakup pemetaan spasial, text mining kendala konsumsi TTD, dan segmentasi risiko berbasis machine learning."
    )

    st.subheader("Metodologi")
    st.markdown(
        "1. **Pemetaan Hotspot**: Clustering K-Means pada prevalensi anemia per desa/kecamatan dengan visualisasi peta interaktif.\n"
        "2. **Analisis Kendala TTD**: Pemrosesan bahasa alami berbahasa Indonesia menggunakan stemming Sastrawi dan word cloud tematik.\n"
        "3. **Segmentasi Risiko**: Random Forest Classifier dengan tuning hyperparameter dan simulasi what-if untuk rekomendasi intervensi."
    )

    st.subheader("Sumber Data & Batasan")
    st.markdown(
        "- Data berasal dari survei ibu hamil di Lombok Utara (formulir Google).\n"
        "- Potensi bias: self-reporting dan keterbatasan ukuran sampel."
    )

    st.subheader("Kontak")
    st.markdown("Email: prayaadhiganaglobal@gmail.com · WhatsApp: +62 812-3456-7890")

    st.subheader("Versi")
    st.markdown(f"Dashboard version: **{APP_VERSION}** · Pembaruan terakhir: Desember 2025")


def render_error(message: str) -> None:
    """Display a user-friendly error block."""
    st.error(message)
    st.markdown(
        "Silakan periksa kembali file data atau hubungi administrator bila masalah berlanjut."
    )


def main() -> None:
    """Run the Streamlit application."""
    configure_logging()
    st.set_page_config(
        page_title="Anemia Risk Intelligence",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styling()

    default_label = list(NAVIGATION_LABELS.keys())[0]
    pending_label = st.session_state.pop("navigation_pending", None)

    if "navigation" not in st.session_state:
        st.session_state["navigation"] = default_label

    if pending_label is not None:
        st.session_state["navigation"] = pending_label
        st.session_state["navigation_widget"] = pending_label

    st.session_state.setdefault("navigation_widget", st.session_state["navigation"])

    sidebar_logo = load_logo()
    with st.sidebar:
        if sidebar_logo:
            st.image(sidebar_logo, use_container_width=True)
        st.title("Navigasi")
        selected_label = st.radio(
            "Pilih Modul",
            options=list(NAVIGATION_LABELS.keys()),
            key="navigation_widget",
        )

    st.session_state["navigation"] = selected_label

    data_error: Optional[str] = None
    dataset: Optional[pd.DataFrame] = None
    data_path = resolve_data_path()

    try:
        dataset = load_dataset(str(data_path))
    except FileNotFoundError:
        data_error = f"File data tidak ditemukan pada lokasi: {data_path}"
    except Exception as exc:  # pragma: no cover - guard unexpected errors
        logging.exception("Data loading failed: %s", exc)
        data_error = f"Gagal memuat data: {exc}"

    selected_key = NAVIGATION_LABELS[selected_label]

    if selected_key == "home":
        render_home(dataset if data_error is None else None)
    elif selected_key == "module_1":
        if data_error:
            render_error(data_error)
        else:
            modul1_hotspot_mapping.render(dataset)
    elif selected_key == "module_2":
        if data_error:
            render_error(data_error)
        else:
            modul2_ttd_barriers.render(dataset)
    elif selected_key == "module_3":
        if data_error:
            render_error(data_error)
        else:
            modul3_risk_segmentation.render(dataset)
    elif selected_key == "about":
        render_about()

    add_footer()


if __name__ == "__main__":
    main()

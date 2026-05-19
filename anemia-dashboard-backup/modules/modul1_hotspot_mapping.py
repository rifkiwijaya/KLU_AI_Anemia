"""Module 1: Spatial hotspot mapping for anemia prevalence."""
from __future__ import annotations

import hashlib
import logging
from io import BytesIO
from typing import Dict, List, Tuple

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium
import plotly.express as px

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "Desa",
    "Kecamatan",
    "Hb",
    "Status_Anemia",
)

OPTIONAL_METRICS: Dict[str, str] = {
    "Avg_Hb": "Hb",
    "Avg_LILA": "LILA",
    "Avg_Jarak_Faskes": "Jarak_Faskes",
    "Avg_Frekuensi_Protein_Hewani": "Frekuensi_Protein_Hewani",
    "Avg_Kepatuhan_TTD": "Kepatuhan_TTD",
}

MAP_CENTER: Tuple[float, float] = (-8.354, 116.277)  # Lombok Utara approx
RISK_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "High": (30.0, float("inf")),
    "Medium": (15.0, 30.0),
    "Low": (0.0, 15.0),
}
RISK_COLORS: Dict[str, str] = {
    "High": "#E74C3C",
    "Medium": "#F39C12",
    "Low": "#27AE60",
}


@st.cache_data(show_spinner=False)
def aggregate_location_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate anemia metrics per village-subdistrict cluster.

    Args:
        dataframe: Cleaned survey dataframe.

    Returns:
        Aggregated dataframe with prevalence and supporting metrics.
    """
    group_cols = ["Desa", "Kecamatan"]
    missing_cols = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_cols:
        raise ValueError("Dataset missing required columns: %s" % ", ".join(missing_cols))

    df = dataframe.copy()
    status_labels = df["Status_Anemia"].astype(str).str.strip().str.lower()
    df["Anemia_Flag"] = status_labels.eq("anemia")

    aggregation = {
        "Nama": "count",
        "Anemia_Flag": "sum",
        "Hb": "mean",
    }

    for metric_alias, source_column in OPTIONAL_METRICS.items():
        if source_column in df.columns:
            aggregation[source_column] = "mean"

    grouped = df.groupby(group_cols, dropna=False).agg(aggregation).reset_index()
    grouped.rename(columns={"Nama": "Respondent_Count", "Anemia_Flag": "Anemia_Cases"}, inplace=True)
    grouped["Anemia_Prevalence"] = (
        grouped["Anemia_Cases"] / grouped["Respondent_Count"] * 100
    ).round(2)
    grouped["Avg_Hb"] = grouped.pop("Hb").round(2)

    if "LILA" in aggregation:
        grouped["Avg_LILA"] = grouped.pop("LILA").round(2)
    if "Jarak_Faskes" in aggregation:
        grouped["Avg_Jarak_Faskes"] = grouped.pop("Jarak_Faskes").round(2)
    if "Frekuensi_Protein_Hewani" in aggregation:
        grouped["Avg_Frekuensi_Protein_Hewani"] = grouped.pop("Frekuensi_Protein_Hewani").round(2)
    if "Kepatuhan_TTD" in aggregation:
        grouped["Avg_Kepatuhan_TTD"] = grouped.pop("Kepatuhan_TTD").round(2)

    coords = grouped.apply(
        lambda row: generate_coordinate_identifier(f"{row['Desa']}_{row['Kecamatan']}")
        if pd.notna(row["Desa"]) and pd.notna(row["Kecamatan"])
        else MAP_CENTER,
        axis=1,
        result_type="expand",
    )
    grouped["Latitude"] = coords[0]
    grouped["Longitude"] = coords[1]

    return grouped


def generate_coordinate_identifier(identifier: str, radius: float = 0.2) -> Tuple[float, float]:
    """Generate pseudo coordinates around Lombok Utara for a given identifier.

    Args:
        identifier: Location name string.
        radius: Maximum offset in degrees from the base coordinate.

    Returns:
        Tuple of latitude and longitude.
    """
    digest = hashlib.md5(identifier.encode("utf-8"), usedforsecurity=False).hexdigest()
    int_lat = int(digest[:8], 16)
    int_lon = int(digest[8:16], 16)
    lat_offset = ((int_lat % 1000) / 1000.0 - 0.5) * radius
    lon_offset = ((int_lon % 1000) / 1000.0 - 0.5) * radius
    return MAP_CENTER[0] + lat_offset, MAP_CENTER[1] + lon_offset


def classify_risk(prevalence: float) -> str:
    """Assign risk label based on anemia prevalence."""
    for label, (lower, upper) in RISK_THRESHOLDS.items():
        if lower <= prevalence < upper:
            return label
    return "Low"


def determine_optimal_k(
    feature_matrix: np.ndarray,
    min_k: int = 2,
    max_k: int = 6,
    random_state: int = 42,
) -> int:
    """Select optimal k using silhouette score.

    Args:
        feature_matrix: Scaled feature matrix for clustering.
        min_k: Minimum cluster count to evaluate.
        max_k: Maximum cluster count to evaluate.
        random_state: Reproducibility seed.

    Returns:
        Optimal number of clusters.
    """
    if feature_matrix.shape[0] < min_k:
        return feature_matrix.shape[0]

    best_k = min_k
    best_score = -1.0

    for k in range(min_k, min(max_k, feature_matrix.shape[0]) + 1):
        with np.errstate(all="ignore"):
            model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels = model.fit_predict(feature_matrix)
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(feature_matrix, labels)
            if score > best_score:
                best_score = score
                best_k = k

    return best_k


def run_clustering(
    dataframe: pd.DataFrame,
    selected_features: List[str],
    desired_k: int,
    enable_auto_k: bool = True,
) -> Tuple[pd.DataFrame, KMeans]:
    """Execute KMeans clustering and append results to dataframe.

    Args:
        dataframe: Aggregated dataframe ready for clustering.
        selected_features: Features used for clustering.
        desired_k: Slider-selected number of clusters.
        enable_auto_k: Whether to override with automatic k selection.

    Returns:
        Tuple with dataframe (including cluster columns) and fitted model.
    """
    features = selected_features.copy()
    if "Anemia_Prevalence" not in features:
        features.insert(0, "Anemia_Prevalence")

    available_features = [feature for feature in features if feature in dataframe.columns]
    if not available_features:
        raise ValueError("No valid features available for clustering")

    feature_df = dataframe[available_features].apply(pd.to_numeric, errors="coerce")
    column_means = feature_df.mean(numeric_only=True)
    feature_df = feature_df.fillna(column_means)
    feature_df = feature_df.fillna(0)

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_df)

    dataframe = dataframe.copy()
    dataframe.loc[:, available_features] = feature_df

    if enable_auto_k:
        optimal_k = determine_optimal_k(feature_matrix, min_k=2, max_k=desired_k)
    else:
        optimal_k = desired_k

    optimal_k = max(2, min(optimal_k, feature_matrix.shape[0]))

    model = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
    dataframe = dataframe.copy()
    dataframe["Cluster"] = model.fit_predict(feature_matrix)
    dataframe["Cluster_Label"] = dataframe["Cluster"].apply(lambda c: f"Cluster {c + 1}")

    return dataframe, model


def create_map(clustered_df: pd.DataFrame) -> folium.Map:
    """Build Folium map with cluster markers."""
    fmap = folium.Map(location=MAP_CENTER, zoom_start=10, control_scale=True)
    marker_cluster = MarkerCluster().add_to(fmap)

    for _, row in clustered_df.iterrows():
        risk = classify_risk(row["Anemia_Prevalence"])
        color = RISK_COLORS.get(risk, "#27AE60")
        popup_html = (
            f"<strong>{row['Desa']}, {row['Kecamatan']}</strong><br>"
            f"Total Responden: {row['Respondent_Count']}<br>"
            f"Kasus Anemia: {row['Anemia_Cases']}<br>"
            f"Prevalensi: {row['Anemia_Prevalence']:.2f}%<br>"
            f"Rata-rata Hb: {row['Avg_Hb']:.2f} g/dL<br>"
            f"Cluster: {row['Cluster_Label']}<br>"
            f"Risiko: {risk}"
        )
        folium.CircleMarker(
            location=(row["Latitude"], row["Longitude"]),
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(marker_cluster)

    add_map_legend(fmap)
    return fmap


def add_map_legend(fmap: folium.Map) -> None:
    """Attach custom legend to the Folium map."""
    legend_html = (
        "<div style=\"position: fixed; bottom: 30px; left: 30px; width: 180px;"
        " background-color: white; border-radius: 8px; padding: 12px;"
        " box-shadow: 0 0 8px rgba(0,0,0,0.15); font-size: 0.85rem;\">"
        "<strong>Kategori Risiko</strong><br>"
        + "<br>".join(
            f"<span style='color:{color};'>&#9679;</span> {label}" for label, color in RISK_COLORS.items()
        )
        + "</div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend_html))


def render(dataframe: pd.DataFrame) -> None:
    """Render the Streamlit interface for Module 1."""
    st.title("Hotspot Mapping: Sebaran Risiko Anemia")
    st.markdown(
        "Modul ini mengidentifikasi wilayah prioritas berdasarkan prevalensi anemia, "
        "menggunakan clustering spasial untuk mendukung distribusi intervensi."
    )

    try:
        aggregated_df = aggregate_location_metrics(dataframe)
    except ValueError as exc:
        st.error(str(exc))
        logger.exception("Aggregation failed: %s", exc)
        return

    with st.sidebar:
        st.subheader("Pengaturan Clustering")
        cluster_slider = st.slider("Jumlah cluster", 3, 7, 3, key="hotspot_cluster_slider")
        auto_k = st.checkbox("Gunakan rekomendasi otomatis", value=True, key="hotspot_auto_k")

        available_features = [
            "Anemia_Prevalence",
            "Avg_Hb",
            "Avg_LILA",
            "Avg_Jarak_Faskes",
            "Avg_Frekuensi_Protein_Hewani",
            "Avg_Kepatuhan_TTD",
        ]
        present_features = [feature for feature in available_features if feature in aggregated_df.columns]
        default_selection = [feature for feature in present_features if feature != "Avg_LILA"]
        selected_features = st.multiselect(
            "Fitur clustering",
            options=present_features,
            default=default_selection or present_features,
            help="Pilih indikator tambahan selain prevalensi untuk analisis cluster.",
        )
        recalc = st.button("Hitung ulang", key="hotspot_recalc")

    run_analysis = recalc or "hotspot_cluster_cache" not in st.session_state

    if not run_analysis:
        clustered_df = st.session_state["hotspot_cluster_cache"]["data"]
        model = st.session_state["hotspot_cluster_cache"]["model"]
        optimal_k = model.n_clusters
    else:
        with st.spinner("Menjalankan clustering spasial..."):
            try:
                clustered_df, model = run_clustering(
                    aggregated_df,
                    selected_features or ["Anemia_Prevalence"],
                    desired_k=cluster_slider,
                    enable_auto_k=auto_k,
                )
            except ValueError as exc:
                st.error(str(exc))
                logger.exception("Clustering failed: %s", exc)
                return
            st.session_state["hotspot_cluster_cache"] = {"data": clustered_df, "model": model}
            optimal_k = model.n_clusters

    st.caption(f"Jumlah cluster aktif: {optimal_k}")

    fmap = create_map(clustered_df)
    st_folium(fmap, width=1400, height=520)

    st.subheader("Ringkasan Cluster")
    summary_columns = [
        "Desa",
        "Kecamatan",
        "Cluster_Label",
        "Respondent_Count",
        "Anemia_Cases",
        "Anemia_Prevalence",
        "Avg_Hb",
    ]
    extra_columns = [
        column for column in clustered_df.columns if column.startswith("Avg_") and column not in summary_columns
    ]
    summary_df = clustered_df[summary_columns + extra_columns]
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    export_buffer = BytesIO()
    summary_df.to_excel(export_buffer, index=False)
    export_buffer.seek(0)
    st.download_button(
        label="Unduh hasil cluster (Excel)",
        data=export_buffer,
        file_name="hasil_cluster_anemia.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Distribusi Prevalensi per Cluster")
    chart_df = clustered_df.groupby("Cluster_Label")["Anemia_Prevalence"].mean().reset_index()
    bar_fig = px.bar(
        chart_df,
        x="Cluster_Label",
        y="Anemia_Prevalence",
        color="Cluster_Label",
        labels={"Cluster_Label": "Cluster", "Anemia_Prevalence": "Prevalensi (%)"},
        title="Rata-rata prevalensi anemia per cluster",
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("Prioritas Penanganan")
    high_risk = clustered_df[clustered_df["Anemia_Prevalence"] >= RISK_THRESHOLDS["High"][0]]
    
    # Store artifacts for export
    export_artifacts = {
        "hotspot_map": {"type": "folium", "object": fmap},
        "cluster_summary_table": {"type": "dataframe", "object": summary_df, "title": "Ringkasan Cluster"},
        "prevalence_by_cluster": {"type": "plotly", "object": bar_fig},
    }
    
    if not high_risk.empty:
        top_locations = high_risk.sort_values("Anemia_Prevalence", ascending=False).head(5)
        st.write("Wilayah prioritas (prevalensi > 30%):")
        st.table(top_locations[["Desa", "Kecamatan", "Anemia_Prevalence", "Avg_Hb", "Respondent_Count"]])
        export_artifacts["priority_locations"] = {
            "type": "dataframe",
            "object": top_locations[["Desa", "Kecamatan", "Anemia_Prevalence", "Avg_Hb", "Respondent_Count"]],
            "title": "Wilayah Prioritas"
        }
    else:
        st.info("Tidak ada cluster dengan risiko tinggi berdasarkan ambang saat ini.")
    
    # Store artifacts in session state for export
    st.session_state["module_1_export_artifacts"] = export_artifacts

    st.markdown("**Rekomendasi Alokasi:**")
    st.markdown(
        "- Fokuskan distribusi TTD dan pemantauan intensif pada cluster risiko tinggi.\n"
        "- Tingkatkan edukasi gizi dan akses pangan bergizi di wilayah risiko tinggi dan sedang.\n"
        "- Koordinasikan transportasi dan layanan antenatal pada desa dengan jarak faskes tinggi."
    )

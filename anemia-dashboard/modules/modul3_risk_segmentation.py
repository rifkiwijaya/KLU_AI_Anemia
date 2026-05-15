"""Module 3: Risk segmentation and intervention recommendations."""
from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize

from utils.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    assign_age_group,
    categorize_hb_status,
    compute_risk_score,
    create_interaction_features,
    derive_kek_status,
)
from utils.visualization import create_metric_card, apply_styling

logger = logging.getLogger(__name__)

RISK_LEVELS: List[str] = ["Low", "Medium", "High"]
MODEL_PATH = Path("models/risk_classifier.joblib")

INTERVENTION_RECOMMENDATIONS: Dict[str, Sequence[str]] = {
    "High": [
        "Lakukan pemeriksaan Hb segera dan konsultasi medis lanjutan.",
        "Pastikan konsumsi TTD harian dengan pengawasan tenaga kesehatan.",
        "Monitoring kondisi minimal satu kali per minggu.",
        "Berikan konseling gizi intensif dan pantau asupan protein.",
        "Evaluasi kebutuhan transfusi darah bila gejala berat.",
        "Rujuk ke klinik kehamilan risiko tinggi untuk pengawasan khusus.",
    ],
    "Medium": [
        "Pastikan konsumsi TTD minimal 90 tablet selama kehamilan.",
        "Lakukan monitoring dua mingguan oleh bidan atau tenaga kesehatan.",
        "Fokuskan edukasi peningkatan kualitas diet sehari-hari.",
        "Berikan materi edukasi manajemen efek samping TTD.",
        "Distribusikan contoh menu tinggi zat besi dan hewani.",
    ],
    "Low": [
        "Pertahankan konsumsi TTD sesuai anjuran tenaga kesehatan.",
        "Monitoring rutin setiap bulan untuk pencegahan.",
        "Dorong penerapan pola makan bergizi seimbang.",
        "Berikan edukasi umum kehamilan dan kebersihan diri.",
        "Perkuat edukasi pencegahan anemia bagi keluarga.",
    ],
}

RISK_COLORS: Dict[str, str] = {
    "High": "#E74C3C",
    "Medium": "#F39C12",
    "Low": "#2ECC71",
}

NUMERIC_ENRICHED: List[str] = list(NUMERIC_FEATURES) + [
    "Kepatuhan_TTD",  # ensuring numeric
    "Hb_x_Kepatuhan",
    "LILA_to_Protein",
]

CATEGORICAL_ENRICHED: List[str] = list(CATEGORICAL_FEATURES) + [
    "Kelompok_Usia",
]


@dataclass(slots=True)
class RiskModelArtifacts:
    """Container for trained model assets and evaluation outputs."""

    model: Pipeline
    label_encoder: LabelEncoder
    feature_names: List[str]
    classification_report: pd.DataFrame
    confusion_matrix: np.ndarray
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]]
    metrics: Dict[str, float]
    feature_importances: pd.DataFrame
    predictions: pd.DataFrame
    neighbors: NearestNeighbors
    transformed_features: np.ndarray
    X: pd.DataFrame
    y: np.ndarray


def escalate_risk(current: str) -> str:
    """Move risk level one step higher when threshold triggered."""
    idx = RISK_LEVELS.index(current)
    return RISK_LEVELS[min(idx + 1, len(RISK_LEVELS) - 1)]


def derive_risk_label(row: pd.Series) -> Optional[str]:
    """Derive risk label from clinical and behavioral indicators."""
    hb = pd.to_numeric(row.get("Hb"), errors="coerce")
    if pd.isna(hb):
        return None

    if hb < 9:
        risk = "High"
    elif hb < 11:
        risk = "Medium"
    else:
        risk = "Low"

    lila = pd.to_numeric(row.get("LILA"), errors="coerce")
    if pd.notna(lila) and lila < 23.5:
        risk = escalate_risk(risk)

    compliance = pd.to_numeric(row.get("Kepatuhan_TTD"), errors="coerce")
    if pd.notna(compliance):
        if compliance < 50:
            risk = escalate_risk(escalate_risk(risk))
        elif compliance < 80:
            risk = escalate_risk(risk)

    age = pd.to_numeric(row.get("Usia"), errors="coerce")
    if pd.notna(age) and (age < 20 or age > 35):
        risk = escalate_risk(risk)

    trimester = str(row.get("Trimester", "")).strip()
    if trimester == "3" and hb < 11:
        risk = escalate_risk(risk)

    return risk


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering consistent with training pipeline."""
    df = dataframe.copy()

    numeric_to_coerce = [
        "Usia",
        "Pendapatan",
        "Jarak_Faskes",
        "Hb",
        "LILA",
        "Frekuensi_Protein_Hewani",
        "Kepatuhan_TTD",
    ]
    for column in numeric_to_coerce:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Kepatuhan_TTD"] = df.get("Kepatuhan_TTD", 0).clip(lower=0, upper=100)

    if "Kelompok_Usia" not in df.columns and "Usia" in df.columns:
        df["Kelompok_Usia"] = assign_age_group(df["Usia"]).astype(str)
    else:
        df["Kelompok_Usia"] = df.get("Kelompok_Usia", "Tidak diketahui").astype(str)

    # Keep derived columns only if they exist in raw data (don't create them)
    if "Kategori_Hb" in df.columns:
        df["Kategori_Hb"] = df["Kategori_Hb"].astype("string").fillna("Tidak diketahui").astype(str)

    if "Status_Anemia" in df.columns:
        df["Status_Anemia"] = df["Status_Anemia"].astype("string").fillna("Non-Anemia").astype(str)

    if "Status_KEK" in df.columns:
        df["Status_KEK"] = df["Status_KEK"].astype("string").fillna("Tidak Berisiko").astype(str)

    df["Risk_Score"] = compute_risk_score(df)
    df = create_interaction_features(df)

    if "Hb_x_Kepatuhan" not in df.columns:
        df["Hb_x_Kepatuhan"] = 0.0
    if "LILA_to_Protein" not in df.columns:
        df["LILA_to_Protein"] = 0.0

    categorical_fill = {
        "Pendidikan": "Tidak diketahui",
        "Trimester": "Tidak diketahui",
        "Gravida": "Tidak diketahui",
        "Terima_TTD": "Tidak",
        "Konsumsi_Inhibitor": "Tidak diketahui",
        "Akses_Air": "Tidak diketahui",
        "Cuci_Tangan": "Tidak diketahui",
        "Paparan_Asap_Rokok": "Tidak",
        "Desa": "Tidak diketahui",
        "Kecamatan": "Tidak diketahui",
    }
    for column, default_value in categorical_fill.items():
        if column in df.columns:
            df[column] = df[column].astype("string").fillna(default_value).astype(str)
        else:
            df[column] = default_value

    df["Trimester"] = df["Trimester"].astype(str)
    df["Gravida"] = df["Gravida"].astype(str)

    return df


def prepare_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset with engineered features and risk labels."""
    df = engineer_features(dataframe)
    df["Risk_Level"] = df.apply(derive_risk_label, axis=1)
    df = df.dropna(subset=["Risk_Level"])
    df["Risk_Level"] = pd.Categorical(df["Risk_Level"], categories=RISK_LEVELS, ordered=True)
    return df


def build_model_pipeline(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> Pipeline:
    """Construct preprocessing and model pipeline."""
    encoder_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_kwargs["sparse_output"] = False
    else:
        encoder_kwargs["sparse"] = False

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(dict.fromkeys(numeric_features))),
            (
                "cat",
                OneHotEncoder(**encoder_kwargs),
                list(dict.fromkeys(categorical_features)),
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(random_state=42, class_weight="balanced")

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", classifier),
    ])
    return pipeline


def compute_feature_names(preprocess: ColumnTransformer) -> List[str]:
    """Extract feature names after preprocessing."""
    feature_names: List[str] = []
    for name, transformer, columns in preprocess.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(columns))
    return feature_names


def persist_model(artifacts: RiskModelArtifacts) -> None:
    """Persist trained model and metadata to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": artifacts.model,
        "label_encoder": artifacts.label_encoder,
        "feature_names": artifacts.feature_names,
    }
    joblib.dump(payload, MODEL_PATH)
    logger.info("Persisted risk model to %s", MODEL_PATH)


def load_model_from_disk() -> Optional[Dict[str, object]]:
    """Load persisted model when available."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as exc:  # pragma: no cover - guard for corruption
            logger.exception("Failed to load persisted model: %s", exc)
    return None


@st.cache_resource(show_spinner=False)
def train_risk_model(dataframe: pd.DataFrame) -> RiskModelArtifacts:
    """Train Random Forest classifier with hyperparameter tuning."""
    prepared = prepare_dataset(dataframe)
    if prepared.empty:
        raise ValueError("Dataset tidak memiliki data risiko yang valid untuk pelatihan.")

    X = prepared.drop(columns=["Risk_Level"])
    y = prepared["Risk_Level"].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    numeric_features = [feature for feature in NUMERIC_ENRICHED if feature in X.columns]
    categorical_features = [feature for feature in CATEGORICAL_ENRICHED if feature in X.columns]

    pipeline = build_model_pipeline(numeric_features, categorical_features)

    param_grid = {
        "model__n_estimators": [150, 250],
        "model__max_depth": [6, 10, None],
        "model__min_samples_split": [2, 4],
        "model__min_samples_leaf": [1, 2],
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    unique_class_count = max(2, min(5, int(np.unique(y_encoded).size)))
    cv = StratifiedKFold(n_splits=unique_class_count, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_model: Pipeline = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    raw_proba = best_model.predict_proba(X_test)

    model_class_indices = list(best_model.classes_)
    total_classes = len(label_encoder.classes_)
    y_proba = np.zeros((raw_proba.shape[0], total_classes), dtype=float)
    for proba_idx, class_index in enumerate(model_class_indices):
        if 0 <= class_index < total_classes:
            y_proba[:, class_index] = raw_proba[:, proba_idx]

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        label_encoder.inverse_transform(y_test),
        label_encoder.inverse_transform(y_pred),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    conf = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))

    roc_info: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    if total_classes <= 1:
        auc = float("nan")
        for label in label_encoder.classes_:
            roc_info[label] = (np.array([0, 1]), np.array([0, 1]), auc)
    elif total_classes == 2:
        risk_priority = {label: idx for idx, label in enumerate(RISK_LEVELS)}
        present_labels = list(label_encoder.classes_)
        # Treat the highest priority risk level present in the dataset as the positive class
        positive_label = max(present_labels, key=lambda label: risk_priority.get(label, -1))
        negative_label = next(label for label in present_labels if label != positive_label)
        positive_index = label_encoder.transform([positive_label])[0]
        y_true_binary = (y_test == positive_index).astype(int)
        proba_positive = y_proba[:, positive_index]
        try:
            auc = roc_auc_score(y_true_binary, proba_positive)
            fpr, tpr, _ = roc_curve(y_true_binary, proba_positive)
        except ValueError:
            auc = float("nan")
            fpr = np.array([0, 1])
            tpr = np.array([0, 1])
        roc_info[positive_label] = (fpr, tpr, auc)
        roc_info[negative_label] = (
            np.array([0, 1]),
            np.array([0, 1]),
            auc,
        )
    else:
        y_test_binarized = label_binarize(y_test, classes=list(range(total_classes)))
        try:
            auc = roc_auc_score(y_test_binarized, y_proba, average="macro", multi_class="ovr")
        except ValueError:
            auc = float("nan")

        for idx, label in enumerate(label_encoder.classes_):
            try:
                fpr, tpr, _ = roc_curve(y_test_binarized[:, idx], y_proba[:, idx])
                roc_info[label] = (fpr, tpr, auc)
            except ValueError:
                roc_info[label] = (np.array([0, 1]), np.array([0, 1]), auc)

    preprocess = best_model.named_steps["preprocess"]
    feature_names = compute_feature_names(preprocess)
    model_estimator: RandomForestClassifier = best_model.named_steps["model"]
    importances = model_estimator.feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    X_full_transformed = preprocess.transform(X)
    neighbors = NearestNeighbors(n_neighbors=min(5, len(X_full_transformed))).fit(X_full_transformed)

    predictions = prepared.copy()
    predictions["Predicted_Risk"] = label_encoder.inverse_transform(best_model.predict(X))
    prob_raw = best_model.predict_proba(X)
    prob_matrix = np.zeros((prob_raw.shape[0], total_classes), dtype=float)
    for proba_idx, class_index in enumerate(model_class_indices):
        if 0 <= class_index < total_classes:
            prob_matrix[:, class_index] = prob_raw[:, proba_idx]

    for idx, class_label in enumerate(label_encoder.classes_):
        predictions[f"Prob_{class_label}"] = prob_matrix[:, idx]

    metrics = {
        "accuracy": accuracy,
        "auc": auc,
        "best_params": grid.best_params_,
    }

    artifacts = RiskModelArtifacts(
        model=best_model,
        label_encoder=label_encoder,
        feature_names=feature_names,
        classification_report=report_df,
        confusion_matrix=conf,
        roc_data=roc_info,
        metrics=metrics,
        feature_importances=importance_df,
        predictions=predictions,
        neighbors=neighbors,
        transformed_features=X_full_transformed,
        X=X,
        y=y_encoded,
    )

    persist_model(artifacts)
    return artifacts


def build_confusion_matrix_chart(conf_matrix: np.ndarray, labels: Sequence[str]) -> go.Figure:
    """Create confusion matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=conf_matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=conf_matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Prediksi", yaxis_title="Aktual")
    return fig


def build_feature_importance_chart(feature_importances: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart for feature importances."""
    fig = px.bar(
        feature_importances.head(10).sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Feature Importance",
    )
    return fig


def build_roc_curve_chart(roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]]) -> go.Figure:
    """Create ROC curve plot."""
    fig = go.Figure()
    for label, (fpr, tpr, auc) in roc_data.items():
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label} (AUC={auc:.2f})")
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def display_interventions(risk_level: str) -> None:
    """Show intervention recommendations for risk level."""
    items = INTERVENTION_RECOMMENDATIONS.get(risk_level, [])
    color = RISK_COLORS.get(risk_level, "#34495E")
    st.markdown(
        f"<div style='background:{color}15;border-left:4px solid {color};padding:16px;border-radius:8px;'>"
        f"<strong>Rekomendasi untuk Risiko {risk_level}</strong><ul>"
        + "".join(f"<li>{item}</li>" for item in items)
        + "</ul></div>",
        unsafe_allow_html=True,
    )


def predict_single_case(artifacts: RiskModelArtifacts, input_data: Dict[str, object]) -> Tuple[str, float, Dict[str, float]]:
    """Predict risk for a single respondent."""
    df = pd.DataFrame([input_data])
    df = engineer_features(df)
    prediction = artifacts.model.predict(df)[0]
    probabilities = artifacts.model.predict_proba(df)[0]
    label = artifacts.label_encoder.inverse_transform([prediction])[0]
    prob_map = {
        artifacts.label_encoder.inverse_transform([idx])[0]: float(prob)
        for idx, prob in enumerate(probabilities)
    }
    confidence = float(max(prob_map.values()))
    return label, confidence, prob_map


def find_similar_cases(
    artifacts: RiskModelArtifacts,
    input_data: Dict[str, object],
    top_n: int = 5,
) -> pd.DataFrame:
    """Find similar respondents using nearest neighbors."""
    df = pd.DataFrame([input_data])
    df = engineer_features(df)
    transformed = artifacts.model.named_steps["preprocess"].transform(df)
    distances, indices = artifacts.neighbors.kneighbors(transformed, n_neighbors=min(top_n, len(artifacts.predictions)))
    similar = artifacts.predictions.iloc[indices[0]].copy()
    similar["Distance"] = distances[0]
    return similar


def batch_predict(artifacts: RiskModelArtifacts, uploaded_file: BytesIO) -> pd.DataFrame:
    """Perform batch predictions from uploaded Excel file."""
    incoming = pd.read_excel(uploaded_file)
    engineered = engineer_features(incoming)
    preds = artifacts.model.predict(engineered)
    probs = artifacts.model.predict_proba(engineered)
    result = engineered.copy()
    result["Predicted_Risk"] = artifacts.label_encoder.inverse_transform(preds)
    for idx, class_label in enumerate(artifacts.label_encoder.classes_):
        result[f"Prob_{class_label}"] = probs[:, idx]
    return result


def download_predictions(dataframe: pd.DataFrame, filename: str) -> BytesIO:
    """Prepare DataFrame for download."""
    buffer = BytesIO()
    dataframe.to_excel(buffer, index=False)
    buffer.seek(0)
    buffer.name = filename
    return buffer


def render(dataframe: pd.DataFrame) -> None:
    """Render Module 3 Streamlit UI."""
    st.title("Segmentasi Risiko dan Rekomendasi Intervensi")
    st.markdown(
        "Modul ini memanfaatkan pembelajaran mesin untuk mengelompokkan ibu hamil berdasarkan risiko anemia dan memberikan rekomendasi intervensi."
    )
    st.markdown(apply_styling(), unsafe_allow_html=True)

    with st.spinner("Melatih model risiko..."):
        try:
            artifacts = train_risk_model(dataframe)
        except ValueError as exc:
            st.error(str(exc))
            logger.exception("Risk model training failed: %s", exc)
            return

    st.caption("Model: Random Forest Classifier dengan cross-validation dan tuning hyperparameter.")

    tabs = st.tabs([
        "Model Overview",
        "Risk Distribution",
        "Predict New Case",
        "Batch Prediction",
        "High-Risk List",
    ])

    with tabs[0]:
        metrics_cols = st.columns(3)
        metrics_cols[0].markdown(
            create_metric_card("Akurasi", f"{artifacts.metrics['accuracy']*100:.1f}%", icon="✅"),
            unsafe_allow_html=True,
        )
        metrics_cols[1].markdown(
            create_metric_card("AUC", f"{artifacts.metrics['auc']:.2f}", icon="📈"),
            unsafe_allow_html=True,
        )
        metrics_cols[2].markdown(
            create_metric_card(
                "Model Params",
                json.dumps(artifacts.metrics["best_params"], ensure_ascii=False),
                icon="⚙️",
            ),
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            build_confusion_matrix_chart(
                artifacts.confusion_matrix,
                artifacts.label_encoder.classes_,
            ),
            use_container_width=True,
        )

        st.plotly_chart(build_feature_importance_chart(artifacts.feature_importances), use_container_width=True)
        st.plotly_chart(build_roc_curve_chart(artifacts.roc_data), use_container_width=True)

        st.subheader("Classification Report")
        st.dataframe(artifacts.classification_report, use_container_width=True)

    with tabs[1]:
        predictions = artifacts.predictions
        risk_counts = predictions["Predicted_Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Jumlah"]
        st.plotly_chart(
            px.pie(
                risk_counts,
                names="Risk",
                values="Jumlah",
                color="Risk",
                color_discrete_map=RISK_COLORS,
                title="Proporsi Risiko",
            ),
            use_container_width=True,
        )

        predictions["Kelompok_Usia"] = predictions.get("Kelompok_Usia", assign_age_group(predictions["Usia"])).astype(str)
        age_group = predictions.groupby(["Kelompok_Usia", "Predicted_Risk"]).size().reset_index(name="Jumlah")
        st.plotly_chart(
            px.bar(
                age_group,
                x="Kelompok_Usia",
                y="Jumlah",
                color="Predicted_Risk",
                barmode="stack",
                color_discrete_map=RISK_COLORS,
                title="Risiko per Kelompok Usia",
            ),
            use_container_width=True,
        )

        trimester_group = predictions.groupby(["Trimester", "Predicted_Risk"]).size().reset_index(name="Jumlah")
        st.plotly_chart(
            px.bar(
                trimester_group,
                x="Trimester",
                y="Jumlah",
                color="Predicted_Risk",
                barmode="group",
                color_discrete_map=RISK_COLORS,
                title="Risiko per Trimester",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            px.density_heatmap(
                predictions,
                x="Kecamatan",
                y="Predicted_Risk",
                histfunc="count",
                color_continuous_scale="YlGnBu",
                title="Sebaran Risiko per Kecamatan",
            ),
            use_container_width=True,
        )

    with tabs[2]:
        st.subheader("Prediksi Kasus Baru")
        with st.form("new_case_form"):
            col1, col2, col3 = st.columns(3)
            usia = col1.number_input("Usia", min_value=15, max_value=50, value=25)
            pendidikan = col1.selectbox(
                "Pendidikan",
                options=sorted(dataframe["Pendidikan"].dropna().unique().tolist()) if "Pendidikan" in dataframe else ["SMA"],
            )
            pendapatan = col1.number_input("Pendapatan", min_value=0, value=1500000, step=100000)

            hb = col2.number_input("Hb (g/dL)", min_value=5.0, max_value=16.0, value=11.0, step=0.1)
            lila = col2.number_input("LILA (cm)", min_value=18.0, max_value=40.0, value=24.0, step=0.1)
            jarak = col2.number_input("Jarak ke Faskes (km)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)

            trimester = col3.selectbox("Trimester", options=["1", "2", "3"], index=1)
            gravida = col3.selectbox(
                "Gravida",
                options=sorted(dataframe["Gravida"].dropna().astype(str).unique().tolist()) if "Gravida" in dataframe else ["1"],
            )
            ttd = col3.selectbox("Terima TTD", options=["Ya", "Tidak"], index=0)

            kepatuhan = st.slider("Kepatuhan TTD (%)", min_value=0, max_value=100, value=80, step=5)
            frekuensi_protein = st.slider("Frekuensi Protein Hewani / minggu", min_value=0, max_value=21, value=7)
            konsumsi_inhibitor = st.selectbox("Konsumsi Inhibitor", options=["Sering", "Kadang", "Jarang", "Tidak"])
            akses_air = st.selectbox("Akses Air Bersih", options=["Baik", "Cukup", "Kurang"])
            cuci_tangan = st.selectbox("Perilaku Cuci Tangan", options=["Baik", "Cukup", "Kurang"])
            paparan_asap = st.selectbox("Paparan Asap Rokok", options=["Ya", "Tidak"], index=1)
            desa = st.text_input("Desa", value="")
            kecamatan = st.selectbox(
                "Kecamatan",
                options=sorted(dataframe["Kecamatan"].dropna().unique().tolist()) if "Kecamatan" in dataframe else [""],
            )

            submitted = st.form_submit_button("Prediksi Risiko")

        if submitted:
            input_payload = {
                "Usia": usia,
                "Pendidikan": pendidikan,
                "Pendapatan": pendapatan,
                "Jarak_Faskes": jarak,
                "Hb": hb,
                "LILA": lila,
                "Status_KEK": derive_kek_status(pd.Series([lila])).iloc[0] if pd.notna(lila) else "Tidak Berisiko",
                "Trimester": trimester,
                "Gravida": gravida,
                "Terima_TTD": ttd,
                "Kepatuhan_TTD": kepatuhan,
                "Kendala_TTD": "",
                "Frekuensi_Protein_Hewani": frekuensi_protein,
                "Konsumsi_Inhibitor": konsumsi_inhibitor,
                "Akses_Air": akses_air,
                "Cuci_Tangan": cuci_tangan,
                "Paparan_Asap_Rokok": paparan_asap,
                "Desa": desa,
                "Kecamatan": kecamatan,
            }
            label, confidence, prob_map = predict_single_case(artifacts, input_payload)
            st.success(f"Risiko terprediksi: {label} (keyakinan {confidence*100:.1f}%)")
            display_interventions(label)

            st.markdown("#### Probabilitas Kategori")
            st.json(prob_map)

            st.markdown("#### What-if Analysis")
            with st.expander("Simulasikan perubahan indikator"):
                sim_hb = st.slider("Hb Simulasi", min_value=5.0, max_value=16.0, value=hb, step=0.1)
                sim_kepatuhan = st.slider("Kepatuhan Simulasi", 0, 100, kepatuhan, step=5)
                sim_lila = st.slider("LILA Simulasi", 18.0, 40.0, lila, step=0.1)
                sim_trimester = st.selectbox("Trimester Simulasi", options=["1", "2", "3"], index=["1", "2", "3"].index(trimester))
                scenario_payload = input_payload | {"Hb": sim_hb, "Kepatuhan_TTD": sim_kepatuhan, "LILA": sim_lila, "Trimester": sim_trimester}
                scenario_label, scenario_conf, _ = predict_single_case(artifacts, scenario_payload)
                st.info(
                    f"Hasil simulasi: {scenario_label} dengan keyakinan {scenario_conf*100:.1f}% setelah penyesuaian indikator."
                )

            st.markdown("#### Kasus Serupa")
            similar_cases = find_similar_cases(artifacts, input_payload)
            st.dataframe(
                similar_cases[
                    [
                        "Nama",
                        "Kecamatan",
                        "Trimester",
                        "Predicted_Risk",
                        "Prob_High" if "Prob_High" in similar_cases else "Predicted_Risk",
                        "Distance",
                    ]
                ],
                use_container_width=True,
            )

            st.info("Fitur pelacakan intervensi akan menandai tindak lanjut terhadap kasus berisiko (roadmap).")

    with tabs[3]:
        st.subheader("Batch Prediction")
        uploaded = st.file_uploader("Unggah file Excel", type=["xlsx", "xls"])
        if uploaded is not None:
            try:
                result_df = batch_predict(artifacts, uploaded)
            except Exception as exc:
                st.error(f"Gagal memproses file: {exc}")
            else:
                st.success("Batch prediction berhasil diproses.")
                st.dataframe(result_df.head(20), use_container_width=True)
                download_buffer = download_predictions(result_df, "prediksi_batch_risiko.xlsx")
                st.download_button(
                    "Unduh hasil prediksi",
                    data=download_buffer,
                    file_name=download_buffer.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with tabs[4]:
        st.subheader("Daftar Risiko Tinggi")
        high_risk = artifacts.predictions[artifacts.predictions["Predicted_Risk"] == "High"]
        if high_risk.empty:
            st.info("Tidak ada kasus dengan risiko tinggi pada dataset saat ini.")
        else:
            st.dataframe(
                high_risk[
                    [
                        "Nama",
                        "Kecamatan",
                        "Desa",
                        "Trimester",
                        "Hb",
                        "LILA",
                        "Kepatuhan_TTD",
                        "Predicted_Risk",
                        "Prob_High" if "Prob_High" in high_risk else "Predicted_Risk",
                    ]
                ],
                use_container_width=True,
            )
            download_buffer = download_predictions(high_risk, "daftar_risiko_tinggi.xlsx")
            st.download_button(
                "Unduh daftar risiko tinggi",
                data=download_buffer,
                file_name=download_buffer.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.markdown("---")
    st.caption(
        "Model risiko ini mendukung pengambilan keputusan berbasis data untuk intervensi anemia ibu hamil."
    )

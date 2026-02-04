from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.set_page_config(page_title="Churn Risk Console", page_icon="C", layout="wide")
    st.error("TensorFlow is not installed in this environment.")
    st.info(
        "Install dependencies from requirements.txt and deploy with Python 3.11 or 3.12, "
        "then restart the app."
    )
    st.stop()


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cleaned_telco_churn.csv"
MODEL_PATH = BASE_DIR / "best_nn_model.keras"
NUMERIC_COLUMNS = ["tenure", "monthlycharges", "totalcharges"]
RAW_FEATURE_COLUMNS = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
]


st.set_page_config(page_title="Churn Risk Console", page_icon="C", layout="wide")


@st.cache_data(show_spinner=False)
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [col.lower() for col in df.columns]
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df = df.dropna().copy()

    X = df.drop(columns=["churn"], errors="ignore")
    X_encoded = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_encoded[NUMERIC_COLUMNS] = scaler.fit_transform(X_encoded[NUMERIC_COLUMNS])
    return df, X_encoded.columns.tolist(), scaler


@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_input(input_df, encoded_columns, scaler):
    x = pd.get_dummies(input_df, drop_first=True)
    x = x.reindex(columns=encoded_columns, fill_value=0)
    x[NUMERIC_COLUMNS] = scaler.transform(x[NUMERIC_COLUMNS])
    return x


def render_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
        :root {
            --ink: #142433;
            --ink-soft: #334f67;
            --panel: #ffffff;
            --panel-border: #ccd9e5;
            --accent: #0b5ea8;
            --accent-strong: #08477f;
            --high-bg: #ffd9cf;
            --high-text: #5a1200;
            --low-bg: #daf6df;
            --low-text: #0f4d25;
        }
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: "Manrope", "Trebuchet MS", "Verdana", sans-serif;
            color: var(--ink);
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 12%, #fff1db 0%, rgba(255, 241, 219, 0.45) 35%, transparent 55%),
                radial-gradient(circle at 92% 15%, #e7f7ef 0%, rgba(231, 247, 239, 0.45) 35%, transparent 58%),
                linear-gradient(155deg, #eef3fb 0%, #e4edf8 44%, #dbe8f5 100%);
        }
        section.main > div {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 1.25rem 1.2rem 1.5rem 1.2rem;
            box-shadow: 0 6px 18px rgba(39, 68, 94, 0.08);
        }
        .hero {
            border: 1px solid #aec3d8;
            background: linear-gradient(130deg, rgba(255, 255, 255, 0.96), rgba(238, 245, 255, 0.96));
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 1.1rem;
        }
        .hero h2 {
            color: #0d2f4d;
            letter-spacing: 0.01em;
        }
        .hero p {
            color: var(--ink-soft);
        }
        h1, h2, h3, h4, p, label, .stMarkdown, .stText {
            color: var(--ink) !important;
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 0.45rem 0.7rem;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--ink-soft) !important;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #0f3961 !important;
            font-weight: 800;
        }
        [data-baseweb="select"] > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background: #ffffff;
            border: 1px solid #9cb1c6 !important;
            color: var(--ink) !important;
        }
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--accent);
        }
        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            background: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid var(--accent-strong) !important;
            font-weight: 700;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: var(--accent-strong) !important;
        }
        .risk {
            border-radius: 14px;
            padding: 0.9rem 1rem;
            color: var(--ink);
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
            border: 1px solid rgba(0, 0, 0, 0.12);
        }
        .risk.high {
            background: var(--high-bg);
            color: var(--high-text);
        }
        .risk.low {
            background: var(--low-bg);
            color: var(--low-text);
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed #97aec5;
            background: rgba(255, 255, 255, 0.9);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_input_form(df):
    st.subheader("Single Customer Prediction")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])

        with right:
            device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
            )
            monthly_min = float(df["monthlycharges"].min())
            monthly_max = float(df["monthlycharges"].max())
            total_min = float(df["totalcharges"].min())
            total_max = float(df["totalcharges"].max())
            monthly_charges = st.slider("Monthly Charges", monthly_min, monthly_max, 70.0, step=0.1)
            default_total = float(df["totalcharges"].median())
            total_charges = st.number_input(
                "Total Charges",
                min_value=total_min,
                max_value=total_max,
                value=default_total,
                step=1.0,
            )

        submitted = st.form_submit_button("Predict Churn")

    if not submitted:
        return None

    row = {
        "gender": gender,
        "seniorcitizen": senior,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phone_service,
        "multiplelines": multiple_lines,
        "internetservice": internet_service,
        "onlinesecurity": online_security,
        "onlinebackup": online_backup,
        "deviceprotection": device_protection,
        "techsupport": tech_support,
        "streamingtv": streaming_tv,
        "streamingmovies": streaming_movies,
        "contract": contract,
        "paperlessbilling": paperless_billing,
        "paymentmethod": payment_method,
        "monthlycharges": monthly_charges,
        "totalcharges": total_charges,
    }
    return pd.DataFrame([row])


def render_batch_section(model, encoded_columns, scaler):
    st.subheader("Batch Prediction (CSV)")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the same input columns (without churn).",
        type=["csv"],
    )
    if uploaded_file is None:
        return

    batch_df = pd.read_csv(uploaded_file)
    batch_df.columns = [col.lower() for col in batch_df.columns]

    missing = [col for col in RAW_FEATURE_COLUMNS if col not in batch_df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    batch_input = batch_df[RAW_FEATURE_COLUMNS].copy()
    batch_input["totalcharges"] = pd.to_numeric(batch_input["totalcharges"], errors="coerce")
    batch_input["monthlycharges"] = pd.to_numeric(batch_input["monthlycharges"], errors="coerce")
    batch_input["tenure"] = pd.to_numeric(batch_input["tenure"], errors="coerce")

    if batch_input[NUMERIC_COLUMNS].isna().any().any():
        st.error("CSV has invalid numeric values in tenure/monthlycharges/totalcharges.")
        return

    x_batch = preprocess_input(batch_input, encoded_columns, scaler)
    probabilities = model.predict(x_batch, verbose=0).ravel()

    results = batch_df.copy()
    results["churn_probability"] = probabilities.round(4)
    results["predicted_label"] = ["Yes" if p >= 0.5 else "No" for p in probabilities]

    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download Predictions",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


def main():
    render_style()
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">Telco Churn Predictor</h2>
            <p style="margin:0.4rem 0 0 0;">Neural network model with cross-validated tuning, ready for interactive inference.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error(f"Reference dataset not found: {DATA_PATH}")
        st.stop()
    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    df, encoded_columns, scaler = load_reference_data()
    model = load_model()

    m1, m2, m3 = st.columns(3)
    m1.metric("NN CV Accuracy", "75.84%")
    m2.metric("NN Test Accuracy", "75.55%")
    m3.metric("DT Test Accuracy", "73.28%")

    input_df = build_input_form(df)
    if input_df is not None:
        x_ready = preprocess_input(input_df, encoded_columns, scaler)
        probability = float(model.predict(x_ready, verbose=0).ravel()[0])
        label = "Likely to Churn" if probability >= 0.5 else "Likely to Stay"
        css_class = "high" if probability >= 0.5 else "low"

        st.markdown(f'<div class="risk {css_class}">{label}</div>', unsafe_allow_html=True)
        st.write(f"Churn Probability: `{probability:.2%}`")
        st.progress(min(max(probability, 0.0), 1.0))

    st.divider()
    render_batch_section(model, encoded_columns, scaler)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path

# ============================
# Page / Theming
# ============================
st.set_page_config(
    page_title="ML Model Demo",
    page_icon="🤖",
    layout="wide",
)

# Minimal custom styling (no need for external CSS files)
st.markdown(
    """
    <style>
      :root {
        --accent: #5b9bd5; /* Soft blue */
        --bg: #0e1117;
        --card: #111827;
        --muted: #6b7280;
      }
      .accent { color: var(--accent) !important; }
      .card {
        background: var(--card);
        padding: 1rem 1.25rem; border-radius: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
      }
      .small { font-size: 0.9rem; color: var(--muted); }
      .metric-pill {
        display:inline-block; padding: .35rem .65rem; border-radius: 999px;
        background: rgba(91,155,213,.1); color: var(--accent); font-weight: 600;
        border: 1px solid rgba(91,155,213,.25);
      }
      .stButton>button {
        border-radius: 12px; font-weight: 600; border: 1px solid rgba(255,255,255,0.15);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# Helpers
# ============================
@st.cache_resource(show_spinner=False)
def load_model_from_path(path: Path):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def read_results_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Expected: model, metric (or accuracy/f1/score)
    if "metric" not in df.columns:
        # Try common names; prefer accuracy then f1
        for alt in ["accuracy", "f1", "score", "auc"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "metric"})
                break
    if "model" not in df.columns:
        # Try name or classifier
        for alt in ["name", "classifier", "estimator"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "model"})
                break
    return df

# Determine problem type

def is_classifier(model) -> bool:
    try:
        from sklearn.base import is_classifier as sk_is_classifier
        return sk_is_classifier(model)
    except Exception:
        # Heuristic: predict_proba exists => classifier
        return hasattr(model, "predict_proba")

# ============================
# Sidebar – Model loading & options
# ============================
st.sidebar.header("⚙️ Settings")

# 1) Load model: either from repo (best_model.pkl) or by upload
model = None
model_status = ""

default_model_path = Path("best_model.pkl")
if default_model_path.exists():
    try:
        model = load_model_from_path(default_model_path)
        model_status = f"Model loaded from **{default_model_path.name}**"
    except Exception as e:
        model_status = f"Failed to load default model: {e}"

uploaded_model = st.sidebar.file_uploader("Upload model file (.pkl)", type=["pkl", "joblib"], help="Preferably a Pipeline or Estimator saved with joblib")
if uploaded_model is not None:
    try:
        model = joblib.load(uploaded_model)
        model_status = f"Model loaded from uploaded file: **{uploaded_model.name}**"
    except Exception as e:
        model = None
        model_status = f"❌ Failed to load model: {e}"

st.sidebar.markdown(f"**Model status:** {model_status}")

# 2) Optional: upload a CSV with model comparison results (model, metric)
results_file = st.sidebar.file_uploader(
    "Model comparison results (CSV)", type=["csv"],
    help="Add a file with a 'model' column and a 'metric' column (or accuracy/f1/score)"
)

# 3) Configure feature inputs
st.sidebar.divider()
st.sidebar.subheader("🧮 Input settings")

# Try to infer number of features
inferred_n_features = None
if model is not None:
    for attr in ["n_features_in_", "n_features_"]:
        if hasattr(model, attr):
            inferred_n_features = int(getattr(model, attr))
            break

n_features = st.sidebar.number_input(
    "Number of features", min_value=1, max_value=200,
    value=int(inferred_n_features or 4), step=1,
    help="The system can try to infer the number, or you can set it manually"
)

# Optional: sample data to infer features
st.sidebar.markdown("**Infer feature inputs from data**")
sample_data_file = st.sidebar.file_uploader(
    "Sample data (CSV)", type=["csv"], key="sample_csv",
    help="Upload a small CSV with your input columns to auto-generate inputs"
)
use_bundled = False
if sample_data_file is None and Path("Healthcare-Diabetes.csv").exists():
    use_bundled = st.sidebar.checkbox("Use bundled Healthcare-Diabetes.csv", value=False)

sample_df = None
if sample_data_file is not None:
    try:
        sample_df = pd.read_csv(sample_data_file)
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
elif use_bundled:
    try:
        sample_df = pd.read_csv("Healthcare-Diabetes.csv")
    except Exception as e:
        st.sidebar.error(f"Failed to read bundled CSV: {e}")

target_column = None
feature_columns = None
if sample_df is not None:
    all_cols = list(sample_df.columns)
    guess_targets = [c for c in all_cols if c.lower() in ["target", "label", "outcome", "y"]]
    target_opts = ["<none>"] + all_cols
    target_idx = 0
    if guess_targets:
        try:
            target_idx = 1 + all_cols.index(guess_targets[0])
        except Exception:
            target_idx = 0
    target_column = st.sidebar.selectbox("Target column (exclude)", options=target_opts, index=target_idx)

    excluded_cols = set()
    if target_column and target_column != "<none>":
        excluded_cols.add(target_column)

    id_like = [c for c in all_cols if c.lower() == "id" or c.lower().endswith("id")]
    default_id_like = [c for c in id_like[:1]]
    if id_like:
        excluded_ids = st.sidebar.multiselect("Exclude ID-like columns", id_like, default=default_id_like)
        excluded_cols.update(excluded_ids)

    feature_columns = [c for c in all_cols if c not in excluded_cols]

show_prob = st.sidebar.checkbox("Show prediction probabilities (for Classification)", value=True)

# ============================
# Header
# ============================
st.markdown("""
# 🤖 <span class="accent">ML Model Deployment</span>
A simple **Streamlit** interface to compare models, select the best model, and try predictions with your own inputs.
""", unsafe_allow_html=True)

# Tabs: Compare | Predict
tab_compare, tab_predict = st.tabs(["📊 Model Comparison", "🔮 Prediction"])

# ============================
# Tab 1: Model Comparison
# ============================
with tab_compare:
    st.markdown("### 📊 Model Performance Comparison")
    help_txt = (
        "You can upload a CSV file from the sidebar containing a 'model' column and a 'metric' column\n"
        "Example: LogisticRegression,0.89\nRandomForest,0.93\nXGBoost,0.95"
    )
    st.markdown(f"<p class='small'>{help_txt}</p>", unsafe_allow_html=True)

    df_results = None
    if results_file is not None:
        try:
            df_results = read_results_csv(results_file.getvalue())
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    # Optional manual input as textarea (CSV-like)
    with st.expander("Or enter results manually (text)"):
        sample_text = """model,metric
Logistic Regression,0.87
Random Forest,0.92
XGBoost,0.95
SVM,0.89
KNN,0.85
"""
        txt = st.text_area("Paste results here (CSV)", sample_text, height=160)
        if st.button("Use this text"):
            try:
                df_results = pd.read_csv(io.StringIO(txt))
            except Exception as e:
                st.error(f"Failed to read text as CSV: {e}")

    if df_results is not None and {"model", "metric"}.issubset({c.lower() for c in df_results.columns} | set(df_results.columns)):
        # Normalize column names again (in case of manual entry)
        cols_lower = [c.lower() for c in df_results.columns]
        df_results.columns = cols_lower
        if "model" not in df_results.columns or "metric" not in df_results.columns:
            st.error("The file must contain 'model' and 'metric' columns")
        else:
            # Clean & sort
            df_plot = df_results.copy()
            df_plot = df_plot.dropna(subset=["model", "metric"]).copy()
            df_plot["metric"] = pd.to_numeric(df_plot["metric"], errors="coerce")
            df_plot = df_plot.dropna(subset=["metric"]).copy()
            df_plot = df_plot.sort_values("metric", ascending=False)

            st.dataframe(df_plot.reset_index(drop=True), use_container_width=True)

            # Plot (Streamlit built-in bar chart)
            chart_df = df_plot.set_index("model")["metric"]
            st.bar_chart(chart_df)

            if not df_plot.empty:
                best_row = df_plot.iloc[0]
                st.markdown(
                    f"**Best model:** <span class='metric-pill'>{best_row['model']}</span> — value: **{best_row['metric']:.4f}**",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Upload the results file or enter them manually to display the comparison.")

# ============================
# Tab 2: Prediction
# ============================
with tab_predict:
    st.markdown("### 🔮 Predict using the model")

    if model is None:
        st.warning("No model loaded yet. Upload a .pkl file from the sidebar or place best_model.pkl in the folder.")
    else:
        problem_type = "Classification" if is_classifier(model) else "Regression"
        st.markdown(f"**Detected problem type:** {problem_type}")

        # Input form
        with st.form("predict_form"):
            cols = st.columns(4)
            inputs = []
            inputs_map = None
            ui_mode = "count"  # one of: sample_df, feature_names, count

            # Prefer: infer from sample data
            if 'feature_columns' in globals() and feature_columns is not None and 'sample_df' in globals() and sample_df is not None:
                ui_mode = "sample_df"
                for idx, col_name in enumerate(feature_columns):
                    series = sample_df[col_name]
                    with cols[idx % 4]:
                        # Numeric columns
                        if pd.api.types.is_bool_dtype(series):
                            default_val = bool(series.mode(dropna=True).iloc[0]) if not series.dropna().empty else False
                            val = st.checkbox(f"{col_name}", value=default_val, key=f"col_{col_name}")
                        elif pd.api.types.is_numeric_dtype(series):
                            # Use median as default to be robust to outliers
                            median_val = series.median()
                            try:
                                default_val = float(median_val)
                            except Exception:
                                default_val = 0.0
                            val = st.number_input(f"{col_name}", value=default_val, key=f"col_{col_name}")
                        else:
                            uniques = series.dropna().unique()
                            # For manageable cardinality, use a selectbox; otherwise, text input
                            if 0 < len(uniques) <= 50:
                                options = [str(u) for u in sorted(uniques, key=lambda x: str(x))]
                                val = st.selectbox(f"{col_name}", options=options, key=f"col_{col_name}")
                            else:
                                default_text = str(series.dropna().iloc[0]) if not series.dropna().empty else ""
                                val = st.text_input(f"{col_name}", value=default_text, key=f"col_{col_name}")
                        inputs.append((col_name, val))
                submitted = st.form_submit_button("Predict")
            else:
                # Next preference: model-declared feature names
                feature_names = None
                if hasattr(model, "feature_names_in_"):
                    try:
                        feature_names = list(getattr(model, "feature_names_in_"))
                    except Exception:
                        feature_names = None

                if feature_names:
                    ui_mode = "feature_names"
                    inputs_map = {}
                    for i, name in enumerate(feature_names):
                        with cols[i % 4]:
                            val = st.number_input(f"{name}", value=0.0, key=f"f_{i}")
                            inputs_map[name] = val
                    submitted = st.form_submit_button("Predict")
                else:
                    # Fallback: simple count-based inputs
                    ui_mode = "count"
                    for i in range(int(n_features)):
                        with cols[i % 4]:
                            val = st.number_input(f"Feature {i+1}", value=0.0, key=f"f_{i}")
                            inputs.append(val)
                    submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                # Build input data according to UI mode
                if ui_mode == "sample_df":
                    row_dict = {name: value for name, value in inputs}
                    X = pd.DataFrame([row_dict])
                elif ui_mode == "feature_names" and inputs_map is not None:
                    X = pd.DataFrame([inputs_map])
                else:
                    X = np.array(inputs, dtype=float).reshape(1, -1)

                y_pred = model.predict(X)
                st.success(f"Output: {y_pred[0]}")

                if show_prob and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X)[0]
                        prob_df = pd.DataFrame({"Class": list(range(len(proba))), "Probability": proba})
                        st.markdown("**Class probabilities:**")
                        st.dataframe(prob_df, use_container_width=True)
                        st.bar_chart(prob_df.set_index("Class"))
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# ============================
# Footer / Tips
# ============================
st.divider()
st.markdown("Finished")

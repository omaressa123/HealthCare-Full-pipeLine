import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path
import sys
import importlib

# ============================
# Page / Theming
# ============================
st.set_page_config(
    page_title="ML Model Demo",
    page_icon="🤖",
    layout="wide",
)

# Minimal custom styling
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
def load_model_from_bytes(file_bytes: bytes):
    # Try to load the model, and if it fails due to missing module, prompt user to install it
    try:
        model = joblib.load(io.BytesIO(file_bytes))
        # If the loaded object is a dict, try to extract a model from common keys
        if isinstance(model, dict):
            # Try common keys for model storage
            possible_keys = ["model", "estimator", "clf", "regressor", "pipeline"]
            found = False
            for key in possible_keys:
                if key in model and hasattr(model[key], "predict"):
                    model = model[key]
                    found = True
                    break
            # If not found, try to find the first value in the dict that has a predict method
            if not found:
                for k, v in model.items():
                    if hasattr(v, "predict"):
                        model = v
                        found = True
                        break
            if not found:
                # Instead of erroring, try to show the user the available keys and types
                keys_types = {k: type(v).__name__ for k, v in model.items()}
                st.error(
                    f"Failed to load model: The uploaded file contains a dictionary, not a model object.<br>"
                    f"Available keys in the dictionary: <code>{list(model.keys())}</code><br>"
                    f"Types of values: <code>{keys_types}</code><br>"
                    "Please upload a file containing a trained scikit-learn or XGBoost model, "
                    "or a dictionary with a key like 'model' or 'estimator' containing the model object with a <code>predict</code> method."
                )
                return None
        if not hasattr(model, "predict"):
            st.error(
                f"Failed to load model: The loaded object is of type <code>{type(model).__name__}</code> and does not have a <code>predict</code> method.<br>"
                "Please upload a file containing a trained scikit-learn or XGBoost model."
            )
            return None
        return model
    except ModuleNotFoundError as e:
        missing_module = str(e).split("'")[1]
        st.error(
            f"Failed to load model: Missing required Python module: {missing_module}.\n"
            f"Please install it in your environment (e.g., pip install {missing_module}) and reload the app."
        )
        return None
    except AttributeError as e:
        st.error(
            f"Failed to load model: {e}.\n"
            "This may be due to version mismatch between the environment where the model was saved and the current environment.\n"
            "Try to use the same library versions as used during model training."
        )
        return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def is_classifier(model) -> bool:
    # For XGBoost and sklearn models
    try:
        from sklearn.base import is_classifier as sk_is_classifier
        return sk_is_classifier(model)
    except Exception:
        return hasattr(model, "predict_proba")

# ============================
# Sidebar – Model upload & options
# ============================
st.sidebar.header("⚙️ Settings")

# Upload model file (for your use: joblib.dump(models, 'best2_model.pkl'))
model_file = st.sidebar.file_uploader(
    "Upload your model (.pkl)", type=["pkl", "joblib"], key="model_uploader",
    help="Upload a scikit-learn or XGBoost model file (joblib or pickle format)"
)

model = None
model_status = ""
inferred_n_features = None
model_load_error = None

if model_file is not None:
    try:
        model = load_model_from_bytes(model_file.getvalue())
        if model is not None:
            model_status = f"✅ Model loaded from **{model_file.name}**<br>Type: <code>{type(model).__name__}</code>"
            for attr in ["n_features_in_", "n_features_"]:
                if hasattr(model, attr):
                    inferred_n_features = int(getattr(model, attr))
                    break
        else:
            model_status = f"❌ Failed to load model from {model_file.name}. See error message above."
    except Exception as e:
        model_status = f"❌ Failed to load model from {model_file.name}: {e}"
else:
    model_status = "No model uploaded. Please upload a .pkl or .joblib file."

st.sidebar.markdown(f"**Model status:** {model_status}", unsafe_allow_html=True)

# Configure feature inputs
st.sidebar.divider()
st.sidebar.subheader("🧮 Input settings")

n_features = st.sidebar.number_input(
    "Number of features", min_value=1, max_value=200,
    value=int(inferred_n_features or 4), step=1, key="n_features_input",
    help="The system can try to infer the number, or you can set it manually"
)

# Sample data to infer features (optional, keep for UI completeness)
st.sidebar.markdown("**Infer feature inputs from data**")
sample_data_file = st.sidebar.file_uploader(
    "Sample data (CSV)", type=["csv"], key="sample_data_uploader",
    help="Upload a small CSV with your input columns to auto-generate inputs"
)
use_bundled = False
if sample_data_file is None and Path("Healthcare-Diabetes.csv").exists():
    use_bundled = st.sidebar.checkbox("Use bundled Healthcare-Diabetes.csv", value=False, key="use_bundled_checkbox")

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
    target_column = st.sidebar.selectbox(
        "Target column (exclude)", options=target_opts, index=target_idx, key="target_column_select"
    )

    excluded_cols = set()
    if target_column and target_column != "<none>":
        excluded_cols.add(target_column)

    id_like = [c for c in all_cols if c.lower() == "id" or c.lower().endswith("id")]
    default_id_like = [c for c in id_like[:1]]
    if id_like:
        excluded_ids = st.sidebar.multiselect(
            "Exclude ID-like columns", id_like, default=default_id_like, key="exclude_ids_multiselect"
        )
        excluded_cols.update(excluded_ids)

    feature_columns = [c for c in all_cols if c not in excluded_cols]

show_prob = st.sidebar.checkbox(
    "Show prediction probabilities (for Classification)", value=True, key="show_prob_checkbox"
)

# ============================
# Header
# ============================
st.markdown("""
# 🤖 <span class="accent">ML Model Deployment</span>
A simple **Streamlit** interface to upload your model and try predictions with your own inputs.
""", unsafe_allow_html=True)

# ============================
# Prediction Section
# ============================
st.markdown("### 🔮 Predict using the model")

if model is None:
    st.warning("No model loaded yet. Please upload a .pkl or .joblib model file in the sidebar.")
else:
    if not hasattr(model, "predict"):
        st.error(f"Loaded model is of type {type(model)} and does not have a 'predict' method. Ensure the .pkl file contains a valid scikit-learn or XGBoost model.")
    else:
        problem_type = "Classification" if is_classifier(model) else "Regression"
        st.markdown(f"**Detected problem type:** {problem_type}")

        with st.form("predict_form"):
            inputs = []
            ui_mode = "count"

            if sample_df is not None and feature_columns:
                ui_mode = "sample_df"
                st.markdown("**Input features**")
                cols = st.columns(4)
                for i, col_name in enumerate(feature_columns[:n_features]):
                    with cols[i % 4]:
                        if pd.api.types.is_numeric_dtype(sample_df[col_name]):
                            value = st.number_input(
                                col_name, value=0.0, step=0.1, format="%.4f", key=f"input_{col_name}"
                            )
                        else:
                            value = st.text_input(col_name, value="", key=f"input_{col_name}")
                        inputs.append((col_name, value))
            else:
                ui_mode = "count"
                st.markdown("**Input features (generic)**")
                cols = st.columns(4)
                for i in range(n_features):
                    with cols[i % 4]:
                        value = st.number_input(
                            f"Feature {i+1}", value=0.0, step=0.1, format="%.4f", key=f"feature_{i}"
                        )
                        inputs.append(value)

            submitted = st.form_submit_button("Predict")

            if submitted:
                try:
                    if ui_mode == "sample_df":
                        row_dict = {name: value for name, value in inputs}
                        X = pd.DataFrame([row_dict])

                        # --- Fix for feature_names mismatch ---
                        # If model expects more columns than provided, add missing columns with default value (e.g., 0 or np.nan)
                        if hasattr(model, "feature_names_in_"):
                            expected_features = list(model.feature_names_in_)
                        elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
                            # XGBoost sklearn API
                            expected_features = list(model.get_booster().feature_names)
                        else:
                            expected_features = list(X.columns)

                        missing_features = [f for f in expected_features if f not in X.columns]
                        if missing_features:
                            for f in missing_features:
                                X[f] = 0  # or np.nan, or a default value
                        # Reorder columns to match model expectation
                        X = X[expected_features]
                    else:
                        X = np.array(inputs, dtype=float).reshape(1, -1)

                    y_pred = model.predict(X)
                    st.success(f"Output: {y_pred[0]}")

                    if show_prob and hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(X)[0]
                            # For XGBoost, class labels may be in model.classes_
                            class_labels = getattr(model, "classes_", list(range(len(proba))))
                            prob_df = pd.DataFrame({"Class": class_labels, "Probability": proba})
                            st.markdown("**Class probabilities:**")
                            st.dataframe(prob_df, use_container_width=True)
                            st.bar_chart(prob_df.set_index("Class"))
                        except Exception:
                            pass
                except Exception as e:
                    # Try to give a more helpful error if it's a feature_names mismatch
                    import re
                    msg = str(e)
                    if "feature_names mismatch" in msg:
                        # Try to extract expected and provided features from the error message
                        match = re.search(r"feature_names mismatch: \[(.*?)\] \[(.*?)\] expected", msg)
                        if match:
                            expected = match.group(1)
                            provided = match.group(2)
                            st.error(
                                f"Feature names mismatch between model and input.<br>"
                                f"**Model expects:** <code>[{expected}]</code><br>"
                                f"**You provided:** <code>[{provided}]</code><br>"
                                f"Please ensure your input columns match the model's expected features, including any ID columns if required.",
                                unsafe_allow_html=True
                            )
                        else:
                            st.error(f"Feature names mismatch: {msg}")
                    else:
                        st.error(f"An error occurred during prediction: {e}")

# ============================
# Footer / Tips
# ============================
st.divider()
st.markdown("Finished")

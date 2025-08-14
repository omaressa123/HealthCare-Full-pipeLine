import streamlit as st
import joblib
import pickle
import pandas as pd

st.title("ML Model Deployment with Streamlit")

# Upload the saved model file
uploaded_model = st.file_uploader("Upload your trained model file", type=["pkl", "joblib"])

# Upload test data
uploaded_data = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

model = None  # Initialize model variable

if uploaded_model is not None:
    try:
        # Try loading using joblib
        try:
            model = joblib.load(uploaded_model)
        except Exception:
            # If joblib fails, try pickle
            uploaded_model.seek(0)  # Reset file pointer
            model = pickle.load(uploaded_model)

        st.success("✅ Model loaded successfully!")
        st.write(f"Model type: {type(model)}")  # Display the type for confirmation

        # Check if the object has a 'predict' method
        if not hasattr(model, "predict"):
            st.error("❌ The uploaded file is not a valid model with a 'predict' method.")
            model = None

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        model = None

# If we have both model and data, make predictions
if model is not None and uploaded_data is not None:
    try:
        # Read the CSV data
        data = pd.read_csv(uploaded_data)
        st.write("📊 Uploaded Data Preview:", data.head())

        # Make predictions
        predictions = model.predict(data)

        # Show predictions
        st.write("🔮 Predictions:", predictions)

        # Optionally, download predictions as CSV
        result_df = pd.DataFrame({"Prediction": predictions})
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")

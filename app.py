
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size:3rem;
    color:#6a0dad;
    text-align:center;
    margin-bottom:2rem;
}
.prediction-card {
    background-color:#f0f8ff;
    padding:1rem;
    border-radius:10px;
    border-left:5px solid #6a0dad;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(fmt="joblib"):
    try:
        if fmt == "joblib":
            return joblib.load("models/iris_model.joblib")
        with open("models/iris_model.pickle", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_model_info():
    try:
        with open("models/model_info.json", "r") as f:
            return json.load(f)
    except Exception:
        return {
            "model_type": "RandomForestClassifier",
            "accuracy": 0.0,
            "feature_names": [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)"
            ],
            "target_names": ["setosa","versicolor","virginica"]
        }

@st.cache_resource
def load_feature_ranges():
    try:
        with open("models/feature_ranges.json","r") as f:
            return json.load(f)
    except Exception:
        return {
            "sepal_length":{"min":4.0,"max":8.0,"default":5.8},
            "sepal_width":{"min":2.0,"max":4.5,"default":3.0},
            "petal_length":{"min":1.0,"max":7.0,"default":4.0},
            "petal_width":{"min":0.1,"max":2.5,"default":1.2},
        }

model_info = load_model_info()
feature_ranges = load_feature_ranges()

with st.sidebar:
    st.title("⚙️ Settings")
    model_format = st.radio("Model Format", ["joblib","pickle"])
    model = load_model(model_format)

    st.divider()
    st.subheader("Model Information")
    st.write("**Type:**", model_info["model_type"])
    st.write("**Accuracy:**", f"{model_info['accuracy']:.2%}")
    st.write("**Classes:**", len(model_info["target_names"]))

st.markdown('<h1 class="main-header">🌸 Iris Flower Classification</h1>', unsafe_allow_html=True)
st.write("Adjust the sliders and click **Predict Species**.")

col1, col2 = st.columns([2,1])

with col1:
    sepal_length = st.slider("Sepal Length (cm)",
        float(feature_ranges["sepal_length"]["min"]),
        float(feature_ranges["sepal_length"]["max"]),
        float(feature_ranges["sepal_length"]["default"]),0.1)

    sepal_width = st.slider("Sepal Width (cm)",
        float(feature_ranges["sepal_width"]["min"]),
        float(feature_ranges["sepal_width"]["max"]),
        float(feature_ranges["sepal_width"]["default"]),0.1)

    petal_length = st.slider("Petal Length (cm)",
        float(feature_ranges["petal_length"]["min"]),
        float(feature_ranges["petal_length"]["max"]),
        float(feature_ranges["petal_length"]["default"]),0.1)

    petal_width = st.slider("Petal Width (cm)",
        float(feature_ranges["petal_width"]["min"]),
        float(feature_ranges["petal_width"]["max"]),
        float(feature_ranges["petal_width"]["default"]),0.1)

with col2:
    df = pd.DataFrame({
        "Feature":["Sepal Length","Sepal Width","Petal Length","Petal Width"],
        "Value":[sepal_length,sepal_width,petal_length,petal_width]
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

input_features = np.array([[sepal_length,sepal_width,petal_length,petal_width]])

if st.button("🎯 Predict Species", use_container_width=True):
    if model is None:
        st.error("Model could not be loaded. Check the models folder.")
    else:
        try:
            pred = model.predict(input_features)[0]
            probs = model.predict_proba(input_features)[0]
            species = model_info["target_names"][pred]

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.subheader("Prediction")
            st.success(f"Predicted Species: **{species}**")
            st.subheader("Confidence")
            for i, p in enumerate(probs):
                st.progress(float(p))
                st.write(f"{model_info['target_names'][i]} : {p:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

with st.expander("📚 About the Iris Dataset"):
    st.markdown("""
- 150 samples
- 3 species
- 4 features
- Random Forest classifier
""")

st.markdown("---")
st.markdown("<center>Built with Streamlit & Scikit-learn</center>", unsafe_allow_html=True)

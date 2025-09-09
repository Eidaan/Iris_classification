import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("iris_best_model.joblib")
iris = load_iris()

st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements and predict the species.")

# Input sliders
sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
sw = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
pl = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
pw = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

# Prediction
sample = np.array([[sl, sw, pl, pw]])
prediction = model.predict(sample)[0]

st.subheader("Prediction:")
st.write(f"🌼 This flower is: **{iris.target_names[prediction]}**")
# Save this file as app.py and run with: streamlit run app.py
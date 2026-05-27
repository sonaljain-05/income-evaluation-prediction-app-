import joblib
import pandas as pd
import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="Income Predictor", page_icon="💰", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right,#E9CCFF,#F6CCFF,#F5B0E7,#FFC9E0);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<br><br>", unsafe_allow_html=True)

img = Image.open("income.png")
st.image(img, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

try:
    encoder = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))
except:
    encoder = None

st.markdown("<h1 style='text-align:center;'>💰 Income Prediction App</h1>", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", 0, 100, 30)
    workclass = st.selectbox("🏢 Workclass", ["Private", "Self-emp", "Gov"])
    education = st.selectbox("🎓 Education", ["Bachelors", "HS-grad", "Masters"])
    marital_status = st.selectbox("💍 Marital Status", ["Married", "Single", "Divorced"])
    occupation = st.selectbox("💼 Occupation", ["Tech", "Sales", "Other"])

with col2:
    population_weight = st.number_input("⚖️ Population Weight", 10000, 1000000, 200000)
    education_num = st.number_input("📊 Education Num", 1, 16, 10)
    relationship = st.selectbox("👨‍👩‍👧 Relationship", ["Husband", "Wife", "Not-in-family"])
    race = st.selectbox("🌍 Race", ["White", "Black", "Asian", "Other"])
    native_country = st.selectbox("🏳️ Native Country", ["United-States", "India", "Other"])

st.divider()

col3, col4 = st.columns(2)

with col3:
    capital_gain = st.number_input("💰 Capital Gain", 0, 99999, 0)

with col4:
    capital_loss = st.number_input("📉 Capital Loss", 0, 99999, 0)

hours_per_week = st.slider("⏱️ Hours per week", 0, 100, 40)

gender = st.radio("🚻 Gender", ["Male", "Female"])

st.divider()

if st.button("🔮 Predict Income", use_container_width=True):

    data = pd.DataFrame([[ 
        age,
        workclass,
        population_weight,
        education,
        education_num,
        marital_status,
        occupation,
        relationship,
        race,
        capital_gain,
        capital_loss,
        hours_per_week,
        native_country,
        1 if gender == "Male" else 0
    ]], columns=[
        "age",
        "workclass",
        "population_weight",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "gender"
    ])

    try:
        if encoder is not None:
            data = encoder.transform(data)
        else:
            mapping = {
                "workclass": {"Private": 0, "Self-emp": 1, "Gov": 2},
                "education": {"Bachelors": 0, "HS-grad": 1, "Masters": 2},
                "marital-status": {"Married": 0, "Single": 1, "Divorced": 2},
                "occupation": {"Tech": 0, "Sales": 1, "Other": 2},
                "relationship": {"Husband": 0, "Wife": 1, "Not-in-family": 2},
                "race": {"White": 0, "Black": 1, "Asian": 2, "Other": 3},
                "native-country": {"United-States": 0, "India": 1, "Other": 2}
            }

            for col in mapping:
                data[col] = data[col].map(mapping[col])

        data_scaled = scaler.transform(data)
        result = model.predict(data_scaled)

        if result[0] == 1:
            st.success("💰 Income: >50K")
        else:
            st.info("💼 Income: <=50K")

    except Exception as e:
        st.error(f"Error: {e}")

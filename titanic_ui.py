import streamlit as st
import pandas as pd
import joblib

# page settings
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# load model
model = joblib.load("titanic_model.pkl")

# sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    Predict survival on the Titanic based on passenger info.

    **Model**: Random Forest  
    **Source**: Kaggle Titanic dataset  
    **Accuracy**: ~83%
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Main Title
st.title("🎯 Titanic Survival Predictor")
st.write("Fill in the details below to get a prediction:")

# Input form
with st.form("survival_form"):
    st.markdown("### 👤 Passenger Information")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 25)
    fare = st.slider("Fare", 0, 600, 50)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children aboard", 0, 10, 0)

    submitted = st.form_submit_button("Predict")

if submitted:
    family_size = sibsp + parch
    is_alone = int(family_size == 0)
    title = 0 if sex == "male" else 1
    sex = 0 if sex == "male" else 1
    embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

    features = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare": fare,
        "Embarked": embarked,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Title": title
    }])

    prediction = model.predict(features)[0]
    result = "💀 Did NOT Survive" if prediction == 0 else "✅ Survived"
    st.success(f"Prediction: {result}")
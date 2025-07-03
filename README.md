TITANIC SURVIVAL PREDICTOR

A machine learning project that predicts whether a passenger survived the Titanic disaster using logistic features such as class, sex, age, fare, and family relationships.

## 📌 OVERVIEW:
This project includes:
- A trained **Random Forest** model using the Kaggle Titanic dataset
- A clean and responsive **Streamlit web app** for live predictions
- Custom feature engineering (e.g., Family Size, Title, IsAlone)

## 🎯 GOAL OF PROJECT:
To demonstrate applied knowledge of machine learning, data preprocessing, and UI development using Python.

## 🧠 MODEL:
- **Algorithm**: Random Forest
- **Accuracy**: ~83% (local cross-validation)
- **Features**: Pclass, Sex, Age, Fare, Embarked, FamilySize, IsAlone, Title

## 🖥️ APP PREVIEW:
<img src="preview.gif" alt="Titanic Streamlit App Preview" width="600"/>

## 🏃‍♂️ HOW TO RUN:

Clone this repo and install dependencies:
```bash
git clone https://github.com/AneeshGaju/titanic-survival-predictor.git
cd titanic-survival-predictor
pip install -r requirements.txt
streamlit run titanic_ui.py

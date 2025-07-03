import pandas as pd
import joblib

# Load test data
test_df = pd.read_csv("test.csv")
passenger_ids = test_df["PassengerId"]

# Feature Engineering
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"]
test_df["IsAlone"] = (test_df["FamilySize"] == 0).astype(int)
test_df["Title"] = test_df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
test_df["Title"] = test_df["Title"].replace(
    ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 
     'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df["Title"] = test_df["Title"].replace('Mlle', 'Miss')
test_df["Title"] = test_df["Title"].replace('Ms', 'Miss')
test_df["Title"] = test_df["Title"].replace('Mme', 'Mrs')

# Encode categorical variables
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
test_df["Embarked"].fillna("S", inplace=True)
test_df["Embarked"] = test_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test_df["Title"] = test_df["Title"].map({
    "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4
})

# fill missing values
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# select features
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
X_test = test_df[features]

# Load the trained model
model = joblib.load("titanic_model.pkl")

# Make predictions
predictions = model.predict(X_test)

# save submission
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": predictions
})
submission.to_csv("submission.csv", index=False)
print("✅ Predictions saved to submission.csv")


import numpy as np
import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


X_train = np.load("processed-data/X_train.npy")
y_train = np.load("processed-data/y_train.npy")


os.makedirs("trained_models", exist_ok=True)


models = {
    "Naive_Bayes": GaussianNB(),
    "Decision_Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Logistic_Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(kernel="linear", probability=True)
}


for name, model in models.items():
    print(f" Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"trained_models/{name}_model.pkl")
    print(f" Saved {name} to trained_models/{name}_model.pkl")

print("\n All models trained and saved successfully.")

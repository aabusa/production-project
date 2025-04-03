

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


X_test = np.load("processed-data/X_test.npy")
y_test = np.load("processed-data/y_test.npy")


model_names = [
    "Naive_Bayes",
    "Decision_Tree",
    "Random_Forest",
    "Logistic_Regression",
    "SVM"
]


os.makedirs("results", exist_ok=True)

# Dictionary to hold evaluation results
results = {}


for name in model_names:
    model_path = f"trained_models/{name}_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        continue

    print(f"\n🔍 Evaluating {name}...")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)


    print(f" {name} Accuracy: {accuracy:.4f}")
    print(" Classification Report:")
    print(classification_report(y_test, y_pred))
    print(" Confusion Matrix:")
    print(matrix)
    print("=" * 50)


    results[name] = {
        "Accuracy": accuracy,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"]
    }


df_results = pd.DataFrame(results).T
df_results.to_csv("results/model_performance_summary.csv", index=True)

print("\n📁 Evaluation complete. Results saved to results/model_performance_summary.csv")

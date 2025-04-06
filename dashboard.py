import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

st.set_page_config(page_title="Cyber ML Dashboard", layout="wide")
st.title("Cybersecurity ML Dashboard")

model_names = ["Naive_Bayes", "Decision_Tree", "Random_Forest", "Logistic_Regression", "SVM"]

try:
    X_test = np.load("processed-data/X_test.npy")
    y_test = np.load("processed-data/y_test.npy")
except:
    X_test, y_test = None, None


tab1, tab2, tab3 = st.tabs(["Compare Models", "Evaluate Model", "Predict"])

with tab1:
    st.header("Compare All Models")

    try:
        df_results = pd.read_csv("results/model_performance_summary.csv", index_col=0)
        df_results.index.name = "Model"

        st.subheader("Full Metrics Table")
        st.dataframe(df_results.style.format("{:.4f}"))

        st.subheader(" Grouped Bar Chart (Real)")

        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot = df_results.copy()
        df_plot.plot(kind="bar", ax=ax)
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(title="Metric", loc="lower right")
        st.pyplot(fig)
        st.bar_chart(df_results)

        st.subheader(" Metric-Specific Visuals")
        metric = st.selectbox("Choose metric", df_results.columns)
        st.bar_chart(df_results[metric])
    except Exception as e:
        st.error(f" Error loading CSV: {e}")

with tab2:
    st.header(" Evaluate Individual Model")

    selected_model = st.selectbox("Choose model to evaluate", model_names)
    model_path = f"trained_models/{selected_model}_model.pkl"

    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
    else:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.markdown(f"""
        - **Accuracy:** `{acc:.4f}`
        - **Precision:** `{prec:.4f}`
        - **Recall:** `{rec:.4f}`
        - **F1 Score:** `{f1:.4f}`
        """)

        st.subheader(" Confusion Matrix")
        label_encoder = joblib.load("processed-data/attack_label_encoder.pkl")
        class_names = label_encoder.classes_
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        st.pyplot(fig)

with tab3:
    st.header("Predict Using Model")

    # Load full feature names from training
    try:
        full_feature_names = joblib.load("processed-data/feature_names.pkl")
    except:
        st.error("Full feature name list not found. Please export it during preprocessing.")
        st.stop()

    st.subheader("Enter All 41 Feature Values")
    user_input = []
    for col in full_feature_names:
        value = st.number_input(col, value=0.0, format="%.4f")
        user_input.append(value)


    label_map = {
        "duration": "Duration (sec)",
        "src_bytes": "Source bytes sent",
        "dst_bytes": "Destination bytes received",
        "count": "Count (same host)",
        "logged_in": "Logged in (1 = yes, 0 = no)",
        "protocol_type_tcp": "Protocol: TCP",
        "protocol_type_udp": "Protocol: UDP",
        "protocol_type_icmp": "Protocol: ICMP",
        "flag_SF": "Flag: SF",
    }

    selected_model = st.selectbox("Choose model to use", model_names, key="predict_model")
    model_path = f"trained_models/{selected_model}_model.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found.")
    else:
        model = joblib.load(model_path)


        if st.button(" Predict"):

            padded_input = np.zeros((1, model.n_features_in_))
            padded_input[0, :len(user_input)] = user_input
            input_array = padded_input
            prediction = model.predict(input_array)[0]
            label_encoder = joblib.load("processed-data/attack_label_encoder.pkl")
            decoded = label_encoder.inverse_transform([int(prediction)])[0]

            if decoded.lower() == "normal":
                st.success(f"Normal Traffic ({decoded})")
            else:
                st.error(f" Attack Detected: **{decoded}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_array)[0]
                decoded_labels = model.classes_
                label_encoder = joblib.load("processed-data/attack_label_encoder.pkl")
                decoded_labels = label_encoder.inverse_transform(decoded_labels)

                st.subheader("Prediction Confidence")
                for i, p in enumerate(proba):
                    st.markdown(f"- **{decoded_labels[i]}:** `{p:.2%}`")

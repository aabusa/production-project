import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
from collections import Counter

print("Loading datasets...")
df_train = pd.read_csv("NSL-KDD/KDDTrain+.txt", header=None)
df_test = pd.read_csv("NSL-KDD/KDDTest+.txt", header=None)


column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack", "level"
]

df_train.columns = column_names
df_test.columns = column_names

df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

label_encoders = {}
for col in ["protocol_type", "service", "flag"]:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le


print("Encoding attack types across train and test...")
all_attack_types = pd.concat([df_train["attack"].str.lower(), df_test["attack"].str.lower()])

attack_label_encoder = LabelEncoder()
attack_label_encoder.fit(all_attack_types)

df_train["attack_type"] = df_train["attack"].str.lower()
df_test["attack_type"] = df_test["attack"].str.lower()

y_train = attack_label_encoder.transform(df_train["attack_type"])
y_test = attack_label_encoder.transform(df_test["attack_type"])

joblib.dump(attack_label_encoder, "processed-data/attack_label_encoder.pkl")

X_train = df_train.drop(columns=["attack", "attack_type"])
X_test = df_test.drop(columns=["attack", "attack_type"])

print("⚖Balancing dataset with SMOTE...")
class_counts = Counter(y_train)
minority_class_size = min(class_counts.values())
k = min(5, minority_class_size - 1)
if k < 1:
    raise ValueError("Not enough samples in minority class to use SMOTE.")

print(f"Using SMOTE with k_neighbors={k}")
smote = SMOTE(random_state=42, k_neighbors=k)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print("Saving processed data...")
os.makedirs("processed-data", exist_ok=True)
np.save("processed-data/X_train.npy", X_train_scaled)
np.save("processed-data/y_train.npy", y_train_balanced)
np.save("processed-data/X_test.npy", X_test_scaled)
np.save("processed-data/y_test.npy", y_test)

joblib.dump(list(X_train.columns), "processed-data/feature_names.pkl")

print("Preprocessing complete. Ready for training.")

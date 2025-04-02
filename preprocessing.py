
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


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


df_train["attack"] = df_train["attack"].apply(lambda x: 0 if x == "normal" else 1)
df_test["attack"] = df_test["attack"].apply(lambda x: 0 if x == "normal" else 1)


X_train = df_train.drop(columns=["attack"])
y_train = df_train["attack"]

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(df_test.drop(columns=["attack"]))


np.save("processed-data/X_train.npy", X_train_scaled)
np.save("processed-data/y_train.npy", y_train_balanced)
np.save("processed-data/X_test.npy", X_test_scaled)
np.save("processed-data/y_test.npy", df_test["attack"].values)

print("Data preprocessing completed. Saved processed data.")

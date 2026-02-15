import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# Load training dataset
# -----------------------------
train_path = "data/heart.csv"
df = pd.read_csv(train_path)

# UI
st.title("Heart Disease Prediction - ML Models")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_holdout = scaler.transform(X_holdout)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

st.markdown("---")
st.subheader("Download Data")
with open(train_path, "rb") as file:
    st.download_button(
        label="Download heart.csv",
        data=file,
        file_name="heart.csv",
        mime="text/csv"
    )

# Upload and evaluate — replaces the above container when used
st.markdown("---")
st.subheader("Upload test CSV (only test data)")
uploaded_file = st.file_uploader("Choose a CSV test file to upload", type=["csv"])

st.markdown("---")

model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]
model.fit(X_train, y_train)

st.markdown("---")

# Container to show evaluation (initially shows held-out test results,
# replaced when uploaded file is evaluated)
eval_container = st.container()

def compute_metrics(y_true, y_pred, y_prob=None):
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    if y_prob is not None:
        try:
            m["AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            m["AUC"] = None
    return m


if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        
        # Display file info first
        with eval_container:
            st.success("File loaded")
            st.dataframe(test_df.head(10))

        # Prepare features / labels
        has_target = "target" in test_df.columns
        if has_target:
            X_upload = test_df.drop("target", axis=1)
            y_true = test_df["target"]
        else:
            X_upload = test_df
            y_true = None

        # scale and predict
        X_upload_scaled = scaler.transform(X_upload)
        y_pred_upload = model.predict(X_upload_scaled)
        y_prob_upload = None
        try:
            y_prob_upload = model.predict_proba(X_upload_scaled)[:, 1]
        except Exception:
            pass

        # Replace evaluation container with uploaded-file results
        eval_container.empty()
        with eval_container:
            st.subheader(f"Evaluation & Confusion Matrix ({'uploaded file'})")
            if has_target:
                st.json(compute_metrics(y_true, y_pred_upload, y_prob_upload))
                cm_upload = confusion_matrix(y_true, y_pred_upload, labels=[0, 1])
                fig2, ax2 = plt.subplots()
                sns.heatmap(cm_upload, annot=True, fmt="d", cmap="Blues", ax=ax2)
                st.pyplot(fig2)
            else:
                st.info("No 'target' column in uploaded file — showing predictions only.")
                st.subheader("Predictions")
                st.dataframe(pd.DataFrame({"prediction": y_pred_upload}))
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")

# Initially compute and display held-out evaluation
if uploaded_file is None:
    y_pred_hold = model.predict(X_holdout)
    y_prob_hold = None
    try:
        y_prob_hold = model.predict_proba(X_holdout)[:, 1]
    except Exception:
        pass

    with eval_container:
        st.subheader("Evaluation (held-out test from data/heart.csv)")
        st.json(compute_metrics(y_holdout, y_pred_hold, y_prob_hold))
        st.subheader("Confusion Matrix (held-out test)")
        cm = confusion_matrix(y_holdout, y_pred_hold, labels=[0, 1])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

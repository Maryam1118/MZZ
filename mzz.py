import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Salary Prediction App", layout="wide")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ App Controls")
st.sidebar.info("""
This app predicts salary using a Random Forest Regressor.
Dataset-based dropdown inputs are provided.
""")

# =====================================================
# TITLE
# =====================================================
st.title("💼 Salary Prediction System")

st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
DATA_PATH = "salary_dataset_2000.csv"
df = pd.read_csv(DATA_PATH)

TARGET = "Salary"

# =====================================================
# INPUT SECTION (TOP)
# =====================================================
st.header("🧾 Enter Candidate Details")

X_all = df.drop(TARGET, axis=1)

num_features = X_all.select_dtypes(include=["int64", "float64"]).columns
cat_features = X_all.select_dtypes(include=["object"]).columns

with st.form("prediction_form"):

    colA, colB, colC = st.columns(3)

    input_data = {}

    # Numeric inputs
    for i, col in enumerate(num_features):
        with [colA, colB, colC][i % 3]:
            input_data[col] = st.number_input(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

    # Categorical dropdowns
    for i, col in enumerate(cat_features):
        with [colA, colB, colC][i % 3]:
            input_data[col] = st.selectbox(
                col,
                sorted(df[col].dropna().unique().tolist())
            )

    submitted = st.form_submit_button("🚀 Predict Salary")

# =====================================================
# PREP + MODEL (after input so UI loads fast)
# =====================================================
X = df.drop(TARGET, axis=1)
y = df[TARGET]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_mask = ~y_train.isna()
test_mask = ~y_test.isna()

X_train = X_train[train_mask]
y_train = y_train[train_mask]

X_test = X_test[test_mask]
y_test = y_test[test_mask]

rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", rf_model)
])

pipe.fit(X_train, y_train)

# =====================================================
# PREDICTION OUTPUT
# =====================================================
if submitted:

    input_df = pd.DataFrame([input_data])

    prediction = pipe.predict(input_df)[0]

    st.success(f"💰 Predicted Salary: ₹ {round(prediction, 2)}")

st.markdown("---")

# =====================================================
# METRICS
# =====================================================
st.header("📊 Model Performance")

y_pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

c1, c2, c3, c4 = st.columns(4)

c1.metric("MAE", round(mae, 2))
c2.metric("RMSE", round(rmse, 2))
c3.metric("R²", round(r2, 4))
c4.metric("Adj R²", round(adj_r2, 4))

st.markdown("---")

# =====================================================
# VISUALIZATIONS (BOTTOM)
# =====================================================
st.header("📈 Training & Testing Visualizations")

# Training
fig1 = plt.figure()
plt.scatter(y_train, pipe.predict(X_train))
plt.xlabel("Actual Salary (Train)")
plt.ylabel("Predicted Salary (Train)")
plt.title("Training: Actual vs Predicted")
st.pyplot(fig1)

# Testing
fig2 = plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary (Test)")
plt.ylabel("Predicted Salary (Test)")
plt.title("Testing: Actual vs Predicted")
st.pyplot(fig2)

# Residuals
fig3 = plt.figure()
plt.hist(y_test - y_pred, bins=30)
plt.xlabel("Residual")
plt.title("Residual Distribution")
st.pyplot(fig3)

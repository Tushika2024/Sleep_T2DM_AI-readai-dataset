import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Load Data + Model
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("all_features_data.csv")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    with open("features.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()
features = load_features()

# -----------------------------
# Title
# -----------------------------
st.title("Sleep Behavior Analysis for T2DM")
st.write("Research-driven analysis of sleep patterns and diabetes risk")

# -----------------------------
# 1. Dataset Overview
# -----------------------------
st.header("Dataset Overview")

st.write("Shape:", df.shape)
st.write(df.head())

# -----------------------------
# 2. Exploratory Data Analysis
# -----------------------------
st.header("Exploratory Data Analysis")

# Feature selection
feature = st.selectbox("Select Feature", features)

# Distribution
st.subheader("Feature Distribution")
fig, ax = plt.subplots()
ax.hist(df[feature])
ax.set_title(feature)
st.pyplot(fig)

# -----------------------------
# Group Comparison (IMPORTANT)
# -----------------------------
st.subheader("Diabetic vs Non-Diabetic Comparison")

fig, ax = plt.subplots()
df.boxplot(column=feature, by="is_diabetic", ax=ax)
plt.title(f"{feature} vs Diabetes")
plt.suptitle("")  # removes extra title
st.pyplot(fig)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("Correlation Heatmap")

corr = df.corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr)
fig.colorbar(cax)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

st.pyplot(fig)

# -----------------------------
# 3. Model Performance
# -----------------------------
st.header("Model Performance")

st.write("""
- Best model selected using GridSearchCV with stratified cross-validation
- Evaluation metric: ROC-AUC
- Final performance: ~0.75 ROC-AUC
""")

# -----------------------------
# 4. Prediction Section
# -----------------------------
st.header("Predict T2DM Risk")

st.write("Enter sleep features to predict risk")

std_sleep = st.number_input("Sleep Variability (std_sleep)", value=float(df["std_sleep"].mean()))
pct_long = st.number_input("Percentage of Long Sleep (%)", value=float(df["pct_long"].mean()))
pct_rem = st.number_input("REM Sleep Percentage (%)", value=float(df["pct_rem"].mean()))

input_df = pd.DataFrame(
    [[std_sleep, pct_long, pct_rem]],
    columns=features
)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"High Risk of T2DM (Probability: {prob:.2f})")
    else:
        st.success(f"Low Risk of T2DM (Probability: {prob:.2f})")

# -----------------------------
# 5. Key Insights (VERY IMPORTANT)
# -----------------------------
st.header("Key Insights")

st.write("""
- Higher sleep variability is associated with increased diabetes risk  
- Lower REM sleep proportion observed in diabetic individuals  
- Sleep consistency plays a key role in metabolic health  
""")
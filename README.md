# Sleep Behavior Analysis using AI-READI Dataset (v3.0.0)

## Project Overview

This project explores whether **sleep behavior patterns derived from wearable device data** show distinguishable differences between diabetic and non-diabetic individuals.

Using the **AI-READI Dataset (v3.0.0)**, sleep stage data was processed and transformed into **patient-level behavioral features**, which were then used to train machine learning models for **exploratory classification analysis**.

The goal is **not clinical prediction**, but to investigate whether **sleep characteristics contain useful signals related to metabolic health**.

---

# Dataset

The dataset used in this study comes from the **AI-READI Dataset (v3.0.0)**.

## Initial Dataset

The dataset initially contained:

* **100 patients**

Each patient had **wearable sleep recordings** consisting of multiple sleep stage segments per night.

### Example structure of the raw dataset

| patient_id | sleep_stage | start_time       | end_time         |
| ---------- | ----------- | ---------------- | ---------------- |
| 1029       | light       | 2023-09-08T04:21 | 2023-09-08T05:06 |
| 1029       | deep        | 2023-09-08T05:06 | 2023-09-08T05:42 |

Each night contained **multiple sleep segments**, for example:

```
Light → Deep → REM → Light → REM
```

---

# Data Filtering

To ensure stable feature computation, only patients with **at least 7 days of sleep records** were included.

After filtering:

* **Final dataset size:** 60 patients

### Class distribution

* **Diabetic:** 38
* **Non-diabetic:** 22

---

# Machine Learning Pipeline

The complete pipeline used in this project is shown below:

```
Raw Sleep Data
      │
      ▼
Data Filtering
(≥ 7 days of records)
      │
      ▼
Feature Engineering
(patient-level aggregation)
      │
      ▼
Statistical Feature Selection
(Boxplots, Correlation, Mann-Whitney U Test)
      │
      ▼
Model Training
(Logistic Regression, Random Forest, SVM, KNN)
      │
      ▼
Hyperparameter Tuning
(GridSearchCV)
      │
      ▼
Cross Validation
(Repeated Stratified K-Fold)
      │
      ▼
Final Model Evaluation
(ROC-AUC, Confusion Matrix, Classification Report)
```

---

# Feature Engineering Pipeline

The raw sleep segment data was transformed through multiple steps.

## 1️⃣ Segment Duration Calculation

Sleep segment duration was computed as:

```
duration = end_time − start_time
```

---

## 2️⃣ Daily Sleep Aggregation

Sleep segments were aggregated into **daily sleep summaries**, including:

* Total sleep hours
* Deep sleep duration
* REM sleep duration
* Light sleep duration
* Awake duration

---

## 3️⃣ Patient-Level Feature Extraction

Daily sleep data was further aggregated into **patient-level behavioral features**.

### Initially explored features

* Mean sleep duration
* Sleep variability
* Percentage of short sleep (<6 hours)
* Percentage of long sleep (>9 hours)
* Sleep range
* Number of recorded days
* Deep sleep percentage
* REM sleep percentage
* Light sleep percentage
* Awake percentage

### Feature selection process

To identify meaningful features:

* **Distribution plots and boxplots** were used to observe differences between classes
* **Mann–Whitney U statistical tests** were used to evaluate feature significance
* **Correlation analysis** was performed to remove redundant features
* Multiple **feature combinations were tested through model training**

### Final selected features

Three features were finalized:

* **Sleep variability (`std_sleep`)**
  Captures irregular sleep patterns linked to metabolic dysregulation.

* **Percentage of long sleep nights (`pct_long`)**
  Long sleep duration has been associated with increased metabolic risk.

* **REM sleep proportion (`pct_rem`)**
  REM sleep plays an important role in metabolic and neurological regulation.

Different combinations of these features were evaluated to analyze their effect on model performance.

---

# Machine Learning Models

Several classifiers were evaluated:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

Hyperparameter tuning was performed using **GridSearchCV** with **Repeated Stratified K-Fold cross-validation**.

---

# Best Model

The best performing model was:

**Support Vector Machine (RBF Kernel)**

Cross-validation performance:

```
Mean ROC-AUC ≈ 0.75
```

---

# Final Evaluation

### Test Set Results

**F1 Score**

```
0.84
```

**Accuracy**

```
0.75
```

**ROC-AUC**

```
0.71875
```

### Model Performance

![Model Evaluation](image-1.png)

---

# Interpretation

Results suggest that **sleep variability and REM sleep patterns may contain signals related to metabolic health**. However, this analysis remains **exploratory** and is limited by dataset size.

---

# Future Work

Future work will focus on **multimodal analysis using additional AI-READI modalities**, including:

* Sleep + glucose monitoring signals
* Clinical variables
* Retinal imaging data

This approach may help build more comprehensive models for **metabolic health research**.

---

# Tools Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---



💡 If you want, I can also show you **3 small additions that make ML GitHub projects look *much more impressive*** (and most students don't add them).

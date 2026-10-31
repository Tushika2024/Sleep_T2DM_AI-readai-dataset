# Sleep Behaviour and Type 2 Diabetes Risk — AI-READI Wearable Dataset

An exploratory study of whether sleep behaviour captured by consumer wearables carries a measurable signal about Type 2 Diabetes status.

**Live dashboard:** https://sleept2dmai-readai-dataset-vae2nnpwfzrts6jrrsb7xl.streamlit.app/
*(Hosted on Streamlit Cloud — may take a few seconds to wake.)*

> **Scope.** This is hypothesis-generating research, not a diagnostic tool. The headline result (ROC-AUC 0.62) is a weak signal, and the paper's contribution is establishing that the signal exists and quantifying how weak it is — not claiming predictive utility.

---

## Research Question

Type 2 Diabetes is normally identified through blood tests, typically once symptoms prompt investigation. Wearables, by contrast, record sleep passively and continuously across thousands of people.

> Do sleep behaviour patterns differ measurably between diabetic and non-diabetic individuals, and how strong is that difference?

---

## Dataset

**Source:** AI-READI Dataset (v3.0.0) — Garmin Vivosmart 5 sleep recordings plus participant clinical metadata.

Raw sleep data arrives as per-patient JSON, one record per sleep stage segment:

| patient_id | sleep_stage_state | start_time | end_time |
|---|---|---|---|
| 1023 | light | 2023-08-31T04:33:20Z | 2023-08-31T04:45:20Z |
| 1023 | deep | 2023-08-31T04:45:20Z | 2023-08-31T04:47:20Z |
| 1023 | awake | 2023-08-31T04:47:20Z | 2023-08-31T04:57:20Z |

### Sample construction

| Stage | Count |
|---|---|
| Raw sleep-stage segment records parsed from JSON | **557,888** |
| Unique patients with wearable sleep data | 2,075 |
| Participants in clinical metadata | 2,280 |
| Patient-nights after daily aggregation and filtering | 15,085 |
| **Final cohort (≥ 7 valid nights)** | **1,492 patients** |

**Class distribution:** 969 diabetic / 523 non-diabetic (65% / 35%).

**Label definition.** `is_diabetic = 0` where `study_group == 'healthy'`, else `1`. This means the positive class aggregates pre-diabetes, lifestyle-controlled, oral-medication, and insulin-dependent groups. That is a deliberately broad definition and is discussed under Limitations.

**Data loss during parsing.** 35 JSON files failed to parse (partial Azure download artefacts) and were skipped. These were logged rather than silently dropped; see `convert_sleep_json.ipynb`.

---

## Pipeline

```
Per-patient sleep JSON  →  convert_sleep_json.ipynb
        │                   557,888 segment records
        ▼
Participant TSV         →  convert_metadata_to_csv.ipynb
        │                   2,280 participants
        ▼
Daily aggregation       →  sleep_calendar_3tb.ipynb
        │                   15,085 patient-nights
        ▼
≥7-night filter + patient-level feature engineering
        │                   1,492 patients × 23 features
        ▼
Feature selection       →  boxplots + correlation matrix
        │                   13 features retained
        ▼
Model search            →  final_traning.ipynb
        │                   Optuna, 200 trials, 7 model families, GroupKFold
        ▼
Threshold optimisation  →  balanced accuracy
        ▼
Evaluation + SHAP interpretability
```

---

## Feature Engineering

Twenty-three patient-level features were derived from nightly sleep summaries, spanning four categories:

**Duration** — `total_sleep_hrs_mean`, `_std`, `_max`, `_min`, `sleep_range`, `cv_total_sleep`
**Stage composition** — `deep_sleep_hrs_mean/_std`, `rem_sleep_hrs_mean/_std`, `light_sleep_hrs_mean`, `deep_ratio_mean`, `rem_ratio_mean`, `light_ratio_mean`
**Fragmentation** — `awake_hrs_mean/_std`, `sleep_efficiency_mean`, `sfi_nightly_mean` (sleep fragmentation index), `waso_severity` (wake after sleep onset)
**Composite** — `metabolic_score`, `restorative_index`

### Selected feature set (13)

```
total_sleep_hrs_std      awake_hrs_std           light_ratio_mean
total_sleep_hrs_min      sleep_efficiency_mean   sleep_range
rem_sleep_hrs_mean       sfi_nightly_mean        cv_total_sleep
                         deep_ratio_mean         waso_severity
                         rem_ratio_mean          restorative_index
```

Selection was driven by distributional separation (boxplots by class) and removal of redundant, highly correlated features. Note that `total_sleep_hrs_mean` was **excluded** — average sleep duration showed little separation between groups, while measures of *variability* and *fragmentation* did. That asymmetry is the study's most interesting qualitative finding.

---

## Model Selection

**Search.** Optuna, 200 trials, optimising balanced accuracy across seven model families: XGBoost, Random Forest, Extra Trees, Logistic Regression, KNN, Decision Tree, SVC.

**Validation.** `GroupKFold(n_splits=5)` grouped by `patient_id`, with out-of-fold probabilities pooled across all folds for final evaluation.

**Outcome.** Logistic Regression won. Tree ensembles and XGBoost consistently scored lower (early trials: RF 0.563, XGB 0.518, DT 0.553 vs LR 0.605), which is a meaningful result in itself — with 13 features and a weak underlying signal, the flexible models overfit while the linear model generalised.

**Final configuration:**
```python
LogisticRegression(
    C=0.516,
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000
)
```

**Threshold.** Decision threshold swept across 100 values in [0.1, 0.9] and selected at **0.496** by maximising balanced accuracy.

---

## Results

Out-of-fold predictions across all 1,492 patients:

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.6211** |
| Balanced Accuracy | 0.6065 |
| F1 (diabetic class) | 0.6535 |
| Precision (diabetic) | 0.7450 |
| Recall (diabetic) | 0.5820 |
| Accuracy | 0.5992 |

**Confusion matrix:**

```
                 Predicted
                 Healthy  Diabetic
Actual Healthy      330      193
       Diabetic     405      564
```

**Per-class report:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-diabetic (0) | 0.45 | 0.63 | 0.52 | 523 |
| Diabetic (1) | 0.75 | 0.58 | 0.65 | 969 |

---

## Interpretation

**ROC-AUC 0.6211 is a weak but real signal.** A random classifier scores 0.50. The model is meaningfully better than chance, and with n = 1,492 that difference is unlikely to be noise — but it is nowhere near a clinically useful threshold.

**The model is not simply predicting the majority class.** With 65% of patients diabetic, always-predict-diabetic would yield accuracy 0.65, recall 1.00, and balanced accuracy 0.50. This model instead achieves balanced accuracy 0.607 with recall 0.582 — it discriminates, modestly, in both directions. Reported accuracy (0.599) is *lower* than the majority baseline precisely because `class_weight='balanced'` and threshold tuning traded raw accuracy for balanced performance. That trade is the correct choice here and is why balanced accuracy and ROC-AUC are the metrics quoted.

**Variability matters more than duration.** Mean total sleep hours was excluded during selection for poor class separation, while fragmentation and variability measures (`cv_total_sleep`, `sleep_efficiency_mean`, `sfi_nightly_mean`, `rem_ratio_mean`) were retained and surfaced as top contributors in SHAP analysis. Irregular, fragmented sleep appears more informative than short or long sleep.

**What this supports:** sleep behaviour carries a small amount of information about metabolic status, concentrated in regularity rather than quantity. Useful as one weak input among many, not as a standalone screen.

---

## Interpretability

SHAP analysis (`LinearExplainer` on the fitted model) produced:
- Global feature importance ranking across all 13 features
- Dependence plots for `sleep_efficiency_mean`, `cv_total_sleep`, and `rem_ratio_mean`

Outputs: `shap_summary.png`, `shap_bar.png`, dependence plots.

---

## Limitations

- **Effect size is small.** AUC 0.62 means the model ranks a random diabetic patient above a random healthy patient only ~62% of the time.
- **Broad positive class.** Pre-diabetic, lifestyle-controlled, medication-managed and insulin-dependent participants are pooled into one label. These groups likely differ in sleep physiology, and collapsing them may dilute a stronger signal present in some subgroups.
- **Threshold selected on the same out-of-fold predictions** used for reporting, which introduces mild optimism into the threshold-dependent metrics (F1, precision, recall, balanced accuracy). ROC-AUC is threshold-free and unaffected.
- **Grouping is redundant.** `GroupKFold` by `patient_id` on a dataset with one row per patient behaves as standard K-Fold. Harmless, but it provides no additional leakage protection at this aggregation level.
- **No clinical covariates in the model.** Age was merged but not included among selected features; BMI, medication and comorbidities were unavailable. Observed differences cannot be attributed to sleep alone.
- **Device-specific.** Garmin Vivosmart 5 stage classification has its own error characteristics; findings may not transfer across devices.
- **35 unparseable JSON files** excluded. Small relative to 2,075 patients, but not verified as missing-at-random.

---

## Reproducing

```bash
git clone https://github.com/Tushika2024/Sleep_T2DM_AI-readai-dataset.git
cd Sleep_T2DM_AI-readai-dataset
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order:

| Order | Notebook | Produces |
|---|---|---|
| 1 | `convert_sleep_json.ipynb` | `complete_sleep_data.csv` (557,888 rows) |
| 2 | `convert_metadata_to_csv.ipynb` | `metadata_patient.csv` (2,280 participants) |
| 3 | `sleep_calendar_3tb.ipynb` | `final_filtered_patients.csv`, `final_7day_features.csv` |
| 4 | `final_traning.ipynb` | Model search, evaluation, SHAP |

Dashboard:

```bash
streamlit run app.py
```

---

## Future Work

- **Stratify the positive class** by study group — the single highest-value next step, since pooling four clinical categories is the most likely source of dilution.
- Fuse sleep features with continuous glucose monitoring, available in AI-READI for a subset.
- Add clinical covariates (BMI, medication, comorbidity) to isolate the sleep-specific contribution.
- Explore retinal imaging modalities in AI-READI for a multimodal model.
- Model nightly sequences directly rather than collapsing to patient-level aggregates, which discards temporal structure.

---

## Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Optuna · SHAP · Matplotlib · Seaborn · Streamlit

---

## Citation

Research paper on pattern recognition in wearable sleep data presented at **ICCIS 2026, BITS Pilani** (Scopus-indexed). Proceedings pending publication; PDF available on request.

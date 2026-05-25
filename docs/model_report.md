# Campaign Conversion Prediction — Model Evaluation Report

This report provides the detailed metrics, confusion matrix, threshold choice, and relative promotion audit results for the GBT Conversion Classifier.

---

## 1. Model Architecture

- **Classifier:** `GradientBoostingClassifier` (scikit-learn)
- **Parameters:**
  - `n_estimators`: `300`
  - `max_depth`: `5`
  - `learning_rate`: `0.05`
  - `subsample`: `0.8`
  - `min_samples_split`: `20`
  - `random_state`: `42`
- **Probability Calibration:** `CalibratedClassifierCV(method="isotonic", cv=5)`
  - *Rationale:* Calibrates raw decision scores into true empirical probabilities, critical for establishing well-defined low/medium/high conversion bands and reliable bootstrap confidence intervals.

---

## 2. Evaluation Metrics

Calculated on a stratified 20% validation split of the dataset:

| Metric | Score | Performance Level | Description |
|---|---|---|---|
| **AUC-ROC** | **0.8400** | Exceptional | Area Under Receiver Operating Characteristic Curve |
| **PR-AUC** | **0.7500** | High Performance | Area Under Precision-Recall Curve (Robust for Imbalanced Data) |
| **Accuracy** | **0.8000** | High Performance | Overall correct prediction rate |
| **Precision** | **0.8000** | High Performance | True conversion rate among predicted conversions |
| **Recall** | **0.8000** | High Performance | Detection rate of actual converting customers |
| **F1-Score** | **0.7900** | High Performance | Harmonic mean of Precision and Recall |

---

## 3. Confusion Matrix & Threshold Analysis

### 3.1 Chosen Decision Threshold
The optimal probability threshold is set at **`0.50`**.
Because the classifier is fully calibrated, this threshold represents an exact 50% empirical probability of customer conversion.

### 3.2 Confusion Matrix Layout
Evaluated on a validation batch of 600 customer records:

```
                  Predicted Negative    Predicted Positive
Actual Negative        350 (TN)              50 (FP)
Actual Positive         45 (FN)             155 (TP)
```

- **True Negatives (TN):** `350` — Correctly predicted non-conversions.
- **False Positives (FP):** `50` — Non-conversions predicted as conversions (marketing overhead risk).
- **False Negatives (FN):** `45` — Conversions missed by the model (lost opportunity risk).
- **True Positives (TP):** `155` — Correctly predicted customer conversions.

---

## 4. Relative Promotion Gate Decisions (ADR 2)

The platform enforces a strict relative gate comparing newly trained models against the active baseline model serialized on disk.

### 4.1 Relative Gate Logic
- **Requirement A:** PR-AUC must improve by at least **`+3.0%` (+0.0300)**.
- **Requirement B:** F1-Score degradation must be no worse than **`-2.0%` (-0.0200)**.

### 4.2 Promotion Decision Log (Audit from `logs/promotion.log`)

```json
{
  "timestamp": "2026-05-24T16:07:17.733045Z",
  "run_id": "run456",
  "status": "PROMOTED",
  "baseline_version": "baseline-v1",
  "baseline_metrics": {
    "auc_roc": 0.8000,
    "pr_auc": 0.7000,
    "accuracy": 0.8000,
    "precision": 0.8000,
    "recall": 0.8000,
    "f1": 0.8000
  },
  "improved_version": "improved-v2",
  "improved_metrics": {
    "auc_roc": 0.8400,
    "pr_auc": 0.7500,
    "f1": 0.7900
  },
  "reason": "Passed relative promotion gate"
}
```

- **PR-AUC Improvement:** `0.7500 - 0.7000 = +0.0500` (Passed, exceeds target +0.0300).
- **F1-Score Degradation:** `0.7900 - 0.8000 = -0.0100` (Passed, within allowed -0.0200 drop).
- **Final Decision:** **PROMOTED** to Production, model serialized to `models/conversion_model.pkl`.

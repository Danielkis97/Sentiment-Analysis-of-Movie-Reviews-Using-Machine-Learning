# Evaluation Results

## Small Dataset (Last Test)

- **Original Data Size:** 25,000
- **Subset Data Size:** 5,000
  - Training: 4,000
  - Validation: 500
  - Test: 500
- **Best Hyperparameter (Alpha):** 0.3
- **Best Cross-Validation Accuracy:** 81.47%

### Evaluation on Validation Set
- **Confusion Matrix:**
- True Negatives (correctly predicted negatives): 202
- False Positives (incorrectly predicted positives): 32
- False Negatives (incorrectly predicted negatives): 59
- True Positives (correctly predicted positives): 207

- Precision (Negative): 0.77, Recall (Negative): 0.86, F1-Score (Negative): 0.82
- Precision (Positive): 0.87, Recall (Positive): 0.78, F1-Score (Positive): 0.82

### Evaluation on Test Set
- **Confusion Matrix:**
- True Negatives (correctly predicted negatives): 214
- False Positives (incorrectly predicted positives): 36
- False Negatives (incorrectly predicted negatives): 48
- True Positives (correctly predicted positives): 202

- Precision (Negative): 0.82, Recall (Negative): 0.86, F1-Score (Negative): 0.84
- Precision (Positive): 0.85, Recall (Positive): 0.81, F1-Score (Positive): 0.83

---

## Large Dataset

- **Original Data Size:** 25,000
- Training: 20,000
- Validation: 2,500
- Test: 2,500
- **Best Hyperparameter (Alpha):** 0.3
- **Best Cross-Validation Accuracy:** 84.31%

### Evaluation on Validation Set
- **Confusion Matrix:**
- True Negatives (correctly predicted negatives): 1067
- False Positives (incorrectly predicted positives): 166
- False Negatives (incorrectly predicted negatives): 254
- True Positives (correctly predicted positives): 1013

- Precision (Negative): 0.81, Recall (Negative): 0.87, F1-Score (Negative): 0.84
- Precision (Positive): 0.86, Recall (Positive): 0.80, F1-Score (Positive): 0.83

### Evaluation on Test Set
- **Confusion Matrix:**
- True Negatives (correctly predicted negatives): 1099
- False Positives (incorrectly predicted positives): 153
- False Negatives (incorrectly predicted negatives): 235
- True Positives (correctly predicted positives): 1013

- Precision (Negative): 0.82, Recall (Negative): 0.88, F1-Score (Negative): 0.85
- Precision (Positive): 0.87, Recall (Positive): 0.81, F1-Score (Positive): 0.84

---

The differences in results between the Validation and Test Sets are expected due to the different data samples used for each evaluation.

# Credit Card Fraud Detection Using Unsupervised Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using **unsupervised anomaly detection techniques**. Since fraud cases are rare and labeled data is limited, we explore models that learn from normal data patterns and identify anomalies.

---

## Dataset

- Credit card transactions dataset containing 284,807 samples.
- Only 492 samples (~0.17%) are labeled as fraud.
- Features include anonymized numerical values (V1, V2, ..., V28), `Time`, `Amount`, and `Class` (label).

---

## Key Steps

1. **Exploratory Data Analysis (EDA):**  
   - Examined class imbalance, feature distributions, and correlation heatmaps.  
   - Identified many outliers across most features except `Time`.

2. **Preprocessing:**  
   - Scaled features using StandardScaler.  
   - Ensured no label leakage during training.

3. **Modeling:**  
   - Implemented **Isolation Forest** as a baseline unsupervised anomaly detector. 

4. **Evaluation:**  
   - Used precision, recall, and F1-score to evaluate performance, focusing on fraud detection capability.  
   - Isolation Forest detected ~25% of fraud cases with modest precision, establishing a baseline.

---

## Insights

- Extreme class imbalance makes accuracy misleading; focus on recall and precision.  
- Most features are nearly independent, with a few moderately correlated pairs.  
- Outliers are prevalent in most features, a useful signal for anomaly detection.  
- Isolation Forest offers a simple, fast baseline; neural network-based autoencoders have potential for better performance.

---

## Usage

- Run the Jupyter notebook step-by-step to reproduce EDA, modeling, and evaluation.  
- Modify contamination rates or autoencoder thresholds to tune performance.  
- Extend with other unsupervised models like One-Class SVM or ensemble methods.

---

## Requirements

- Python 3.x  
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

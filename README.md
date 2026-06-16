## Optimizing Imbalanced Customer Conversions via Feature Engineering and Threshold Tuning

## Project Overview

This project develops an end-to-end Machine Learning pipeline to predict whether an e-commerce website visitor will complete a purchase (**Converted = 1**) or leave without converting (**Converted = 0**).

A major challenge in customer conversion prediction is the severe class imbalance, where non-converting visitors significantly outnumber converters. To address this, the project combines robust data preprocessing, behavioral feature engineering, and decision-threshold optimization to improve predictive performance and business relevance.

---

## Objectives

* Predict customer conversion behavior.
* Handle missing values and outliers effectively.
* Engineer meaningful behavioral features from user activity.
* Optimize classification thresholds beyond the default 0.50 cutoff.
* Improve model accuracy while maintaining generalization on unseen data.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Machine Learning Pipeline

### 1. Data Cleaning & Preprocessing

* Median imputation for missing values.
* Outlier treatment using 99th percentile capping.
* Prevention of data leakage through training-set statistics.

### 2. Feature Engineering

Custom behavioral features were created to capture user engagement patterns:

* Buyer Intent Score
* Total Activity Product
* Extreme Buyer Signal
* Product Focus Ratio
* Income Tier Segmentation

These features help expose non-linear customer behaviors to the classification model.

### 3. Model Development

A regularized Logistic Regression model was trained using:

* `max_iter = 4000`
* `C = 1.5`

The model serves as a strong and interpretable baseline for conversion prediction.

### 4. Threshold Optimization

Instead of relying on the default probability threshold of **0.50**, the project performs a fine-grained threshold sweep from **0.10 to 0.90** with intervals of **0.001**.

This optimization identifies the decision boundary that maximizes overall classification accuracy.

---

## Results

| Metric            | Score  |
| ----------------- | ------ |
| Training Accuracy | 84.92% |
| Test Accuracy     | 65.13% |

### Key Outcomes

* Improved customer conversion prediction.
* Significant reduction in false positive classifications.
* Better handling of imbalanced data compared to default model settings.
* Strong balance between model performance and generalization.

---

## Author

**Kasi Viswanath Sharma**
Machine Learning | Data Analytics | Data Visualization

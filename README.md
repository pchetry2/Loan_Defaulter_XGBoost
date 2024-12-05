# Overview
Developed a Machine Learning model to predict loan defaults, achieving ~70% Recall value. Used XGBoost classifier to identify high-risk applications and key risk indicators through gain-based feature importance analysis.

**Problem Statement**
Financial institutions need accurate loan default prediction models to:

Minimize default-related losses
Enable early risk identification
Meet regulatory requirements
Optimize resource allocation
Improve portfolio risk management

**Dataset Size**: 307,511 loan applications × 121 features
**Features Categories:**

**Demographic**: age, gender, education, employment
**Financial:** Income, Loan amount, annuity
**External Scores:** Three normalized risk ratings
**Documentation:** 21 document flags
**Regional:** Population, rating, housing statistics
**Social:** Family, social circle information


**Target Variable:** Binary default status (1 = Default, 0 = Non-Default)

**Implementation Details**
Data Preprocessing
Missing Value Analysis
Train-Test Data Split (0.70-0.30)

_Feature Engineering:_
Weight of Evidence (WOE) Encoding for categorical variables

**XGBoost Configuration**

best_params = {
    'max_depth': 6,
    'min_child_weight': 1,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'colsample_bytree': 0.5,
    'scale_pos_weight': 11,
    'n_estimators': 60
}

**Feature Importance (Gain-based)**
![Screenshot 2024-12-04 at 10 46 47 PM](https://github.com/user-attachments/assets/e178a60e-4d97-4cf8-8618-1ec19e44be40)

**Model Performance**
Training Data Performance

_**Recall:**_ 0.76
_**Precision:**_ 0.21
_**F1 Score:**_ 0.33

Testing Data Performance

**Recall:**__ 0.62
**Precision:**__ 0.17
**F1 Score:**__ 0.27

**Risk Bucket Analysis**

**Highest Risk Bucket (0.701-0.97):**

**Default Rate:** 32.5%
**Capture Rate:** 40.3%
**Volume:** 21,526 applications


**Lowest Risk Bucket (0.0005-0.112):**

**Default Rate:** 0.06%
**Capture Rate:** 0.07%
**Volume:** 21,526 applications

![Screenshot 2024-12-04 at 10 44 14 PM](https://github.com/user-attachments/assets/bdf58721-fbf7-4d3e-b520-a0fa51529edb)


**Business Impact**
**Risk Assessment**

Early identification of high-risk applications
40.3% of defaults captured in highest risk segment


**Decision Support**

Probability scores for risk-based pricing
Feature importance for application review focus
Risk segmentation for portfolio management

**Operational Efficiency**

Risk assessment
Standardized evaluation criteria
Data-driven decision making

**Tools used**

Python
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn


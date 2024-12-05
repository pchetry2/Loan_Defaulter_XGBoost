# Loan_Defaulter_XGBoost:
Developed an XGBoost model to classify Loan Defaulters with ~70% Recall value and identified top Risk as well as Capacity-based profiling features impacting write-offs by estimating Gain-based Feature Importance. 


**Problem Statement**
Financial institutions face significant challenges in assessing credit risk and predicting loan defaults. With default rates affecting both profitability and regulatory compliance, there's a critical need for accurate prediction models.

**Business Challenge**
High costs associated with loan defaults
Need for early risk identification
Regulatory requirements for risk assessment
Resource allocation for debt collection
Portfolio risk management

**Technical Objectives**

Build ML model to predict credit default probability
Identify key risk indicators
Create actionable risk scoring system
Enable real-time risk assessment
Develop interpretable model insights

**Data Context**

**Dataset Size**: 307,511 loan applications × 121 features
**Features Categories:**

**Demographic**: age, gender, education, employment
**Financial:** Income, Loan amount, annuity
**External Scores:** Three normalized risk ratings
**Documentation:** 21 document flags
**Regional:** Population, rating, housing statistics
**Social:** Family, social circle information


**Target Variable:** Binary default status (1 = Default, 0 = Non-Default)

**Success Metrics**

**Model Performance:**

Target Recall > 70%
Risk Segmentation Effectiveness
Feature Importance Insights
Default Capture Rate in Top Risk Buckets

![Screenshot 2024-12-04 at 10 44 14 PM](https://github.com/user-attachments/assets/5e652de8-719b-4c64-ad89-b8c5439d30e2)



**Implementation Details**
Data Preprocessing
Missing Value Analysis
Train-Test Data Split
Feature Engineering:

Weight of Evidence (WOE) Encoding

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


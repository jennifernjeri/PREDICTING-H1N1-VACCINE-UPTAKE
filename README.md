# PREDICTING-H1N1-VACCINE-UPTAKE
Data Science Class Phase 3 Project on Predictive Modelling Techniques

Project Done By: Jennifer Njeri

![image](https://github.com/jennifernjeri/H1N1-and-Seasonal-Flu-Vaccines/assets/25104993/1f3daeb4-4255-4374-a822-415c174b52f5)

# INTRODUCTION
The data for this project is sourced from the National 2009 H1N1 Flu Survey conducted in the United States following the Influenza outbreak of 2009. The datasets can be found on the  Driven Data Website.

The dataset primarily consists of categorical variables with binary and numerical values. Additionally, certain columns contain coded data.

# Project Objectives

Ideal Predictive Model: Develop a robust predictive model capable of estimating the likelihood of H1N1 vaccine uptake for individuals.
Feature Importance: Identify and prioritize the key features that contribute to the decision-making process regarding H1N1 vaccination.
Recommendations: Provide actionable insights to health authorities and policymakers to enhance targeted vaccination strategies.

# MODEL SUCCESS CRITERIA

In the context of predicting vaccine intake, capturing as many true positive cases (individuals taking the vaccine) is crucial.

Recall, measuring the effectiveness of classification, is well-suited for this purpose. Identifying the characteristics of vaccine uptake informs targeted campaigns, allowing for efficient resource allocation and improved vaccination within specific demographics.

F1 Score, as a harmonic mean of precision and recall, ensures a balanced trade-off, sensitive to both false positives and false negatives. This aligns with the objectives of the prediction task.

Additionally, I'll use AUC-ROC to gauge the overall performance of the model.

Metrics:

AUC-ROC score of 85% and above
Balance Recall considering the target variable is heavily imbalanced. 50% and above.
80% and above accuracy.
F1 Score of 50% and above

# DISTRIBUTION OF H1N1 VACCINE COMPARED TO SEASONAL VACCINE

![image](https://github.com/jennifernjeri/H1N1-and-Seasonal-Flu-Vaccines/assets/25104993/a72288d1-a51d-4ecd-9549-55b1b569600e)

More individuals are opting for Seasonal Flu Vaccine over the H1N1 Vaccine.

# CORRELATION BETWEEN VARIABLES
![image](https://github.com/jennifernjeri/H1N1-and-Seasonal-Flu-Vaccines/assets/25104993/37a15d22-3da6-4939-b0d9-dcf258a41aea)

The correlations observed against the H1N1 Vaccine are within moderate levels, suggesting potential compatibility for regression modeling without encountering multicollinearity issues. Lasso and Ridge Regularization shall solve for any multicollinearity.

# CLASS IMBALANCE
![image](https://github.com/jennifernjeri/H1N1-and-Seasonal-Flu-Vaccines/assets/25104993/117b733d-9be1-484a-9dcd-2deea5ea7b8b)

There is a class imbalance in our target variable

# Solve for class imbalance
![image](https://github.com/jennifernjeri/H1N1-and-Seasonal-Flu-Vaccines/assets/25104993/5d609061-0aca-4459-b283-3fb861bdabe6)

Class Imbalance Solved using SMOTE

# BASELINE MODEL
AUC-ROC SCORES
LogisticRegression - AUC-ROC: 0.8769
DecisionTreeClassifier - AUC-ROC: 0.6887
RandomForestClassifier - AUC-ROC: 0.8765
KNeighborsClassifier - AUC-ROC: 0.7682

Classification scores
LogisticRegression - Accuracy: 0.8535, Recall: 0.5672, F1 Score: 0.6276, AUC-ROC: 0.8769
DecisionTreeClassifier - Accuracy: 0.7830, Recall: 0.5217, F1 Score: 0.5114, AUC-ROC: 0.6887
RandomForestClassifier - Accuracy: 0.8478, Recall: 0.4825, F1 Score: 0.5798, AUC-ROC: 0.8765
KNeighborsClassifier - Accuracy: 0.8111, Recall: 0.4095, F1 Score: 0.4856, AUC-ROC: 0.7682

![image](https://github.com/jennifernjeri/H1N1-and-Seasonal-Flu-Vaccines/assets/25104993/836dc971-a273-4576-875f-e05dcfb6952a)

Considering our objectives, the models which prioritize Recall, F1 Score, and AUC-ROC are:
1. Logistic Regression has the highest Recall, F1 Score, and AUC-ROC among all models, making it a strong candidate.
2. Random Forest Classifier has a good balance of Accuracy, Recall, and F1 Score. The AUC-ROC is also high.

# Modeling with balanced (SMOTE) and Scaled (StandardScaler) data

We will use Logistic Regression and Random Forest Classifiers.
Classification Report:
Logistic Regression Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.95      0.91      3397
           1       0.72      0.51      0.60       945

    accuracy                           0.85      4342
   macro avg       0.80      0.73      0.75      4342
weighted avg       0.84      0.85      0.84      4342

Logistic Regression AUC-ROC: 0.8747618891863814

Random Forest Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.93      0.90      3397
           1       0.68      0.55      0.61       945

    accuracy                           0.85      4342
   macro avg       0.78      0.74      0.76      4342
weighted avg       0.84      0.85      0.84      4342

Random Forest AUC-ROC: 0.8723036043318646

The Logistic Regression performs slightly lower than the Random Forest model in terms of recall. Let us boost the recall scores in the Logistic regression model.
While the default recall threshold is typically set at 0.5 for binary classification to achieve a balance, the goal here is to optimize TPR performance. Thus, we will fine-tune the threshold, taking into consideration the importance of precision as well.

![image](https://github.com/jennifernjeri/H1N1-and-Seasonal-Flu-Vaccines/assets/25104993/d13256d0-0d19-4e81-81da-0ad5ab8de625)

As Recall threshold increases, the precision decreases. We will use a threshold of 0.2 to strike a balance between precision and recall.

# Hyperparameter Tuning on Logistic Regression and XGBOOST
We will use GridSearchCV to find the best hyperparameters for our Logistic Regression model, Random Forest and XGBoost model and assess their performance on the test set.

Classification Report:
Classification Report - Logistic Regression:
               precision    recall  f1-score   support

           0       0.98      0.14      0.24      3397
           1       0.24      0.99      0.39       945

    accuracy                           0.32      4342
   macro avg       0.61      0.56      0.31      4342
weighted avg       0.82      0.32      0.27      4342

Classification Report - Random Forest:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00      3397
           1       0.22      1.00      0.36       945

    accuracy                           0.22      4342
   macro avg       0.11      0.50      0.18      4342
weighted avg       0.05      0.22      0.08      4342

Classification Report - XGBoost:
               precision    recall  f1-score   support

           0       0.88      0.94      0.91      3397
           1       0.70      0.53      0.61       945

    accuracy                           0.85      4342
   macro avg       0.79      0.74      0.76      4342
weighted avg       0.84      0.85      0.84      4342

Overall Performance

Best Logistic Regression Model Metrics:
Accuracy: 0.3217
Recall: 0.9894
F1 Score: 0.3884
AUC-ROC: 0.8488
Best XGBoost Model Metrics:
Accuracy: 0.8489
Recall: 0.5344
F1 Score: 0.6062
AUC-ROC: 0.8796
Best Random Forest Model Metrics:
Accuracy: 0.2176
Recall: 1.0000
F1 Score: 0.3575
AUC-ROC: 0.4570

Observation

Logistic Regression:
High recall for class 1 (0.99), indicating it correctly identifies positive instances. Low precision for class 1 (0.24), suggesting a high number of false positives. Overall low accuracy (0.32).

Random Forest:
Perfect recall for class 1 (1.00), meaning it correctly identifies all positive instances. Low precision for class 1 (0.22), indicating a high number of false positives. Extremely low accuracy (0.22).

XGBoost:
Balanced recall (0.53) and precision (0.70) for class 1. Higher overall accuracy (0.85) compared to the other models.

Reasoning:
While Random Forest has perfect recall for class 1, its precision is very low, leading to a high number of false positives and low accuracy.
XGBoost strikes a balance between recall, precision, and accuracy. It performs well across all metrics.

# Objective One: Ideal Predictive Model
Develop a robust predictive model capable of estimating the likelihood of H1N1 vaccine uptake for individuals.
XGBoost seems to be the best-performing model among the three, considering a balance between precision, recall, and accuracy. It offers a better trade-off between correctly identifying positive instances and minimizing false positives.

# Objective Two: Identify and prioritize the key features that contribute to the decision-making process regarding H1N1 vaccination.
![image](https://github.com/jennifernjeri/H1N1-and-Seasonal-Flu-Vaccines/assets/25104993/cee2d745-7f3c-4cf2-b8e4-7a5b47aa4cfc)

Top 20 feature importance.


# Objective Three: Recommendations
1. Encourage Seasonal Vaccine Uptake: Given that seasonal_vaccine is the most important feature, public health campaigns should emphasize and promote the importance of receiving the seasonal flu vaccine.
2. Promote Doctor Recommendations: As doctor_recc_h1n1 and doctor_recc_seasonal are significant, efforts should be made to enhance communication between healthcare professionals and the public. Encourage doctors to recommend both H1N1 and seasonal flu vaccines during patient visits.
3. Address Perceived Risks: Since opinion_h1n1_risk and opinion_seas_sick_from_vacc are influential, public health messaging should address and clarify any misconceptions or concerns regarding the perceived risks associated with H1N1 and seasonal flu vaccinations.
4. Target Health Workers: The importance of health_worker as a feature suggests that targeting healthcare workers for vaccination campaigns and ensuring their high vaccination rates could positively influence the general public.
5. Effective Communication Strategies: Recognizing the impact of opinions on vaccine effectiveness (opinion_h1n1_vacc_effective and opinion_seas_vacc_effective), public health campaigns should employ clear and compelling communication strategies to convey the effectiveness of both H1N1 and seasonal flu vaccines.
6. Employment Status Considerations: The feature employment_status_Not in Labor Force is significant. Tailoring vaccination campaigns to different employment statuses and addressing barriers specific to those not in the labor force could improve overall vaccine uptake.
7. Diversity and Racial Considerations: The features race_White and race_Black suggest considering diversity and tailoring campaigns to specific racial or ethnic groups to ensure inclusivity and effectiveness.
8. Behavioral Interventions: Focusing on behavioral aspects, such as behavioral_large_gatherings and behavioral_touch_face, indicates the importance of interventions promoting preventive behaviors in high-risk situations.
9. Child Vaccination Considerations: Given that child_under_6_months is a significant feature, campaigns should address concerns and provide information about the safety and importance of vaccinating children under six months.
10. Income and Economic Considerations: Acknowledging the importance of income_poverty, addressing economic barriers and offering accessibility to free or low-cost vaccination services can contribute to increased vaccine uptake.

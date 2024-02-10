# Insurance_Claims
Predicting Insurance Claim Fraud 
Predicting Insurance Claim Fraud 

Insurance fraud is a significant problem for the insurance industry, causing significant financial losses and driving up the cost of insurance for everyone. According to the Coalition against Insurance Fraud, insurance fraud costs an estimated $80 billion a year across all lines of insurance in the United States alone. Therefore, it is crucial for insurance companies to identify and prevent fraudulent claims as early as possible to minimize losses. 

In this project, we aim to predict whether an insurance claim is fraudulent or not using machine learning algorithms. To achieve this goal, we will use the Insurance Claim Fraud Detection dataset from Kaggle. The dataset contains information on insurance claims, such as policy holder information, claim information, and payment information. We will use this data to train and test machine learning models to identify fraudulent claims. 

Data Description 

The dataset contains information on 1000 insurance claims, which were used to build a machine learning system to detect fraudulent claims. The following features are included in the dataset: 

months_as_customer: the number of months the customer has been with the insurance company. 

age: the age of the customer. 

policy_number: the unique identifier for the policy. 

policy_bind_date: the date the policy was bound. 

policy_state: the state in which the policy was issued. 

policy_csl: the combined single limit for bodily injury and property damage coverage. 

policy_deductible: the amount the customer pays out of pocket before insurance coverage begins. 

policy_annual_premium: the amount the customer pays annually for the insurance policy. 

umbrella_limit: the maximum amount of coverage the customer can receive in addition to their primary policy. 

insured_zip: the ZIP code of the customer's residence. 

insured_sex: the gender of the customer. 

insured_education_level: the highest level of education attained by the customer. 

insured_occupation: the customer's occupation. 

insured_hobbies: the customer's hobbies. 

insured_relationship: the customer's relationship to the policyholder. 

capital-gains: the amount of capital gains the customer has received. 

capital-loss: the amount of capital losses the customer has incurred. 

incident_date: the date of the incident. 

incident_type: the type of incident (e.g., collision, theft, fire, etc.). 

collision_type: the type of collision (e.g., rear-end, head-on, etc.). 

incident_severity: the severity of the incident (e.g., minor damage, major damage, total loss, etc.). 

authorities_contacted: the authorities contacted after the incident (e.g., police, fire department, etc.). 

incident_state: the state in which the incident occurred. 

incident_city: the city in which the incident occurred. 

incident_location: the location of the incident (e.g., intersection, parking lot, etc.). 

incident_hour_of_the_day: the hour of the day at which the incident occurred. 

number_of_vehicles_involved: the number of vehicles involved in the incident. 

property_damage: whether or not property damage occurred in the incident. 

bodily_injuries: the number of bodily injuries in the incident. 

witnesses: the number of witnesses to the incident. 

police_report_available: whether or not a police report is available for the incident. 

total_claim_amount: the total amount of the claim. 

injury_claim: the amount claimed for injuries. 

property_claim: the amount claimed for property damage. 

vehicle_claim: the amount claimed for vehicle damage. 

auto_make: the make of the customer's vehicle. 

auto_model: the model of the customer's vehicle. 

auto_year: the year of the customer's vehicle. 

fraud_reported: whether or not the claim was fraudulent. 

These features provide information about the customers, the policies they hold, and the incidents they were involved in. The dataset will be used to build a machine learning model to detect fraudulent claims. 

Prediction Task 

The prediction task for this project is to classify whether an insurance claim is fraudulent or not. We will use the features in the dataset to train machine learning models to predict the target variable, which is the binary indicator of fraud or not. The dataset will be split into training and test sets, where the models will be trained on the training set and evaluated on the test set. 

We will use classification algorithms, such as decision tree, random forest, and XGBoost, to predict the target variable. We will evaluate the performance of the models using standard metrics, such as accuracy, precision, recall and F1-score. 

Functionality of the System 

The machine learning model would be used to build a fraud detection system for insurance claims. The system will take in various features related to an insurance claim, such as the type of claim, the amount of the claim, the claimant's age, and other relevant factors, and use the trained model to predict whether the claim is fraudulent or not. 

For instance, the system can receive information about a new insurance claim and extract relevant features such as the claimant's age, the amount claimed, the type of claim, and other related data. The system will then apply the trained Decision Tree model to predict whether the claim is fraudulent or not. 

Based on the model's predictions, the system can take different actions, such as approving the claim, flagging the claim for further review, or rejecting the claim outright. By using such a system, insurance companies can reduce the risk of fraudulent claims, which can save them significant amounts of money in the long run. 

However, it is important to note that the system's performance may be affected by the imbalance nature of the target variable and the imputation technique used for missing data. Therefore, the system should be carefully designed, and its performance should be continuously monitored to ensure that it is providing accurate predictions. 

Real World Problem Solved by the system 

The fraud detection system built using the machine learning model can be applied to solve several real-world problems in the insurance industry. Some of these problems include: 

Reducing financial losses due to fraudulent claims: The system can help insurance companies to identify fraudulent claims and prevent them from paying out money to fraudsters. This can help to reduce financial losses and protect the company's bottom line. 

Improving operational efficiency: Fraudulent claims can be time-consuming and expensive to investigate, which can slow down the claims process and increase operational costs. By using a fraud detection system, insurance companies can identify fraudulent claims early and streamline the claims process, improving operational efficiency. 

Enhancing customer satisfaction: Fraudulent claims can delay the processing of legitimate claims, which can lead to customer dissatisfaction. By detecting and preventing fraudulent claims, insurance companies can ensure that legitimate claims are processed quickly and efficiently, improving customer satisfaction. 

Mitigating reputational risks: Insurance companies that are unable to detect and prevent fraudulent claims can damage their reputation, leading to a loss of customer trust and loyalty. By implementing a fraud detection system, insurance companies can mitigate reputational risks and maintain their brand reputation. 

Compliance with regulatory requirements: Many regulatory bodies require insurance companies to have effective fraud detection and prevention measures in place. By implementing a fraud detection system, insurance companies can ensure that they are compliant with these regulations and avoid penalties or legal issues. 

As a summary, the fraud detection system built using the machine learning model can help insurance companies to address several real-world problems, including financial losses, operational inefficiencies, customer dissatisfaction, reputational risks, and compliance issues. 

Algorithm Selection 

The selection of an appropriate algorithm for the prediction models used in building this system involved a systematic process that aimed to evaluate the performance of several machine learning algorithms on the dataset. The goal was to select the algorithm that provided the best predictive performance, as measured by the accuracy, precision, recall, and F1-score. 

The first step in the algorithm selection process was to split the dataset into training and testing sets. The training set was used to fit the model, and the testing set was used to evaluate the model's performance. To ensure that the model's performance was not overestimated, a 5-fold cross-validation scheme was employed, whereby the dataset was divided into 5 equal parts, and each part was used as a validation set once while the other parts were used to train the model. 

Next, several machine learning algorithms were selected based on their suitability for binary classification problems. These algorithms were Decision Tree, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest, AdaBoost, XGBoost, and Gradient Boosting. 

For each algorithm, the model was trained on the training data and evaluated on the testing data using the cross-validation scheme. The performance of the model was evaluated based on the accuracy, precision, recall, and F1-score. The algorithm that provided the best performance was selected for further tuning. 

Once the best-performing algorithm was identified, its hyperparameters were tuned to further improve its performance. Hyperparameters are parameters that are not learned during the training process but are set before training the model. The hyperparameters that were tuned included the maximum depth of the decision tree, the number of neighbors for KNN, the number of estimators for the ensemble models, and the learning rate for the gradient boosting models. 

After tuning the hyperparameters, the final model was trained on the entire dataset and used to make predictions on new data. 

Finally, The algorithm that provided the best performance was selected for further tuning, and its hyperparameters were adjusted to improve its performance further. The final model was trained on the entire dataset and used to make predictions on new data. 

Measurement of Success 

The measurement of success of the fraud detection system can be evaluated using various metrics that reflect the system's performance in detecting fraudulent insurance claims accurately. The following are some of the commonly used metrics to measure the success of a binary classification model like this: 

Accuracy: It measures the proportion of correctly classified instances over the total number of instances. It is a simple and intuitive metric but can be misleading in the presence of imbalanced classes. 

Precision: It is the proportion of true positive instances over all positive instances predicted by the model. It measures how well the model can predict fraud cases without falsely flagging genuine cases. 

Recall: It is the proportion of true positive instances over all positive instances in the dataset. It measures how well the model can detect fraud cases in the dataset. 

F1-score: It is a weighted harmonic mean of precision and recall. It combines both metrics to provide an overall measure of the model's accuracy in detecting fraud cases. 

To measure the system's success, a combination of these metrics can be used, depending on the business requirements and the cost of false negatives and false positives. For instance, if the cost of false negatives (missing fraudulent claims) is high, then recall should be given more importance while optimizing the model. On the other hand, if the cost of false positives (flagging genuine claims as fraudulent) is high, then precision should be given more importance while optimizing the model. 

In this case, the system's success can be measured by comparing the accuracy, precision, recall, and F1-score of the different machine learning algorithms used to build the system. The algorithm that provides the highest overall score on these metrics would be considered the best algorithm for the task of fraud detection in the given insurance claim dataset. Additionally, the system's performance can also be evaluated by comparing the metrics of the final system with the baseline metrics and assessing the improvement in the model's performance. 

Result and discussion 

Exploratory Data Analysis 

The dataset used for this project was obtained from Kaggle and contains information on insurance claims, including policy and claim details. The dataset consists of 39 variables, including categorical and continuous features. The target variable is a binary indicator of whether the claim is fraudulent or not. 

During the exploratory data analysis, it was observed that there were missing observations in some of the categorical features such as Collision type, Property damage, and Police report available. These missing values were imputed using the mode imputation technique since the features containing missing observations are categorical in nature. The correlation heatmap was also obtained and used to identify any potential correlations between features. Additionally, some variables that were not necessary for the prediction as well as those with high correlations were dropped, resulting in a final dataset with 25 features, including the target variable. 

Data Preprocessing 

The data preprocessing steps for the insurance claim fraud detection system involved the following: 

Missing value imputation: The missing values in the categorical features were imputed using the most frequently occurring category in the respective features. This approach is known as mode imputation, and it is a common method for handling missing values in categorical features. 

Correlation analysis: The dataset contained some highly correlated features, which were treated by removing one of the correlated features. This step helps in reducing the dimensionality of the dataset and avoiding multicollinearity, which can affect the model's performance. 

Feature selection: After removing some correlated and unnecessary features, there remained 25 features, including the target variable. This step helps in reducing the dimensionality of the dataset and avoiding overfitting. 

Outlier treatment: There were outliers in the continuous features, which were treated using a standard scaler. Standardization involves scaling the features to have a mean of 0 and a standard deviation of 1. This approach is beneficial when there are outliers in the dataset as it reduces their impact on the model's performance. 

One-hot encoding: The categorical variable was dummified using one-hot encoding, which creates binary features for each category in the variable. This approach helps in converting categorical data into a format that can be easily understood by machine learning algorithms. 

Model Selection and Performance 

Nine different machine learning models were trained on the preprocessed dataset, including Decision Tree, SVC, KNN, Random Forest, Extra Trees, Ada Boost, XGBoost, Stochastic Gradient Boosting (SGB), and Gradient Boosting. The accuracy of the models was evaluated using cross-validation and the results are as follows: 

Test Accuracy of the trained model 

Model 

Accuracy 

Precision 

Recall 

F1-Score 

Decision Tree 

The table shows the evaluation metrics of several models for detecting insurance claim fraud using the Kaggle insurance claim dataset. The models' performances are measured in terms of accuracy, precision, recall, and F1-score. 

Accuracy: measures how often the model makes the correct prediction. The model with the highest accuracy is Decision Tree with 0.772. 

Precision: measures the proportion of true positive predictions out of all positive predictions. Precision indicates how confident we can be that the model is correct when it makes a positive prediction. KNN and SVC, with a precision of 0.38, have the lowest precision, indicating that they make many false positive predictions. 

Recall: measures the proportion of true positives that were correctly predicted by the model out of all actual positives. Recall indicates how well the model identifies true positive cases. Decision Tree has the highest recall with 0.71. 

F1-Score: measures the balance between precision and recall. A high F1-score indicates that the model is performing well in both precision and recall. Decision Tree has the highest F1-Score with 0.7. 

Given that the target variable (insurance claim fraud) is usually rare, it is highly likely that the dataset has an imbalance nature. It is therefore crucial to consider the limitations of the system's performance. It appears that the Decision Tree model has the best overall performance in this case, although other models, such as Random Forest and Extra Trees, also performed well. However, the performance of the other models is relatively poor, with several models having precision, recall, and F1-score below 0.5, indicating that they may not be suitable for the fraud detection task. 

Furthermore, the imputation of missing data using mode may introduce some bias into the system. For example, if the imputed values occur frequently in the data, the system may not be able to capture the real patterns in the data, which may reduce the model's accuracy. 

In summary, the Decision Tree model appears to be the best performer in detecting insurance claim fraud. However, the imbalance nature of the target variable may limit the system's overall performance. Additionally, imputing missing data using the mode may also affect the system's accuracy. 

 
Discussion and Conclusion 

The project's objective was to predict whether an insurance claim is fraudulent or not using machine learning algorithms. Through exploratory data analysis and data preprocessing, a dataset of 25 features was prepared, including the target variable. Nine different machine learning models were trained on the preprocessed dataset, and the Decision Tree and SVC models were found to have the highest accuracy. 

However, it is important to note that the performance of the models may be affected by the class imbalance in the target variable, with fraudulent claims being a minority. Therefore, oversampling or undersampling techniques could be employed to address this issue. However, as effective as Oversampling/undersampling may be, it causes sampling bias, since the randomness (fairness to all cases) is violated. Another possible challenge of the model may be the missing value imputation which increases the bias of the model. In many cases, the imputed value may be different from the true but missed value.  

In conclusion, the ability to accurately predict insurance fraud can have significant impacts on reducing costs and improving the overall effectiveness of insurance systems. The machine learning models developed in this project can be further improved through fine-tuning and the incorporation of additional data sources. 

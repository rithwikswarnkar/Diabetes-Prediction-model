# Diabetes Prediction: Data Preprocessing and Model Training

This repository contains a Python code pipeline for predicting diabetes using a dataset and several machine learning techniques, including data preprocessing, model training, hyperparameter tuning, and evaluation.

## Overview

The dataset contains health-related features and a target variable indicating whether an individual has diabetes (1) or not (0). The goal is to build a robust machine learning pipeline that handles missing values, class imbalance, feature engineering, and multiple model evaluation techniques.

### Key Steps in the Pipeline:

1. **Data Preprocessing**: Includes handling missing values, feature scaling, and removing invalid or missing data.
2. **Class Imbalance Handling**: Using SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class.
3. **Model Training and Hyperparameter Tuning**: An XGBoost classifier is trained with hyperparameter tuning using RandomizedSearchCV.
4. **Ensemble Learning**: Combining multiple classifiers (Random Forest, XGBoost, and Logistic Regression) using soft voting.
5. **Advanced Resampling Techniques**: Exploring Borderline-SMOTE and SMOTEENN to improve class balance and model performance.
6. **Feature Engineering**: Creating new features (interaction terms and log transformations) to improve model prediction.

---

## Requirements

Make sure to install the following Python packages:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib
```

---

## Data Preprocessing

1. **Loading Data**: The dataset is loaded from a CSV file using `pandas`.

   ```python
   df = pd.read_csv("path_to_data/diabetes.csv")
   ```

2. **Handling Missing Values**: Columns with a value of `0` are considered invalid and replaced with `NaN`. Missing values are then imputed using the median value of each column.

3. **Target Distribution**: The distribution of the target variable (`Outcome`) is analyzed to check for class imbalance.

4. **Dropping Irrelevant Features**: In this case, the `Insulin` column is dropped due to its high proportion of invalid (zero) values.

---

## Class Imbalance Handling

To handle class imbalance, the training dataset is resampled using the **SMOTE** algorithm to generate synthetic samples for the minority class. The resampling is performed only on the training set:

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

The class distributions before and after resampling are printed to verify balance.

---

## Model Training and Hyperparameter Tuning

1. **XGBoost Classifier**: An XGBoost classifier is initialized and tuned using `RandomizedSearchCV`. The hyperparameter grid includes various combinations of parameters such as `n_estimators`, `max_depth`, `learning_rate`, and others.

2. **Randomized Search**: The best hyperparameters are selected using 3-fold cross-validation and F1-score as the evaluation metric.

   ```python
   random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=30, scoring='f1', cv=3, n_jobs=-1)
   random_search.fit(X_train_resampled, y_train_resampled)
   ```

3. **Model Evaluation**: The model is evaluated on the validation set, and the classification report (precision, recall, F1-score) is displayed.

---

## Ensemble Learning

An ensemble of three classifiers — RandomForest, XGBoost, and Logistic Regression — is created using a **VotingClassifier** with *soft voting*. This combines the predicted probabilities of each classifier to make the final prediction.

```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
rf = RandomForestClassifier(n_estimators=200, max_depth=7)
xgb = xgb.XGBClassifier(**random_search.best_params_, random_state=42)
lr = LogisticRegression(max_iter=1000)
ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)], voting='soft')
ensemble.fit(X_train_resampled, y_train_resampled)
```

---

## Advanced Resampling Techniques

Two advanced resampling techniques are applied to the training data:

1. **Borderline-SMOTE**: Focuses on generating synthetic samples near the decision boundary of the minority class.

2. **SMOTEENN**: Combines SMOTE and ENN (Edited Nearest Neighbors) to remove noisy samples from the dataset.

```python
from imblearn.over_sampling import BorderlineSMOTE
bl_smote = BorderlineSMOTE(random_state=42)
X_train_bl, y_train_bl = bl_smote.fit_resample(X_train, y_train)
```

---

## Feature Engineering

1. **Interaction Terms**: New features such as `BMI_Age` (BMI \* Age) and `Glucose_BP` (Glucose \* BloodPressure) are created.

2. **Log Transformation**: Log transformations are applied to `Glucose` and `BMI` to reduce skewness.

```python
def add_engineered_features(df):
    df['BMI_Age'] = df['BMI'] * df['Age']
    df['Glucose_BP'] = df['Glucose'] * df['BloodPressure']
    df['Log_Glucose'] = np.log1p(df['Glucose'])
    df['Log_BMI'] = np.log1p(df['BMI'])
    return df
```

These new features are added to the training, validation, and test datasets before model retraining.

---

## Model Evaluation & Metrics

The final model is evaluated using various metrics, including:

1. **Classification Report**: Precision, recall, and F1-score.
2. **Confusion Matrix**: Visualized to understand true positive, true negative, false positive, and false negative rates.
3. **Feature Importance**: Using the XGBoost model, the top features based on `gain` are visualized.

```python
import matplotlib.pyplot as plt
import xgboost as xgb

# Feature importance visualization
xgb.plot_importance(best_xgb, max_num_features=10, importance_type='gain', show_values=True)
```

---

## Conclusion

The code includes all the necessary steps to preprocess data, handle class imbalance, create new features, tune models, and evaluate their performance. This comprehensive approach ensures that the machine learning model has the best chance to predict diabetes accurately.

---

## License

This code is open-source and available for use under the MIT License. Feel free to use and modify the code for your own purposes.

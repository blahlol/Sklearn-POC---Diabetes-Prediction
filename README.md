# Diabetes Prediction using Pima Indians Database - Exploratory Data Analysis, Training models based on various Classification algorithms, Model Evaluation, Model Selection, and Hyperparameter tuning.

## Overview

This repository contains a Jupyter notebook that demonstrates how to perform binary classification to predict whether a patient has diabetes using the Pima Indians Diabetes Dataset. The notebook utilizes various classification algorithms provided by the `sklearn` library, evaluates their performance, and selects the best-performing model and further improves it by hyperparameter tuning.

## Dataset

The dataset used is the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data), which includes medical predictor variables and a target variable indicating diabetes presence.

### Data Description

- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure
- **SkinThickness**: Triceps skinfold thickness
- **Insulin**: 2-Hour serum insulin
- **BMI**: Body mass index
- **DiabetesPedigreeFunction**: Diabetes pedigree function score
- **Age**: Age of the patient
- **Outcome**: Target variable (0 for non-diabetic, 1 for diabetic)

## Contents

1. **Import Libraries**: Necessary libraries for data analysis, preprocessing, and machine learning.
2. **Loading the Dataset**: Load the dataset from a specified source.
3. **Data Analysis and Preprocessing**:
   - Check data types and missing values
   - Analyze data distribution
   - Impute missing values and recheck distribution
   - Perform correlation analysis
   - Address class imbalance using SMOTE
4. **Training Classification Models**:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Naive Bayes
   - Decision Tree
   - Random Forest
   - Multi-Layer Perceptron (MLP)
5. **Model Evaluation**: Evaluate models using classification metrics.
   - Confusion Matrix
   - Accuracy
   - Precision
   - Recall  
6. **Hyperparameter Tuning**: Improve model performance using RandomizedSearchCV.

## Requirements

To run this notebook, you need to have the following Python libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `imbalanced-learn`

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn

```

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the LICENSE file for details.

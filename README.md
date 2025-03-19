# **Heart Disease Prediction Using Machine Learning**

## **Project Overview**
This project focuses on predicting the likelihood of **cardiac arrest** based on health indicators using two machine learning approaches: **Logistic Regression** and **Convolutional Neural Networks (CNN)**. The dataset used is the **Heart Disease Health Indicators Dataset (BRFSS 2015)**, which contains over 250,000 records with 22 health-related features. The goal is to build and evaluate models that can accurately classify individuals at risk of cardiac arrest.

---

## **Dataset**
The dataset includes the following key features:
- **Target Variable**: `HeartDiseaseorAttack` (renamed to `cardiac_arrest`).
- **Features**: High blood pressure, high cholesterol, BMI, smoking status, diabetes, physical activity, mental health, and more.
- **Size**: 253,680 rows and 18 columns after preprocessing.

---

## **Approach**
### **1. Logistic Regression**
- **Objective**: Binary classification to predict cardiac arrest (0 or 1).
- **Steps**:
  - Data preprocessing: Dropped irrelevant columns (`PhysActivity`, `AnyHealthcare`, `Education`, `Income`).
  - Standardized the dataset using `StandardScaler`.
  - Split the data into training and testing sets (80:20 ratio).
  - Trained a Logistic Regression model using `scikit-learn`.
  - Evaluated the model using **precision, recall, F1-score, and confusion matrix**.
- **Results**:
  - Accuracy: **91%**.
  - Precision for class 1 (cardiac arrest): **55%**.
  - Recall for class 1: **13%**.
  - F1-score for class 1: **21%**.

### **2. Convolutional Neural Networks (CNN)**
- **Objective**: Explore the use of CNNs for binary classification of cardiac arrest.
- **Steps**:
  - Preprocessed the data similarly to the Logistic Regression approach.
  - Reshaped the data to fit CNN input requirements.
  - Built a CNN model using `TensorFlow/Keras`.
  - Trained and evaluated the model using accuracy and loss metrics.
- **Results**:
  - (To be added based on CNN model performance).

---

## **Key Findings**
1. **Logistic Regression**:
   - The model achieved high accuracy (**91%**) but struggled with recall for the positive class (cardiac arrest), indicating a challenge in identifying true positives.
   - The confusion matrix revealed that the model predicted most instances as class 0 (no cardiac arrest), leading to a high false-negative rate.

2. **CNN**:
   - (To be added based on CNN results).

3. **Data Insights**:
   - **Correlation Analysis**: Features like `HighBP`, `HighChol`, and `BMI` showed moderate correlation with the target variable.
   - **Class Imbalance**: The dataset was imbalanced, with only **9.4%** of the samples belonging to the positive class (cardiac arrest).

---

## **Tools and Technologies**
- **Programming Languages**: Python.
- **Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, TensorFlow/Keras.
- **Machine Learning Models**: Logistic Regression, Convolutional Neural Networks (CNN).
- **Evaluation Metrics**: Precision, Recall, F1-score, Confusion Matrix.

---

## **How to Use This Repository**
1. **Dataset**: The dataset (`heart_disease_health_indicators_BRFSS2015.csv`) is included in the repository.
2. **Code**:
   - `logistic_regression.ipynb`: Contains the code for Logistic Regression analysis.
   - `cnn.ipynb`: Contains the code for CNN analysis.
3. **Results**:
   - Visualizations: Correlation heatmap, confusion matrix, and classification reports.
   - Model performance metrics.

---

## **Future Work**
- Address class imbalance using techniques like **SMOTE** or **class weighting**.
- Experiment with other models like **Random Forest**, **XGBoost**, or **LSTM** for time-series analysis.
- Deploy the model as a web application using **Streamlit** or **Flask**.

---

## **Contributors**
- Usama Imtiaz

# 🧠 Stroke Prediction Analysis

## 📋 Project Overview
This project aims to predict the likelihood of a stroke in patients using a dataset of medical records. By leveraging **machine learning** techniques, the model identifies individuals at high risk of stroke based on their health and demographic data. This early prediction can help in timely interventions and improve patient outcomes.

The project includes:
- **Data cleaning and preprocessing** to ensure the dataset is ready for model training.
- **Exploratory data analysis (EDA)** to uncover trends and patterns in the data.
- **Machine learning model training and evaluation** to build a predictive model.
- **Feature engineering** to enhance the model’s accuracy.

## 💡 Key Questions Answered
1. **What are the key factors that contribute to stroke risk?**
   - By analyzing patient data such as age, hypertension, glucose levels, and BMI, the project highlights the most influential features driving stroke risk.

2. **How accurately can we predict stroke risk?**
   - Multiple machine learning models, including logistic regression, decision trees, and neural networks, are evaluated to determine the best predictive approach.

3. **How can we effectively preprocess and clean medical data?**
   - Detailed steps are taken to clean, scale, and encode the data for effective model training.

## 📁 Project Structure

```plaintext
├── Stroke_Prediction_Analysis.ipynb                 # Main notebook with model building and evaluation
├── Stroke_Prediction_Data_Cleaning_and_Analysis.ipynb # Data cleaning and exploratory data analysis
├── data_processing.py                                # Script for loading and preprocessing the data
├── model_training.py                                 # Script for training and evaluating machine learning models
├── utils.py                                          # Utility functions for data splitting and scaling
├── test_data_processing.py                           # Unit tests for data processing

## ⚙️ Technical Details

### 1. **Data Cleaning and Preprocessing** (`Stroke_Prediction_Data_Cleaning_and_Analysis.ipynb`)
   - **Data Loading**: The dataset is loaded from a CSV file using `pandas`.
   - **Handling Missing Values**: Missing values are handled by using strategies such as mean/mode imputation for continuous variables and one-hot encoding for categorical variables.
   - **Feature Transformation**:
     - **Categorical Variables**: Categorical variables are one-hot encoded using `pandas.get_dummies()` to convert them into numerical values.
     - **Scaling**: Numerical variables are scaled using `MinMaxScaler` from `scikit-learn` to normalize the feature range and improve model performance.
   - **Exploratory Data Analysis (EDA)**: 
     - Histograms, box plots, and correlation matrices are used to visualize distributions and relationships between features.
     - Insights are drawn regarding feature importance, data balance, and outliers.

### 2. **Model Training** (`Stroke_Prediction_Analysis.ipynb`)
   - The following machine learning models are implemented:
     - **Logistic Regression**: A simple linear model for binary classification.
     - **Decision Trees**: A tree-based model that splits data based on feature values.
     - **Random Forest**: An ensemble model that builds multiple decision trees to reduce variance and improve accuracy.
     - **Neural Networks**: A deep learning model (if applicable) built using **TensorFlow/Keras** to capture non-linear relationships in the data.
   - **Model Evaluation Metrics**:
     - **Accuracy**: Percentage of correct predictions.
     - **Precision and Recall**: For evaluating the model’s ability to handle class imbalance and minimize false positives/negatives.
     - **F1-score**: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.
     - **AUC-ROC**: A performance metric that measures the trade-off between sensitivity and specificity, particularly useful for binary classification problems.

### 3. **Utility Functions** (`utils.py`)
   - **Data Splitting**: A function to split the dataset into training and testing sets using `train_test_split` from `scikit-learn`. This ensures that the model is evaluated on unseen data to prevent overfitting.
   - **Scaling**: Numerical features are scaled using MinMaxScaler, normalizing values to a 0-1 range, improving model stability and convergence.

### 4. **Testing** (`test_data_processing.py`)
   - Unit tests are implemented using Python's `unittest` framework to ensure the correctness of data loading, cleaning, and preprocessing steps.
   - **Test Cases**:
     - **Data Loading**: Ensures that the dataset is correctly loaded as a `pandas DataFrame` and is not empty.
     - **Preprocessing**: Validates that preprocessing functions, such as one-hot encoding and scaling, are applied correctly.
   - These tests ensure that the data pipeline is functional and produces the correct outputs before model training.

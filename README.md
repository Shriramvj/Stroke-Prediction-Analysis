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

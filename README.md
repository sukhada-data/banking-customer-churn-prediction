# Banking Customer Churn Prediction

## Project Overview
This project predicts whether a bank customer is likely to leave the bank using a Machine Learning classification model.

The application allows users to enter customer details and receive a prediction of whether the customer is likely to churn or remain with the bank.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Model
- Algorithm: Decision Tree Classifier
- Maximum Depth: 5
- Encoding: One-Hot Encoding

## Application Features
- Interactive customer input form
- Customer churn prediction
- Churn probability
- Retention probability

## Project Files
- `customer_churn_train_model.py` – Model training script
- `customer_churn_app.py` – Streamlit application
- `churn_model.joblib` – Saved model and encoder
- `requirements.txt` – Required Python libraries
- `.gitignore` – Files excluded from GitHub

## Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run customer_churn_app.py
```

## Author
Sukhada Kulkarni
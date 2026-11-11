import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib

# filepath = r'D:\ML_Datasets\Deployment\1_Streamlit\4_ML_Model\4_banking_customer_churn'
# filename = filepath + r'\churn_model.joblib'

# artifacts = joblib.load(filename)
# loaded_model = artifacts['model']
# loaded_encoder = artifacts['encoder']
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(BASE_DIR, "churn_model.joblib")

artifacts = joblib.load(filename)

loaded_model = artifacts['model']
loaded_encoder = artifacts['encoder']

# customer = pd.read_csv(r'D:\ML_Datasets\16_Decision_Tree\Banking_Customer_Churn.csv')

# customer.drop(columns = ['RowNumber', 'CustomerId', 'Surname'], inplace = True)

# new_customer = customer.loc[[7], :]
# st.write(new_customer)

categorical_features = ['Geography', 'Gender']

st.title('Banking Customer Churn Prediction')
# encoded_columns = loaded_encoder.transform(new_customer[categorical_features])
# other_columns = new_customer.drop(columns = categorical_features)
# encoded_customer = pd.concat([other_columns, encoded_columns], axis = 1)
# encoded_customer

# X = encoded_customer.drop(columns = 'Exited')
# prediction = loaded_model.predict(X)[0]
# st.write(prediction)

# st.divider()

with st.sidebar:
    entered_credit_score = st.number_input('Credit Score', min_value = 350, max_value = 850) 

    entered_gegography = st.selectbox('Geography', options = ['France', 'Germany', 'Spain'])

    entered_gender = st.radio('Gender', options = ['Male', 'Female'], horizontal = True)

    entered_age = st.number_input('Age',  min_value = 18, max_value = 92)

    entered_tenure = st.number_input('Tenure',  min_value = 0, max_value = 10)

    entered_balance = st.number_input('Balance')

    entered_no_of_products = st.number_input('No. of Products',  min_value = 1, max_value = 4)

    entered_has_credit_card = st.toggle('Has Credit Card')

    entered_is_active_member = st.toggle('Is Active Member')

    entered_salary = st.number_input('Estimated_Salary')

new_customer = pd.DataFrame(columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                                       'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                                       'EstimatedSalary'])

new_customer.loc[0] = [entered_credit_score, entered_gegography, entered_gender, entered_age,
                       entered_tenure, entered_balance, entered_no_of_products,
                       entered_has_credit_card, entered_is_active_member, entered_salary]

st.dataframe(new_customer)

categorical_features = ['Geography', 'Gender']

encoded_columns = loaded_encoder.transform(new_customer[categorical_features])
other_columns = new_customer.drop(columns = categorical_features)
encoded_customer = pd.concat([other_columns, encoded_columns], axis = 1)
encoded_customer

# with st.sidebar:
#     if st.button('Predict', type = 'primary'):
#         prediction = loaded_model.predict(encoded_customer)[0]
#         predicted_probabilities = loaded_model.predict_proba(encoded_customer)[0]
#         predicted_prob_percent = (predicted_probabilities * 100).round(1)
#         probability_churn = predicted_probabilities[1]
#         probability_not_churn = predicted_probabilities[0]


# if prediction:
#     st.error(f'Customer is likely to churn with {probability_churn:.2f} % chance')
# else:
#     st.success(f'Customer is not likely to churn with {probability_not_churn:.2f} % chance.')


# with st.sidebar:

#     if st.button('Predict', type='primary'):

#         prediction = loaded_model.predict(encoded_customer)[0]

#         predicted_probabilities = loaded_model.predict_proba(encoded_customer)[0]

#         probability_churn = predicted_probabilities[1] * 100
#         probability_not_churn = predicted_probabilities[0] * 100

#         if prediction == 1:

#             st.error(
#                 f'Customer is likely to churn '
#                 f'with {probability_churn:.2f}% probability.'
#             )

#         else:

#             st.success(
#                 f'Customer is not likely to churn '
#                 f'with {probability_not_churn:.2f}% probability.'
#             )


# Initialize prediction variable
prediction = None

# Predict button in the LEFT sidebar
with st.sidebar:

    if st.button('Predict', type='primary'):

        prediction = loaded_model.predict(encoded_customer)[0]

        predicted_probabilities = loaded_model.predict_proba(encoded_customer)[0]

        probability_churn = predicted_probabilities[1] * 100
        probability_not_churn = predicted_probabilities[0] * 100


# Display prediction on the RIGHT main page
if prediction is not None:

    st.divider()

    st.subheader("Customer Churn Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ Customer is likely to churn "
            f"with {probability_churn:.2f}% probability."
        )

    else:

        st.success(
            f"✅ Customer is not likely to churn "
            f"with {probability_not_churn:.2f}% probability."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability_churn:.2f}%"
        )

    with col2:
        st.metric(
            "Retention Probability",
            f"{probability_not_churn:.2f}%"
        )

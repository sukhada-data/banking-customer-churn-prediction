# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.pipeline import Pipeline
# import joblib

# # filepath = r'D:\ML_Datasets\Deployment\1_Streamlit\4_ML_Model\4_banking_customer_churn'
# # filename = filepath + r'\churn_model.joblib'

# # artifacts = joblib.load(filename)
# # loaded_model = artifacts['model']
# # loaded_encoder = artifacts['encoder']
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# filename = os.path.join(BASE_DIR, "churn_model.joblib")

# artifacts = joblib.load(filename)

# loaded_model = artifacts['model']
# loaded_encoder = artifacts['encoder']

# # customer = pd.read_csv(r'D:\ML_Datasets\16_Decision_Tree\Banking_Customer_Churn.csv')

# # customer.drop(columns = ['RowNumber', 'CustomerId', 'Surname'], inplace = True)

# # new_customer = customer.loc[[7], :]
# # st.write(new_customer)

# categorical_features = ['Geography', 'Gender']

# st.title('Banking Customer Churn Prediction')
# # encoded_columns = loaded_encoder.transform(new_customer[categorical_features])
# # other_columns = new_customer.drop(columns = categorical_features)
# # encoded_customer = pd.concat([other_columns, encoded_columns], axis = 1)
# # encoded_customer

# # X = encoded_customer.drop(columns = 'Exited')
# # prediction = loaded_model.predict(X)[0]
# # st.write(prediction)

# # st.divider()

# with st.sidebar:
#     entered_credit_score = st.number_input('Credit Score', min_value = 350, max_value = 850) 

#     entered_gegography = st.selectbox('Geography', options = ['France', 'Germany', 'Spain'])

#     entered_gender = st.radio('Gender', options = ['Male', 'Female'], horizontal = True)

#     entered_age = st.number_input('Age',  min_value = 18, max_value = 92)

#     entered_tenure = st.number_input('Tenure',  min_value = 0, max_value = 10)

#     entered_balance = st.number_input('Balance')

#     entered_no_of_products = st.number_input('No. of Products',  min_value = 1, max_value = 4)

#     entered_has_credit_card = st.toggle('Has Credit Card')

#     entered_is_active_member = st.toggle('Is Active Member')

#     entered_salary = st.number_input('Estimated_Salary')

# new_customer = pd.DataFrame(columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
#                                        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
#                                        'EstimatedSalary'])

# new_customer.loc[0] = [entered_credit_score, entered_gegography, entered_gender, entered_age,
#                        entered_tenure, entered_balance, entered_no_of_products,
#                        entered_has_credit_card, entered_is_active_member, entered_salary]

# st.dataframe(new_customer)

# categorical_features = ['Geography', 'Gender']

# encoded_columns = loaded_encoder.transform(new_customer[categorical_features])
# other_columns = new_customer.drop(columns = categorical_features)
# encoded_customer = pd.concat([other_columns, encoded_columns], axis = 1)
# encoded_customer

# # with st.sidebar:
# #     if st.button('Predict', type = 'primary'):
# #         prediction = loaded_model.predict(encoded_customer)[0]
# #         predicted_probabilities = loaded_model.predict_proba(encoded_customer)[0]
# #         predicted_prob_percent = (predicted_probabilities * 100).round(1)
# #         probability_churn = predicted_probabilities[1]
# #         probability_not_churn = predicted_probabilities[0]


# # if prediction:
# #     st.error(f'Customer is likely to churn with {probability_churn:.2f} % chance')
# # else:
# #     st.success(f'Customer is not likely to churn with {probability_not_churn:.2f} % chance.')


# # with st.sidebar:

# #     if st.button('Predict', type='primary'):

# #         prediction = loaded_model.predict(encoded_customer)[0]

# #         predicted_probabilities = loaded_model.predict_proba(encoded_customer)[0]

# #         probability_churn = predicted_probabilities[1] * 100
# #         probability_not_churn = predicted_probabilities[0] * 100

# #         if prediction == 1:

# #             st.error(
# #                 f'Customer is likely to churn '
# #                 f'with {probability_churn:.2f}% probability.'
# #             )

# #         else:

# #             st.success(
# #                 f'Customer is not likely to churn '
# #                 f'with {probability_not_churn:.2f}% probability.'
# #             )


# # Initialize prediction variable
# prediction = None

# # Predict button in the LEFT sidebar
# with st.sidebar:

#     if st.button('Predict', type='primary'):

#         prediction = loaded_model.predict(encoded_customer)[0]

#         predicted_probabilities = loaded_model.predict_proba(encoded_customer)[0]

#         probability_churn = predicted_probabilities[1] * 100
#         probability_not_churn = predicted_probabilities[0] * 100


# # Display prediction on the RIGHT main page
# if prediction is not None:

#     st.divider()

#     st.subheader("Customer Churn Prediction Result")

#     if prediction == 1:

#         st.error(
#             f"⚠️ Customer is likely to churn "
#             f"with {probability_churn:.2f}% probability."
#         )

#     else:

#         st.success(
#             f"✅ Customer is not likely to churn "
#             f"with {probability_not_churn:.2f}% probability."
#         )

#     col1, col2 = st.columns(2)

#     with col1:
#         st.metric(
#             "Churn Probability",
#             f"{probability_churn:.2f}%"
#         )

#     with col2:
#         st.metric(
#             "Retention Probability",
#             f"{probability_not_churn:.2f}%"
#         )

import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Banking Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FB;
}

[data-testid="stSidebar"] {
    background-color: #EAF0F7;
}

.main-title {
    font-size: 36px;
    font-weight: 750;
    color: #142D4E;
    margin-bottom: 5px;
}

.subtitle {
    color: #64748B;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-title {
    color: #142D4E;
    font-size: 23px;
    font-weight: 650;
}

.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border-left: 6px solid #0D9488;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    margin-top: 15px;
}

.churn-card {
    border-left: 6px solid #DC5252;
}

.result-heading {
    font-size: 22px;
    font-weight: 700;
    color: #142D4E;
}

.result-description {
    color: #64748B;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}

.stButton button {
    background-color: #142D4E;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 600;
    padding: 10px 20px;
    width: 100%;
}

.stButton button:hover {
    background-color: #0D9488;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD TRAINED MODEL AND ENCODER
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(BASE_DIR, "churn_model.joblib")

artifacts = joblib.load(filename)

loaded_model = artifacts['model']
loaded_encoder = artifacts['encoder']

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏦 Banking Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent customer retention analysis '
    'powered by Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# SIDEBAR - CUSTOMER INPUTS
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 👤 Customer Details")
    st.caption("Enter the customer's information below.")

    st.divider()

    st.markdown("### 📊 Financial Information")

    entered_credit_score = st.number_input(
        "Credit Score",
        min_value=350,
        max_value=850,
        value=650
    )

    entered_balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=15000.0,
        step=1000.0
    )

    entered_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    st.divider()

    st.markdown("### 👥 Personal Information")

    entered_geography = st.selectbox(
        "Geography",
        options=['France', 'Germany', 'Spain']
    )

    entered_gender = st.radio(
        "Gender",
        options=['Male', 'Female'],
        horizontal=True
    )

    entered_age = st.number_input(
        "Age",
        min_value=18,
        max_value=92,
        value=35
    )

    st.divider()

    st.markdown("### 🏦 Banking Relationship")

    entered_tenure = st.number_input(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5
    )

    entered_no_of_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2
    )

    entered_has_credit_card = st.toggle(
        "Has Credit Card"
    )

    entered_is_active_member = st.toggle(
        "Active Member"
    )

    st.divider()

    predict_button = st.button(
        "🔍 Predict Customer Churn",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# CREATE CUSTOMER DATAFRAME
# --------------------------------------------------

new_customer = pd.DataFrame(
    columns=[
        'CreditScore',
        'Geography',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary'
    ]
)

new_customer.loc[0] = [
    entered_credit_score,
    entered_geography,
    entered_gender,
    entered_age,
    entered_tenure,
    entered_balance,
    entered_no_of_products,
    entered_has_credit_card,
    entered_is_active_member,
    entered_salary
]

# --------------------------------------------------
# ENCODING - PRESERVE EXISTING MODEL LOGIC
# --------------------------------------------------

categorical_features = ['Geography', 'Gender']

encoded_columns = loaded_encoder.transform(
    new_customer[categorical_features]
)

other_columns = new_customer.drop(
    columns=categorical_features
)

encoded_customer = pd.concat(
    [other_columns, encoded_columns],
    axis=1
)

# --------------------------------------------------
# CUSTOMER OVERVIEW
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Customer Overview</div>',
    unsafe_allow_html=True
)

st.caption("Review the customer's profile before running the prediction.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Credit Score", entered_credit_score)

with col2:
    st.metric("Customer Age", entered_age)

with col3:
    st.metric("Tenure", f"{entered_tenure} Years")

with col4:
    st.metric("Account Balance", f"₹{entered_balance:,.0f}")

with st.expander("View Complete Customer Information"):

    st.dataframe(
        new_customer,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🎯 Churn Prediction Analysis</div>',
    unsafe_allow_html=True
)

if predict_button:

    prediction = loaded_model.predict(encoded_customer)[0]

    predicted_probabilities = loaded_model.predict_proba(
        encoded_customer
    )[0]

    probability_churn = predicted_probabilities[1] * 100
    probability_not_churn = predicted_probabilities[0] * 100

    if prediction == 1:

        st.markdown("""
        <div class="result-card churn-card">
            <div class="result-heading">⚠️ High Churn Risk</div>
            <div class="result-description">
                This customer is predicted to leave the bank.
                Consider taking proactive customer retention measures.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card">
            <div class="result-heading">✅ Customer Likely to Stay</div>
            <div class="result-description">
                This customer is predicted to remain with the bank.
                Continue providing a positive banking experience.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

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

    st.write("### Churn Risk Indicator")

    st.progress(
        int(round(probability_churn)),
        text=f"Churn Probability: {probability_churn:.2f}%"
    )

else:

    st.info(
        "👈 Enter the customer details in the sidebar "
        "and click **Predict Customer Churn** to view the result."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "<div style='text-align:center; color:#64748B; "
    "font-size:13px;'>"
    "Banking Customer Churn Prediction | "
    "Machine Learning Application"
    "</div>",
    unsafe_allow_html=True
)

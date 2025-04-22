import pandas as pd
from Inference import LoanInference

inference = LoanInference(
    model_path='xgb_model.pkl',
    scaler_path='scaler.pkl',
    columns_path='columns.pkl'
) 

# Streamlit UI untuk input pengguna
st.title("Loan Application Prediction")
st.write("Masukkan data pemohon untuk prediksi aplikasi pinjaman")

# Form input untuk data pemohon
person_gender = st.selectbox("Gender", ["Male", "Female"])
person_income = st.number_input("Income (in USD)", min_value=0)
person_education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
loan_intent = st.selectbox("Loan Intent", ["Personal", "Home Improvement", "Debt Consolidation", "Other"])
person_home_ownership = st.selectbox("Home Ownership", ["Own", "Rent", "Mortgage"])
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", [0, 1])

# Membuat dictionary dari inputan
input_data = {
    "person_gender": person_gender,
    "person_income": person_income,
    "person_education": person_education,
    "loan_intent": loan_intent,
    "person_home_ownership": person_home_ownership,
    "previous_loan_defaults_on_file": previous_loan_defaults_on_file
}

# Prediksi berdasarkan input pengguna
if st.button("Predict Loan Status"):
    pred, prob = inference.predict(input_data)
    st.write(f"Prediction: {'Approved' if pred == 1 else 'Rejected'}")
    st.write(f"Probability of approval: {prob:.2f}")

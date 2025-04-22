import streamlit as st
import pandas as pd
import joblib
from Inference import LoanXGBoostModelInference

# Muat model inference
model_inference = LoanXGBoostModelInference(
    model_path='xgb_model.pkl',
    scaler_path='scaler.pkl',
    columns_path='columns.pkl'
)

# streamlit 
st.title("Aplikasi Prediksi Pinjaman ")

# Input form
person_age = st.number_input("Usia", min_value=18, max_value=100, step=1)
person_gender = st.selectbox("Gender", ['Laki-laki', 'Perempuan'])
person_education = st.selectbox("Tingkat Pendidikan", ['S1', 'S2', 'S3', 'Diploma', 'Lainnya'])
person_income = st.number_input("Pendapatan Tahunan (Rp)", min_value=1000000, step=100000)
person_emp_exp = st.number_input("Pengalaman Bekerja (Tahun)", min_value=0, step=1)
person_home_ownership = st.selectbox("Status Kepemilikan Rumah", ['Milik Sendiri', 'Sewa', 'Orang Tua'])
loan_amnt = st.number_input("Jumlah Pinjaman (Rp)", min_value=1000, step=1000)
loan_intent = st.selectbox("Tujuan Pinjaman", ['Pendidikan', 'Rumah', 'Kendaraan', 'Lainnya'])
loan_int_rate = st.number_input("Suku Bunga Pinjaman (%)", min_value=0.0, step=0.1)
loan_percent_income = st.number_input("Pinjaman sebagai Persentase Pendapatan (%)", min_value=1.0, step=0.1)
cb_person_cred_hist_length = st.number_input("Durasi Kredit (Tahun)", min_value=0, step=1)
credit_score = st.slider("Skor Kredit", min_value=300, max_value=850, step=1)
previous_loan_defaults_on_file = st.selectbox("Tunggakan Pinjaman Sebelumnya", ['Ya', 'Tidak'])

# Button untuk prediksi
if st.button("Prediksi"):
    # Convert input user menjadi DataFrame untuk diproses
    user_data = pd.DataFrame([{
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }])

    # Prediksi
    predictions, _ = model_inference.predict(user_data)
    
    # Tampilkan hasil prediksi
    if predictions[0] == 1:
        st.success("Pinjaman Disetujui ✅")
    else:
        st.error("Pinjaman Ditolak ❌")


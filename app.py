import streamlit as st
import pandas as pd
from Inference import LoanXGBoostModelInference

# Muat model inference (pastikan semua komponen preprocessing di-load)
model_inference = LoanXGBoostModelInference(
    model_path='xgb_model.pkl',
    scaler_path='scaler.pkl',
    imputer_path='imputer.pkl',
    columns_path='columns.pkl',
    encoders_path='encoders.pkl'
)

# Judul aplikasi
st.title("Aplikasi Prediksi Pinjaman 🏦")
st.subheader("Cek kelayakan pinjaman Anda dengan model prediktif.")

# Layout untuk input data
st.markdown("### Data Pribadi")
col1, col2 = st.columns(2)
with col1:
    person_age = st.number_input("Usia", min_value=18, max_value=100, step=1)
    person_gender = st.selectbox("Gender", ['Laki-laki', 'Perempuan'])
    person_education = st.selectbox("Tingkat Pendidikan", ['S1', 'S2', 'S3', 'Diploma', 'Lainnya'])
    person_income = st.number_input("Pendapatan Tahunan (Rp)", min_value=1000000, step=100000)
with col2:
    person_emp_exp = st.number_input("Pengalaman Bekerja (Tahun)", min_value=0, step=1)
    person_home_ownership = st.selectbox("Status Kepemilikan Rumah", ['Milik Sendiri', 'Sewa', 'Orang Tua'])

st.markdown("### Data Pinjaman")
col3, col4 = st.columns(2)
with col3:
    loan_amnt = st.number_input("Jumlah Pinjaman (Rp)", min_value=1000, step=1000)
    loan_intent = st.selectbox("Tujuan Pinjaman", ['Pendidikan', 'Rumah', 'Kendaraan', 'Lainnya'])
    loan_int_rate = st.number_input("Suku Bunga Pinjaman (%)", min_value=0.0, step=0.1)
with col4:
    loan_percent_income = st.number_input("Pinjaman sebagai Persentase Pendapatan (%)", min_value=1.0, step=0.1)
    cb_person_cred_hist_length = st.number_input("Durasi Kredit (Tahun)", min_value=0, step=1)
    credit_score = st.slider("Skor Kredit", min_value=300, max_value=850, step=1)

previous_loan_defaults_on_file = st.selectbox("Tunggakan Pinjaman Sebelumnya", ['Ya', 'Tidak'])

# Button untuk prediksi
if st.button("Prediksi"):
    # Konversi input user menjadi DataFrame
    user_data = pd.DataFrame([{
        'person_age': person_age,
        'person_gender': person_gender.lower(),  # Sesuaikan format dengan data preprocessing
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

    # Prediksi menggunakan model
    predictions = model_inference.predict(user_data)
    probabilities = model_inference.model.predict_proba(model_inference.preprocess_new_data(user_data))

    # Tampilkan hasil prediksi
    if predictions[0] == 1:
        st.success("Pinjaman Disetujui ✅")
        st.write(f"Confidence Score: {probabilities[0][1] * 100:.2f}%")
    else:
        st.error("Pinjaman Ditolak ❌")
        st.write(f"Confidence Score: {probabilities[0][0] * 100:.2f}%")

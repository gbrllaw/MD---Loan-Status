import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# Inisialisasi class untuk prediksi
class LoanXGBoostModelInference:
    def __init__(self, model_path, scaler_path, columns_path):
        # Memuat model, scaler, dan struktur kolom
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.columns = joblib.load(columns_path)
        
    def preprocess_new_data(self, new_data):
        # Pastikan kolom 'loan_status' ada di data baru dan tetap diprediksi
        features = new_data.copy()  # Jangan hapus kolom 'loan_status' jika tidak diperlukan

        # preprocessing
        features['person_gender'] = features['person_gender'].str.lower()
        features['person_gender'] = features['person_gender'].replace('fe male', 'female')

        # Imputasi missing value
        imputer = SimpleImputer(strategy='median')
        features['person_income'] = imputer.fit_transform(features[['person_income']])

        # Scaling numerik
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        features[numeric_cols] = self.scaler.transform(features[numeric_cols])

        # Encoding binary kategorikal
        label_cols = ['person_gender', 'previous_loan_defaults_on_file']
        for col in label_cols:
            encoder = LabelEncoder()
            features[col] = encoder.fit_transform(features[col])

        # One-hot encoding untuk kolom multikategori
        one_hot_cols = ['person_education', 'loan_intent', 'person_home_ownership']
        features = pd.get_dummies(features, columns=one_hot_cols, drop_first=True)

        # Samakan struktur kolom
        features, _ = features.align(pd.DataFrame(columns=self.columns), join='left', axis=1, fill_value=0)
        
        return features
    
    def predict(self, new_data):
        # Pisahkan target 'loan_status' dan fitur (jika ada target di data)
        if 'loan_status' in new_data.columns:
            target = new_data['loan_status']
            new_data = new_data.drop('loan_status', axis=1)  # Drop 'loan_status' untuk input fitur
        else:
            target = None

        processed_data = self.preprocess_new_data(new_data)

        # Debugging: Cek kolom processed_data
        print("Processed data columns:", processed_data.columns)

        prediction = self.model.predict(processed_data)
        return prediction, target

# Membaca dataset yang akan diprediksi
new_data = pd.read_csv('Dataset_A_loan.csv')  # Ganti dengan path yang sesuai

# Memuat model dan informasi yang diperlukan
model_inference = LoanXGBoostModelInference(
    model_path='xgb_model.pkl',
    scaler_path='scaler.pkl',
    columns_path='columns.pkl'
)

# Melakukan prediksi
predictions = model_inference.predict(new_data)

# Menampilkan hasil prediksi
print("Predictions:", predictions))

import joblib
import pandas as pd

class LoanXGBoostModelInference:
    def __init__(self, model_path, scaler_path, imputer_path, columns_path, encoders_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)
        self.columns = joblib.load(columns_path)
        self.encoders = joblib.load(encoders_path)

    def preprocess_new_data(self, new_data):
        features = new_data.copy()
        if 'loan_status' in features.columns:
            features = features.drop('loan_status', axis=1)

        # Cleaning data
        features['person_gender'] = features['person_gender'].str.lower()
        features['person_gender'] = features['person_gender'].replace('fe male', 'female')

        # Imputasi nilai kosong
        features['person_income'] = self.imputer.transform(features[['person_income']])

        # Scaling numerik
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        features.loc[:, numeric_cols] = self.scaler.transform(features[numeric_cols])

        # Encoding binary kategorikal
        label_cols = ['person_gender', 'previous_loan_defaults_on_file']
        for col in label_cols:
            encoder = self.encoders[col]
            features[col] = encoder.transform(features[col])

        # One-hot encoding untuk kolom multikategori
        one_hot_cols = ['person_education', 'loan_intent', 'person_home_ownership']
        features = pd.get_dummies(features, columns=one_hot_cols, drop_first=True)

        # Samakan struktur kolom
        features, _ = features.align(pd.DataFrame(columns=self.columns), join='left', axis=1, fill_value=0)

        return features

    def predict(self, new_data):
        processed_data = self.preprocess_new_data(new_data)
        prediction = self.model.predict(processed_data)
        return prediction

# loan_inference.py
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
import pandas as pd

class LoanInference:
    def __init__(self, model_path, scaler_path, columns_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.columns = joblib.load(columns_path)

    def preprocess_input(self, input_data):
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Lowercase gender and fix typo
        df['person_gender'] = df['person_gender'].str.lower()
        df['person_gender'] = df['person_gender'].replace('fe male', 'female')

        # Impute missing values (assume only 'person_income' as in training)
        imputer = SimpleImputer(strategy='median')
        df['person_income'] = imputer.fit_transform(df[['person_income']])

        # Scale numeric features
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Encode binary categorical
        label_cols = ['person_gender', 'previous_loan_defaults_on_file']
        for col in label_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit(df[col]).transform(df[col])

        # One-hot encode multi-categorical columns
        one_hot_cols = ['person_education', 'loan_intent', 'person_home_ownership']
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

        # Samakan struktur kolom
        df = df.reindex(columns=self.columns, fill_value=0)

        return df

    def predict(self, input_data):
        processed_data = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)[:, 1]
        return prediction[0], probability[0]

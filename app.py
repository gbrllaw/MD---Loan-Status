import pandas as pd
from Inference import LoanXGBoostModelInference

new_data = pd.read_csv('Dataset_A_loan.csv')

model_inference = LoanXGBoostModelInference(
    model_path='xgb_model.pkl',
    scaler_path='scaler.pkl',
    columns_path='columns.pkl'
)

predictions, actual_values = model_inference.predict(new_data)
print("Predictions:", predictions)

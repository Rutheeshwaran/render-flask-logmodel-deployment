import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib
import dill

# This custom function must be here because your jolib file needs it
def cap_outliers_iqr(a):
    x_col = a.copy()
    for i in x_col.columns:
        Q1 = x_col[i].quantile(0.25)
        Q3 = x_col[i].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        x_col[i] = x_col[i].clip(lower, upper)
    return x_col

app = Flask(__name__)

with open('stroke.dill', 'rb') as file: # <--- CHANGE #2: USE DILL TO LOAD
    logestic_model = dill.load(file)

# --- ADD 'id' BACK TO THE COLUMN LIST ---
COLUMN_NAMES = [
    'id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- ADD 'id' TO THE FORM PROCESSING LOGIC ---
        form_values = [
            int(request.form.get('id', '0')), # Get the id
            request.form.get('gender', ''),
            float(request.form.get('age', '0')),
            int(request.form.get('hypertension', '0')),
            int(request.form.get('heart_disease', '0')),
            request.form.get('ever_married', ''),
            request.form.get('work_type', ''),
            request.form.get('Residence_type', ''),
            float(request.form.get('avg_glucose_level', '0.0')),
            float(request.form.get('bmi', '0.0')),
            request.form.get('smoking_status', '')
        ]

        # Create the DataFrame with 11 columns
        input_df = pd.DataFrame([form_values], columns=COLUMN_NAMES)
        
        # Make the prediction
        prediction = logestic_model.predict(input_df)
        output = prediction[0]

        if output == 1:
            result_text = 'YES, there is a high risk of stroke.'
        else:
            result_text = 'NO, there is a low risk of stroke.'
        
        return render_template('index.html', predicted_val=f"Prediction: {result_text}")

    except Exception as e:
        error_message = f"An error occurred: Please check your inputs. Details: {e}"
        return render_template('index.html', predicted_val=error_message)

if __name__ == '__main__':
    app.run(debug=True)
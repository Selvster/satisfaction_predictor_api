from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

CORS(app) 


class FunctionTransformer1(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.columns] = np.log(X_[self.columns])
        return X_


class FunctionTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.columns] = np.log(X_[self.columns] + 0.1)
        return X_


# Load the pre-trained model
loaded_model = joblib.load('./model.pkl')

# Create transformers
transformer1 = FunctionTransformer1(columns=['Flight Distance'])
transformer2 = FunctionTransformer2(columns=['Departure Delay'])
# Define columns here
columns = ['ID', 'Gender', 'Age', 'Customer Type', 'Type of Travel', 'Class',
           'Flight Distance', 'Departure Delay', 'Arrival Delay',
           'Departure and Arrival Time Convenience', 'Ease of Online Booking',
           'Check-in Service', 'Online Boarding', 'Gate Location',
           'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness',
           'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
           'In-flight Entertainment', 'Baggage Handling']

# API endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract input data directly from the request
        input_data = data.get('data')

        # Preprocess input data
        input_df = pd.DataFrame([input_data], columns=columns)
        input_df_transformed = transformer1.transform(input_df)
        input_df_transformed = transformer2.transform(input_df_transformed)

        # Make predictions
        predictions = loaded_model.predict(input_df_transformed)

        return jsonify({'prediction': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(port=8000)

from flask import Flask, render_template, request
from joblib import load
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = 'model/linear_regression_model.pkl'  # Path to model in your folder
model = load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data (input values from the user)
        features = [
            float(request.form['Open']),
            float(request.form['High']),
            float(request.form['Low']),
            float(request.form['Previous']),
            float(request.form['Change']),
            float(request.form['%Change']),
            int(request.form['Year']),
            int(request.form['Month']),
            int(request.form['Day']),
            int(request.form['Day_of_week'])
        ]
        
        # Convert features to a 2D array (model expects 2D array)
        features = np.array(features).reshape(1, -1)

        # Predict the stock price using the trained model
        prediction = model.predict(features)

        # Ensure the prediction is a scalar value
        prediction_value = prediction.item()  # This converts the numpy array to a scalar

        # Return the predicted value to the user
        return render_template('index.html', prediction_text='Predicted Stock Price: {:.2f}'.format(prediction_value))

    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
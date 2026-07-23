from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
# Ensure your original model.pkl is in the same directory
try:
    model = joblib.load('model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract numerical values from the HTML form
        # Adjust data types (e.g., int vs float) based on how you trained the model
        age = float(request.form['Age'])
        gender = float(request.form['Gender'])
        region = float(request.form['Region'])
        occupation = float(request.form['Occupation'])
        income = float(request.form['Income'])

        # Create a DataFrame with the exact feature names the model expects
        input_data = pd.DataFrame(
            [[age, gender, region, occupation, income]],
            columns=['Age', 'Gender', 'Region', 'Occupation', 'Income']
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f'Prediction Result: {prediction.upper()}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: Please ensure all inputs are valid numbers. Details: {str(e)}')

if __name__ == '__main__':
    # Use 0.0.0.0 for deployment
    app.run(host='0.0.0.0', port=5000, debug=False)

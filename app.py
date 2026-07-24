from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the model
# Important: Ensure your actual binary pickle file is named 'model.pkl' and is in the same directory.
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

# Single-file HTML template with modern CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Prediction Model</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #74ebd5 0%, #9face6 100%);
            min-height: 100vh;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #333;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 400px;
            box-sizing: border-box;
        }
        h2 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-weight: 600;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
            font-size: 14px;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            box-sizing: border-box;
            font-size: 14px;
            font-family: 'Poppins', sans-serif;
            transition: border-color 0.3s ease;
        }
        input:focus {
            border-color: #74ebd5;
            outline: none;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #74ebd5 0%, #9face6 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 10px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(159, 172, 230, 0.4);
        }
        .result-card {
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            animation: fadeIn 0.5s ease;
        }
        .success { background: #e8f8f5; color: #1abc9c; border: 1px solid #1abc9c; }
        .error { background: #fdedec; color: #e74c3c; border: 1px solid #e74c3c; font-size: 14px; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div class="container">
        <h2>✨ AI Predictor</h2>
        <form method="POST">
            <div class="form-group">
                <label>Age</label>
                <input type="number" name="Age" placeholder="e.g., 30" required step="any">
            </div>
            <div class="form-group">
                <label>Gender</label>
                <input type="text" name="Gender" placeholder="Enter gender (or encoded number)" required>
            </div>
            <div class="form-group">
                <label>Region</label>
                <input type="text" name="Region" placeholder="Enter region (or encoded number)" required>
            </div>
            <div class="form-group">
                <label>Occupation</label>
                <input type="text" name="Occupation" placeholder="Enter occupation" required>
            </div>
            <div class="form-group">
                <label>Income</label>
                <input type="number" name="Income" placeholder="e.g., 50000" required step="any">
            </div>
            <button type="submit">Predict Now</button>
        </form>
        
        {% if error_msg %}
            <div class="result-card error">{{ error_msg }}</div>
        {% endif %}
        
        {% if prediction %}
            <div class="result-card success">Prediction Result: {{ prediction }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_msg = None
    
    if request.method == 'POST':
        if model is None:
            error_msg = "Model file ('model.pkl') not found. Please upload it."
            return render_template_string(HTML_TEMPLATE, error_msg=error_msg)

        try:
            # Helper function to convert inputs to floats if possible, otherwise keep as strings.
            # This handles models that were trained on Label Encoded (numeric) categorical variables.
            def parse_input(val):
                try: return float(val)
                except ValueError: return val

            # Constructing the dataframe exactly as the model expects
            input_data = {
                'Age': [parse_input(request.form['Age'])],
                'Gender': [parse_input(request.form['Gender'])],
                'Region': [parse_input(request.form['Region'])],
                'Occupation': [parse_input(request.form['Occupation'])],
                'Income': [parse_input(request.form['Income'])]
            }
            
            df = pd.DataFrame(input_data)
            
            # Predict
            pred = model.predict(df)[0]
            prediction = str(pred).upper()
            
        except Exception as e:
            error_msg = f"Prediction Error: Ensure text inputs match what the model was trained on. Details: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction, error_msg=error_msg)

if __name__ == '__main__':
    # Render binds to the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

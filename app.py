import os
from typing import Optional

import joblib          # <-- use joblib instead of pickle
import pandas as pd
import numpy as np
import io
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODELS_DIR = 'models'
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]


def load_model(model_name: Optional[str]):
    """
    model_name comes from the form and may be None.
    It should match the .pkl filename *without* the .pkl extension.
    Example: 'rf_model' -> models/rf_model.pkl
    """
    if not model_name:
        raise ValueError("No model selected. Make sure your form sends a field named 'model'.")

    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Load with joblib because the models were saved with joblib.dump
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model file '{model_path}' with joblib. "
            f"It may be corrupted or saved with an incompatible library version. "
            f"Original error: {e}"
        )


def get_available_models():
    """
    Returns names like 'rf_model', 'SVM_model' for files
    rf_model.pkl, SVM_model.pkl in the models/ folder.
    """
    models: list[str] = []
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith('.pkl'):
                model_name = file[:-4]  # strip '.pkl'
                models.append(model_name)
    return models


@app.route('/')
def index():
    models = get_available_models()
    return render_template('index.html', models=models, features=FEATURE_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form.get('model')
        model = load_model(model_name)

        features = []
        for feature in FEATURE_NAMES:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({'error': f'Missing value for {feature}'}), 400
            features.append(float(value))

        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        result = 'Malignant (1)' if prediction == 1 else 'Benign (0)'

        return jsonify({
            'prediction': int(prediction),
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        # 1. File checks
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # 2. Model
        model_name = request.form.get('model')
        model = load_model(model_name)

        # 3. Read CSV from uploaded file
        raw_bytes = file.read()
        if not raw_bytes:
            return jsonify({'error': 'Uploaded file is empty'}), 400

        try:
            text = raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = raw_bytes.decode('latin-1')

        df = pd.read_csv(io.StringIO(text))

        # 4. Validate columns
        if len(df.columns) != 30:
            return jsonify({
                'error': f'CSV must have exactly 30 columns. Found {len(df.columns)}'
            }), 400

        df.columns = FEATURE_NAMES

        # 5. Predict
        predictions = model.predict(df.values)

        results = []
        for idx, pred in enumerate(predictions):
            results.append({
                'row': idx + 1,
                'prediction': int(pred),
                'result': 'Malignant (1)' if pred == 1 else 'Benign (0)'
            })

        return jsonify({
            'predictions': results,
            'total': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
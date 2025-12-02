# Breast Cancer Prediction Web App

A simple Flask web application for predicting breast cancer diagnosis using pre-trained machine learning models on the Wisconsin Breast Cancer Diagnostic (WBCD) dataset.

## Features

- Load and use pre-trained machine learning models (.pkl files)
- Model selector to switch between different models (SVM, Random Forest, etc.)
- Manual prediction with 30 input fields for WBCD features
- CSV batch prediction for multiple samples
- Clean, simple UI with responsive design
- Real-time prediction results displaying Benign (0) or Malignant (1)

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Navigate to the project directory:
```bash
cd /tmp/cc-agent/60675932/project
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

If `pip` is not available, try:
```bash
python3 -m pip install -r requirements.txt
```

## Project Structure

```
project/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── models/                 # Directory containing .pkl model files
│   ├── SVM_model.pkl      # Support Vector Machine model
│   └── rf_model.pkl       # Random Forest model
└── templates/
    └── index.html         # Web interface
```

## Running the Application

1. Make sure you're in the project directory

2. Run the Flask app:
```bash
python3 app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## How to Use

### Manual Prediction

1. Select a model from the dropdown (SVM or Random Forest)
2. Enter values for all 30 WBCD features:
   - Mean features (10): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
   - Error features (10): same measurements but for standard error
   - Worst features (10): same measurements but for worst/largest values
3. Click "Predict" to get the result
4. The prediction will show either "Benign (0)" or "Malignant (1)"

### CSV Batch Prediction

1. Select a model from the dropdown
2. Prepare a CSV file with exactly 30 columns (one for each WBCD feature)
3. Click "Choose CSV file" and select your file
4. Click "Predict from CSV"
5. View results in a table showing row number, prediction value, and result label

## CSV File Format

Your CSV file should have 30 columns in this order:
1. mean radius
2. mean texture
3. mean perimeter
4. mean area
5. mean smoothness
6. mean compactness
7. mean concavity
8. mean concave points
9. mean symmetry
10. mean fractal dimension
11. radius error
12. texture error
13. perimeter error
14. area error
15. smoothness error
16. compactness error
17. concavity error
18. concave points error
19. symmetry error
20. fractal dimension error
21. worst radius
22. worst texture
23. worst perimeter
24. worst area
25. worst smoothness
26. worst compactness
27. worst concavity
28. worst concave points
29. worst symmetry
30. worst fractal dimension

Example CSV:
```csv
17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
```

## What This App Does

This application provides a web interface to use your pre-trained breast cancer classification models without needing to retrain them. Here's what was built:

1. **Flask Backend** (`app.py`):
   - Automatically detects and loads all .pkl models from the `models/` directory
   - Provides REST API endpoints for single predictions and batch CSV predictions
   - Handles model loading using Python's pickle module
   - Uses scikit-learn for model predictions
   - Validates input data and provides error handling

2. **Web Interface** (`templates/index.html`):
   - Clean, modern UI with responsive design
   - Model selector dropdown populated from available models
   - 30 input fields organized in a grid layout for manual predictions
   - CSV file upload with batch processing
   - Real-time result display with color coding (green for benign, red for malignant)
   - Error handling and loading states

3. **Model Integration**:
   - Loads your existing SVM and Random Forest models
   - No retraining required
   - Models are loaded on-demand when making predictions
   - Supports any scikit-learn compatible model saved as .pkl

## Troubleshooting

**Issue**: Package installation fails
- **Solution**: Make sure Python 3.7+ is installed. Try using `python3 -m pip install -r requirements.txt`

**Issue**: Models not appearing in dropdown
- **Solution**: Ensure .pkl files are in the `models/` directory and follow the naming pattern `*_model.pkl`

**Issue**: CSV prediction fails
- **Solution**: Verify your CSV has exactly 30 columns with numeric values

**Issue**: Port 5000 already in use
- **Solution**: Change the port in `app.py` by modifying the last line: `app.run(debug=True, host='0.0.0.0', port=5001)`

## Technical Details

- **Framework**: Flask 3.0.0
- **ML Library**: scikit-learn 1.3.2
- **Data Processing**: pandas 2.1.4, numpy 1.26.2
- **Model Format**: pickle (.pkl)
- **Dataset**: Wisconsin Breast Cancer Diagnostic (WBCD) with 30 features

## Notes

- This app only loads and uses pre-trained models
- Models must be scikit-learn compatible and saved using pickle
- The app expects exactly 30 features as per the WBCD dataset specification
- Predictions are binary: 0 (Benign) or 1 (Malignant)

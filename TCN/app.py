"""
Simple Flask server for pose validation predictions
Usage: python app.py
"""

from flask import Flask, request, jsonify
import os
from web_predictor import initialize_service, predict_file

app = Flask(__name__)

# Configuration
MODEL_PATH = "pose_validation_model.pth"

# Initialize model once at startup
try:
    print("Loading model...")
    initialize_service(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)


@app.route('/predict', methods=['GET'])
def predict():
    """
    Predict on a JSON file
    Usage: GET /predict?file=test.json
    """
    try:
        # Get filename from query parameter
        filename = request.args.get('file')
        
        if not filename:
            return jsonify({
                'error': 'Missing file parameter'
            }), 400
        
        # Check if file exists
        if not os.path.exists(filename):
            return jsonify({
                'error': f'File not found: {filename}'
            }), 404
        
        # Make prediction
        results = predict_file(filename)
        
        return jsonify({
            'file': filename,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Starting server on http://localhost:5050")
    print("Usage: GET /predict?file=your_file.json")
    
    app.run(debug=True, host='0.0.0.0', port=5050)
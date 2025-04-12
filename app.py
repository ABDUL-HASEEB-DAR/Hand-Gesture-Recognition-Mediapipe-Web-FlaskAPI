from flask import Flask, request, jsonify, render_template
import numpy as np
from model.keypoint_classifier import KeyPointClassifier

app = Flask(__name__)

# Initialize the classifier
classifier = KeyPointClassifier('model/keypoint_classifier.tflite')

@app.route("/")
def home():
    return render_template("index.html")  # serves your web app

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    landmarks = data.get('landmarks', None)

    if landmarks is None:
        return jsonify({'error': 'Landmarks data is required'}), 400

    if not isinstance(landmarks, list):
        return jsonify({'error': 'Landmarks should be a list'}), 400

    try:
        result = classifier(landmarks)
        # print(f"Result: {result}")

        # 👇 This is key! Ensure it's a JSON-safe native type
        if isinstance(result, (np.integer, np.floating)):
            result = result.item()
        elif isinstance(result, np.ndarray):
            result = result.tolist()

        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': f'Classifier error: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(debug=True)

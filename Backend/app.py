from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)
# Load model (model.pkl must be in same folder)
try:
    model = joblib.load('model.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

@app.route('/')
def home():
    return "Emotion Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = model.predict([text])[0]
    return jsonify({'emotion': prediction})

if __name__ == '__main__':
    app.run(debug=True)

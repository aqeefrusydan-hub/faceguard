from flask import Flask, render_template, request
from detector import load_model, predict_image  # use detector.py helpers
import numpy as np
import os
import cv2

app = Flask(__name__)

# Load model once at startup
MODEL = None
try:
    MODEL = load_model()
    print("✅ Model loaded successfully.")
except Exception as e:
    MODEL = None
    print("⚠️ Could not load model:", e)

app.secret_key = "change_this_to_a_random_secret"
app.config["UPLOAD_FOLDER"] = "static"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Home / Upload page
@app.route('/')
def index():
    return render_template('interface.html')  # Set interface.html as homepage

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        filename = file.filename
        if filename == '':
            return "Empty filename", 400

        # Save uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Use detector.py helper to make prediction
        result, confidence = predict_image(MODEL, file_path)

        return render_template('result.html', filename=filename, result=result, confidence=confidence)

    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)



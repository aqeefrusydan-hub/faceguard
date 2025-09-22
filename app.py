#<<<<<<< HEAD
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
=======
from detector import load_model, predict_image
#>>>>>>> ee23b169 (Updated result.html for image preview)
import os

app = Flask(__name__)
<<<<<<< HEAD
model = load_model("model.h5")  # Load your CNN model
=======
MODEL = None
try:
    MODEL = load_model()
    print("✅ Model loaded successfully.")
except Exception as e:
    MODEL = None
    print("⚠️ Could not load model:", e)

app.secret_key = "change_this_to_a_random_secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
>>>>>>> ee23b169 (Updated result.html for image preview)

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
        file_path = os.path.join("static", filename)
        file.save(file_path)

        ext = filename.split('.')[-1].lower()
        predictions = []

        # ----- IMAGE PROCESSING -----
        if ext in ['jpg', 'jpeg', 'png']:
            img = image.load_img(file_path, target_size=(224,224))
            img_array = image.img_to_array(img)

            # Handle grayscale or RGBA
            if img_array.shape[2] == 4:
                img_array = img_array[:,:,:3]  # Remove alpha
            elif img_array.shape[2] == 1:
                img_array = np.repeat(img_array, 3, axis=2)  # Convert grayscale to RGB

            img_array = np.expand_dims(img_array/255.0, axis=0)
            predictions.append(model.predict(img_array)[0][0])

        # ----- VIDEO PROCESSING -----
        elif ext in ['mp4', 'avi', 'mov']:
            video = cv2.VideoCapture(file_path)
            frames = []
            frame_count = 0
            max_frames = 30  # Use first 30 frames for better accuracy

            while frame_count < max_frames:
                success, frame = video.read()
                if not success:
                    break
                frame = cv2.resize(frame, (224,224))
                if frame.shape[2] == 4:
                    frame = frame[:,:,:3]  # Remove alpha
                frames.append(frame/255.0)
                frame_count += 1

            video.release()
            if len(frames) == 0:
                return "Cannot read video frames", 400

            frames = np.array(frames)
            predictions.extend(model.predict(frames).flatten())

        else:
            return "Unsupported file format", 400

        # ----- FINAL RESULT -----
        final_pred = np.mean(predictions)  # Average over frames or single image
        result = "Real" if final_pred > 0.5 else "Fake"
        confidence = float(final_pred*100 if final_pred > 0.5 else (1-final_pred)*100)

        return render_template('result.html', filename=filename, result=result, confidence=confidence)

    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)


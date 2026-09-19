import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "pneumonia_model.keras"
IMG_SIZE = (128, 128)

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        if model is None:
            error = "Model file not found. Train the model first and place pneumonia_model.keras in the project root."
        elif "file" not in request.files or request.files["file"].filename == "":
            error = "Please select a chest X-ray image."
        else:
            file = request.files["file"]
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            image = tf.keras.utils.load_img(
                filepath, target_size=IMG_SIZE, color_mode="grayscale"
            )
            image_array = tf.keras.utils.img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            probability = float(model.predict(image_array, verbose=0)[0][0])
            prediction = "Pneumonia" if probability >= 0.5 else "Normal"
            confidence = round((probability if probability >= 0.5 else 1 - probability) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)

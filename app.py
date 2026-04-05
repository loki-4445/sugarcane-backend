from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── CLASS NAMES (exact order model learned) ──
classes = [
    'Grassy_shoot',
    'Healthy',
    'Mosaic',
    'Pokkah_Boeng',
    'Red_Rot',
    'Rust',
    'Sett_Rot',
    'Yellow_Leaf',
    'smut'
]

# ── LOAD DISEASE MODEL (TensorFlow EfficientNet) ──
disease_model = tf.keras.models.load_model('sugarcane_disease_efficientnet_92.keras')
print("Disease model loaded!")

# ── LOAD SEVERITY MODEL (unchanged) ──
severity_model = tf.keras.models.load_model('Custom_Severity_DeepLab_Model.h5')
print("Severity model loaded!")

# ── DISEASE PREDICTION FUNCTION ──
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # no rescaling, EfficientNet handles it

    preds = disease_model.predict(img_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    confidence = round(float(np.max(preds[0])) * 100, 2)
    disease = classes[pred_idx]

    return disease, confidence

# ── SEVERITY PREDICTION FUNCTION (unchanged) ──
def predict_severity(image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred_mask = severity_model.predict(img, verbose=0)[0]

    diseased_pixels = np.sum(pred_mask > 0.5)
    total_pixels = pred_mask.size
    severity_percent = round((diseased_pixels / total_pixels) * 100, 2)

    return severity_percent

# ── ROUTES ──
@app.route("/")
def home():
    return "Sugarcane Disease & Severity API Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    disease, confidence = predict_image(image)
    return jsonify({"disease": disease, "confidence": confidence})

@app.route("/severity", methods=["POST"])
def severity():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    severity_percent = predict_severity(image)
    return jsonify({"severity_percent": severity_percent})

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    disease, confidence = predict_image(image)
    severity_percent = predict_severity(image)
    return jsonify({
        "disease": disease,
        "confidence": confidence,
        "severity_percent": severity_percent
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
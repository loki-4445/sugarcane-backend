from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os

# 🔴 VERY IMPORTANT — must be before tensorflow import
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ================= LOAD CLASS NAMES =================
with open("class_names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

num_classes = len(classes)

# ================= LOAD DISEASE MODEL (PYTORCH) =================
device = torch.device("cpu")

model = timm.create_model('mobilenetv3_large_100', pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

model.load_state_dict(torch.load("sugarcane_mobilenetv3.pth", map_location=device))
model.eval()

# ================= LOAD SEVERITY MODEL (TENSORFLOW) =================
severity_model = tf.keras.models.load_model(
    "Custom_Severity_DeepLab_Model.h5",
    compile=False
)

# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= DISEASE PREDICTION =================
def predict_image(image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    disease = classes[predicted.item()]
    confidence = round(confidence.item() * 100, 2)

    return disease, confidence

# ================= SEVERITY PREDICTION =================
def predict_severity(image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred_mask = severity_model.predict(img)[0]

    diseased_pixels = np.sum(pred_mask > 0.5)
    total_pixels = pred_mask.size

    severity_percent = (diseased_pixels / total_pixels) * 100
    severity_percent = round(severity_percent, 2)

    return severity_percent

# ================= ROUTES =================

@app.route("/")
def home():
    return "Sugarcane Disease & Severity API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    image = Image.open(request.files["file"]).convert("RGB")
    disease, confidence = predict_image(image)

    return jsonify({
        "disease": disease,
        "confidence": confidence
    })

@app.route("/severity", methods=["POST"])
def severity():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    image = Image.open(request.files["file"]).convert("RGB")
    severity_percent = predict_severity(image)

    return jsonify({
        "severity_percent": severity_percent
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    image = Image.open(request.files["file"]).convert("RGB")
    disease, confidence = predict_image(image)
    severity_percent = predict_severity(image)

    return jsonify({
        "disease": disease,
        "confidence": confidence,
        "severity_percent": severity_percent
    })

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

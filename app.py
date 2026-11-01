import os
import io
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request
import tensorflow as tf
from db import log_prediction, fetch_all_predictions
app = Flask(__name__)

# Configure upload directory
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------- Load Models -----------------
BRAIN_MODEL_PATH = os.path.join("models", "brain_tumor_mobilenetv2_finetuned.keras")
HEART_MODEL_PATH = os.path.join("models", "ecg_heart_disease_lstm.keras")
BLOOD_MODEL_PATH = os.path.join("models", "blood_group_cnn_model.keras")

brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH)
heart_model = tf.keras.models.load_model(HEART_MODEL_PATH)
blood_model = tf.keras.models.load_model(BLOOD_MODEL_PATH)

BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
HEART_CLASSES = [
    "Normal Beat",
    "Supraventricular Ectopic Beat",
    "Ventricular Ectopic Beat",
    "Fusion Beat",
    "Unknown / Paced Beat"
]
BLOOD_CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'Not_Valid', 'O+', 'O-']

BLOOD_COMPATIBILITY = {
    "A+":  {"donate": "A+, AB+", "receive": "A+, A-, O+, O-"},
    "O+":  {"donate": "O+, A+, B+, AB+", "receive": "O+, O-"},
    "B+":  {"donate": "B+, AB+", "receive": "B+, B-, O+, O-"},
    "AB+": {"donate": "AB+ (Universal Plasma Donor)", "receive": "Everyone (Universal Recipient)"},
    "A-":  {"donate": "A+, A-, AB+, AB-", "receive": "A-, O-"},
    "O-":  {"donate": "Everyone (Universal Donor)", "receive": "O- only"},
    "B-":  {"donate": "B+, B-, AB+, AB-", "receive": "B-, O-"},
    "AB-": {"donate": "AB+, AB-", "receive": "AB-, A-, B-, O-"},
    "Not_Valid": {"donate": "Invalid Sample / Unclear Slide", "receive": "Please re-test with a clear blood smear"}
}
@app.route("/history")
def history():
    records = fetch_all_predictions()
    return render_template("history.html", records=records)
@app.route("/")
def home():
    return render_template("index.html")

# ----------------- Module 1: Brain Tumor -----------------
@app.route("/brain_tumor", methods=["GET", "POST"])
def brain_tumor():
    prediction_result = None
    image_url = None
    breakdown = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("brain_tumor.html", error="No file uploaded")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("brain_tumor.html", error="No file selected")

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            image_url = filepath

            # Image Preprocessing (224x224)
            img = Image.open(filepath).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)

            predictions = brain_model(img_array, training=False).numpy()[0]
            pred_idx = np.argmax(predictions)
            
            prediction_result = {
                "class": BRAIN_CLASSES[pred_idx],
                "confidence": round(float(predictions[pred_idx]) * 100, 2)
            }
            breakdown = [
                {"name": name, "prob": round(float(prob) * 100, 2)}
                for name, prob in zip(BRAIN_CLASSES, predictions)
            ]
# Database me result log karein
            log_prediction(
                patient_name="Guest",
                module_type="Brain Tumor",
                prediction_result=prediction_result["class"],
                confidence=prediction_result["confidence"],
                additional_info="MRI Scan Analysis"
            )
    return render_template("brain_tumor.html", result=prediction_result, image_url=image_url, breakdown=breakdown)

# ----------------- Module 2: Heart Disease (ECG) -----------------
@app.route("/heart_disease", methods=["GET", "POST"])
def heart_disease():
    prediction_result = None
    breakdown = None
    error = None
    raw_signal_preview = None

    if request.method == "POST":
        signal_array = None

        # Mode A: Upload CSV/TXT file containing ECG points
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            try:
                df = pd.read_csv(file, header=None)
                signal_vals = df.values.flatten()[:187]
                if len(signal_vals) < 187:
                    signal_vals = np.pad(signal_vals, (0, 187 - len(signal_vals)), mode='constant')
                signal_array = signal_vals.astype(np.float32)
            except Exception as e:
                error = f"Error reading file: {str(e)}"

        # Mode B: Comma-separated values pasted directly in textarea
        elif request.form.get("ecg_text", "").strip():
            try:
                text_input = request.form.get("ecg_text")
                vals = [float(x.strip()) for x in text_input.replace("\n", ",").split(",") if x.strip()]
                signal_vals = np.array(vals[:187], dtype=np.float32)
                if len(signal_vals) < 187:
                    signal_vals = np.pad(signal_vals, (0, 187 - len(signal_vals)), mode='constant')
                signal_array = signal_vals
            except Exception as e:
                error = "Invalid numerical input. Please paste comma-separated numbers."

        else:
            error = "Please upload an ECG CSV/TXT file or paste sequence points."

        # Perform Prediction if sequence was extracted
        if signal_array is not None:
            model_input = signal_array.reshape(1, 187, 1)

            predictions = heart_model(model_input, training=False).numpy()[0]
            pred_idx = np.argmax(predictions)
            
            prediction_result = {
                "class": HEART_CLASSES[pred_idx],
                "confidence": round(float(predictions[pred_idx]) * 100, 2)
            }
            breakdown = [
                {"name": name, "prob": round(float(prob) * 100, 2)}
                for name, prob in zip(HEART_CLASSES, predictions)
            ]
            raw_signal_preview = [round(float(v), 4) for v in signal_array[:25]]
# Database me result log karein
            log_prediction(
                patient_name="Guest",
                module_type="Heart Disease",
                prediction_result=prediction_result["class"],
                confidence=prediction_result["confidence"],
                additional_info="ECG Signal Sequence (187 points)"
            )

    return render_template(
        "heart_disease.html",
        result=prediction_result,
        breakdown=breakdown,
        error=error,
        preview=raw_signal_preview
    )

# ----------------- Module 3: Blood Group Classification -----------------
# Dono routes handle honge: page kholne ke liye aur form submission ke liye
@app.route("/blood_group", methods=["GET"])
@app.route("/predict_blood_group", methods=["GET", "POST"])
def blood_group():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("blood_group.html", error="Please upload a blood slide image.")

        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Image Preprocessing (224x224 RGB, scaled 0 to 1)
        img = Image.open(filepath).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = blood_model(img_array, training=False).numpy()[0]
        pred_idx = np.argmax(predictions)
        predicted_class = BLOOD_CLASSES[pred_idx]
        confidence_val = round(float(predictions[pred_idx]) * 100, 2)

        compat = BLOOD_COMPATIBILITY.get(predicted_class, {
            "donate": "Information Not Available",
            "receive": "Information Not Available"
        })
# Database me result log karein

        log_prediction(
            patient_name="Guest",
            module_type="Blood Group",
            prediction_result=predicted_class,
            confidence=confidence_val,
            additional_info=f"Donate: {compat['donate']} | Receive: {compat['receive']}"
        )
        # Aapke HTML template ke exact variables:
        return render_template(
            "blood_group.html",
            prediction=predicted_class,
            confidence=confidence_val,
            donate_to=compat["donate"],
            receive_from=compat["receive"],
            img_path=filepath
        )

    return render_template("blood_group.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
   
  
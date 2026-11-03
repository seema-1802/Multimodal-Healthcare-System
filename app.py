import os
import io
import csv
import gc
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, make_response

# TensorFlow CPU & Thread Limiting (Prevents Render 512MB RAM Crash)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf

try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception:
    pass

from db import log_prediction, fetch_all_predictions

app = Flask(__name__)

# Base directory & Uploads configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Model Relative Paths
BRAIN_MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_mobilenetv2_finetuned.keras")
HEART_MODEL_PATH = os.path.join(BASE_DIR, "models", "ecg_heart_disease_lstm.keras")
BLOOD_MODEL_PATH = os.path.join(BASE_DIR, "models", "blood_group_cnn_model.keras")

_brain_model = None
_heart_model = None
_blood_model = None

def get_brain_model():
    global _brain_model, _heart_model, _blood_model
   
    gc.collect()
    if _brain_model is None:
        _brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH)
    return _brain_model

def get_heart_model():
    global _brain_model, _heart_model, _blood_model
 
    gc.collect()
    if _heart_model is None:
        _heart_model = tf.keras.models.load_model(HEART_MODEL_PATH)
    return _heart_model

def get_blood_model():
    global _brain_model, _heart_model, _blood_model
   
    gc.collect()
    if _blood_model is None:
        _blood_model = tf.keras.models.load_model(BLOOD_MODEL_PATH)
    return _blood_model

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

# ----------------- Navigation & Export Routes -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/history")
def history():
    records = fetch_all_predictions()
    return render_template("history.html", records=records)

@app.route("/export_csv")
def export_csv():
    records = fetch_all_predictions()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Patient Name", "Module Type", "Prediction Result", "Confidence (%)", "Additional Info", "Timestamp"])
    
    for row in records:
        writer.writerow([
            row["id"],
            row["patient_name"],
            row["module_type"],
            row["prediction_result"],
            row["confidence"],
            row["additional_info"],
            row["created_at"]
        ])
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=patient_diagnostic_records.csv"
    response.headers["Content-type"] = "text/csv"
    return response

# ----------------- Module 1: Brain Tumor -----------------
@app.route("/brain_tumor", methods=["GET", "POST"])
def brain_tumor():
    prediction_result = None
    image_url = None
    breakdown = None

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("brain_tumor.html", error="Please upload a valid MRI scan.")
        
        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        image_url = os.path.join("static", "uploads", file.filename)

        img = Image.open(filepath).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)

        predictions = get_brain_model()(img_array, training=False).numpy()[0]
        pred_idx = np.argmax(predictions)
        
        prediction_result = {
            "class": BRAIN_CLASSES[pred_idx],
            "confidence": round(float(predictions[pred_idx]) * 100, 2)
        }
        breakdown = [
            {"name": name, "prob": round(float(prob) * 100, 2)}
            for name, prob in zip(BRAIN_CLASSES, predictions)
        ]

       # Predictions ke theek baad:
        del img_array
        
        gc.collect()
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

        if signal_array is not None:
            model_input = signal_array.reshape(1, 187, 1)

            predictions = get_heart_model()(model_input, training=False).numpy()[0]
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

            # Explicit memory cleanup
            del model_input
            del signal_array
            
            gc.collect()

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
@app.route("/blood_group", methods=["GET", "POST"])
@app.route("/predict_blood_group", methods=["GET", "POST"])
def blood_group():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("blood_group.html", error="Please upload a blood slide image.")

        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        display_img_path = os.path.join("static", "uploads", file.filename)

        img = Image.open(filepath).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = get_blood_model()(img_array, training=False).numpy()[0]
        pred_idx = np.argmax(predictions)
        predicted_class = BLOOD_CLASSES[pred_idx]
        confidence_val = round(float(predictions[pred_idx]) * 100, 2)

        compat = BLOOD_COMPATIBILITY.get(predicted_class, {
            "donate": "Information Not Available",
            "receive": "Information Not Available"
        })

       # Predictions ke theek baad:
        del img_array
        
        gc.collect()

        log_prediction(
            patient_name="Guest",
            module_type="Blood Group",
            prediction_result=predicted_class,
            confidence=confidence_val,
            additional_info=f"Donate: {compat['donate']} | Receive: {compat['receive']}"
        )

        return render_template(
            "blood_group.html",
            prediction=predicted_class,
            confidence=confidence_val,
            donate_to=compat["donate"],
            receive_from=compat["receive"],
            img_path=display_img_path
        )

    return render_template("blood_group.html")

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
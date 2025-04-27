from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
from inference import process_video, load_model
import sys
import subprocess
from pathlib import Path
import torch

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.environ.get("UPLOAD_DIR", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 * 1024

MODEL_PATH = os.environ.get("MODEL_PATH", "models/RefInightmodel.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


def load_model_if_needed():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH, device)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    return model is not None


ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Referee Insight API is running"})


@app.route("/process", methods=["POST"])
def process_videos():
    if not load_model_if_needed():
        return jsonify({"error": "Model not available"}), 500

    if "videos" not in request.files:
        return jsonify({"error": "No videos provided"}), 400

    files = request.files.getlist("videos")
    if not files or all(file.filename == "" for file in files):
        return jsonify({"error": "No selected files"}), 400

    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(file_path)
            saved_files.append({"path": file_path, "title": filename})

    if not saved_files:
        return jsonify({"error": "No valid video files uploaded"}), 400

    try:
        results = []
        for file_info in saved_files:
            result = process_video(file_info["path"], model, device)
            results.append({"videoPath": file_info["path"], **result})

        foul_probs = np.mean([r["foulProbabilities"] for r in results], axis=0)
        offense_probs = np.mean([r["offenseProbabilities"] for r in results], axis=0)
        attention_weights = np.mean([r["attentionScore"] for r in results], axis=0)

        max_foul_prob = np.max(foul_probs)
        max_offense_prob = np.max(offense_probs)

        if max_offense_prob > 0.7:
            classification = "Red Card"
            severity_rating = 0.9
        elif max_offense_prob > 0.5 or max_foul_prob > 0.7:
            classification = "Yellow Card"
            severity_rating = 0.6
        else:
            classification = "No Card"
            severity_rating = 0.3

        response = {
            "id": str(uuid.uuid4()),
            "videos": [{"path": f["path"], "title": f["title"]} for f in saved_files],
            "actionClass": "Multiple Views",
            "severityRating": float(severity_rating),
            "classification": classification,
            "timestamp": datetime.now().isoformat(),
            "perViewResults": results,
            "aggregatedPredictions": {
                "foulProbabilities": foul_probs.tolist(),
                "offenseProbabilities": offense_probs.tolist(),
                "attentionWeights": attention_weights.tolist(),
            },
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/visualize", methods=["POST"])
def visualize_video():
    try:
        data = request.json
        if not data or "videoPath" not in data or "attentionScores" not in data:
            return jsonify({"error": "Missing required parameters"}), 400

        video_path = data["videoPath"]
        attention_scores = data["attentionScores"]

        output_dir = os.path.join(app.config["UPLOAD_FOLDER"], "visualizations")
        os.makedirs(output_dir, exist_ok=True)

        video_name = Path(video_path).stem
        visualization_path = os.path.join(output_dir, f"{video_name}_visualization.mp4")

        visualization_script = os.path.join(
            os.path.dirname(__file__), "visualization", "visualize_vars.py"
        )

        cmd = [
            sys.executable,
            visualization_script,
            "--video",
            video_path,
            "--attention",
            json.dumps(attention_scores),
            "--output",
            visualization_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return jsonify({"error": f"Visualization failed: {result.stderr}"}), 500

        return jsonify(
            {
                "message": "Visualization generated successfully",
                "visualizationPath": visualization_path,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/<path:filename>")
def serve_video(filename):
    try:
        return send_file(filename, mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

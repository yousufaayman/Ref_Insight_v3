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
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://refinsight.yousufaayman.com",
            "http://localhost:3000",
            "http://localhost:8080"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

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
            return True
        except Exception as e:
            import traceback
            print(f"Error loading model: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
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
        result = process_video([f["path"] for f in saved_files], model, device)
        
        foul_probs = np.nan_to_num(result["foulProbabilities"], nan=0.0)
        offense_probs = np.nan_to_num(result["offenseProbabilities"], nan=0.0)
        attention_scores = np.nan_to_num(result["attentionScore"], nan=0.0)

        foul_probs = foul_probs / np.maximum(np.sum(foul_probs, axis=1, keepdims=True), 1e-6)
        offense_probs = offense_probs / np.maximum(np.sum(offense_probs, axis=1, keepdims=True), 1e-6)
        attention_scores = attention_scores / np.maximum(np.sum(attention_scores), 1e-6)

        max_foul_prob = float(np.max(foul_probs))
        max_offense_prob = float(np.max(offense_probs))

        if max_offense_prob > 0.7:
            classification = "Red Card"
            severity_rating = 0.9
        elif max_offense_prob > 0.5 or max_foul_prob > 0.7:
            classification = "Yellow Card"
            severity_rating = 0.6
        else:
            classification = "No Card"
            severity_rating = 0.3

        foul_probs_list = foul_probs.tolist()
        offense_probs_list = offense_probs.tolist()
        attention_scores_list = attention_scores.tolist()

        response = {
            "id": str(uuid.uuid4()),
            "videos": [{"path": f"/video/{os.path.basename(f['path'])}", "title": f["title"]} for f in saved_files],
            "actionClass": "Multiple Views",
            "severityRating": float(severity_rating),
            "classification": classification,
            "timestamp": datetime.now().isoformat(),
            "perViewResults": [{
                "videoPath": f"/video/{os.path.basename(f['path'])}",
                "foulProbabilities": foul_probs_list[i] if i < len(foul_probs_list) else [],
                "offenseProbabilities": offense_probs_list[i] if i < len(offense_probs_list) else [],
                "attentionScore": attention_scores_list[i] if i < len(attention_scores_list) else []
            } for i, f in enumerate(saved_files)],
            "aggregatedPredictions": {
                "foulProbabilities": foul_probs_list[0] if foul_probs_list else [],
                "offenseProbabilities": offense_probs_list[0] if offense_probs_list else [],
                "attentionWeights": attention_scores_list[0] if attention_scores_list else [],
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

        
        video_filename = os.path.basename(data["videoPath"])
        
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_filename)
        attention_scores = data["attentionScores"]

        print(f"Video path from request: {data['videoPath']}")
        print(f"Extracted video filename: {video_filename}")
        print(f"Full video path: {video_path}")
        print(f"Attention scores: {attention_scores}")

        
        if not os.path.exists(video_path):
            print(f"Video file not found at: {video_path}")
            return jsonify({"error": "Video file not found"}), 404

        output_dir = os.path.join(app.config["UPLOAD_FOLDER"], "visualizations")
        os.makedirs(output_dir, exist_ok=True)

        video_name = Path(video_filename).stem
        visualization_path = os.path.join(output_dir, f"{video_name}_visualization.png")

        print(f"Output directory: {output_dir}")
        print(f"Visualization path: {visualization_path}")

        visualization_script = os.path.join(
            os.path.dirname(__file__), "visualization", "visualize_single.py"
        )
        model_path = os.path.join(os.path.dirname(__file__), "models", "RefInightmodel.pth")

        
        if not os.path.exists(visualization_script):
            print(f"Visualization script not found at: {visualization_script}")
            return jsonify({"error": "Visualization script not found"}), 500
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return jsonify({"error": "Model file not found"}), 500

        cmd = [
            sys.executable,
            visualization_script,
            video_path,
            json.dumps(attention_scores),
            visualization_path,
            model_path
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("Visualization script output:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            error_msg = f"Visualization failed with return code {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            print(error_msg)
            return jsonify({"error": error_msg}), 500

        
        visualization_files = []
        base_path = os.path.join(output_dir, f"{video_name}_visualization")
        
        
        view_pattern = f"{video_name}_visualization_view*.png"
        view_files = [f for f in os.listdir(output_dir) if f.startswith(f"{video_name}_visualization_view") and f.endswith(".png")]
        
        if view_files:
            
            view_files.sort(key=lambda x: int(x.split("view")[-1].split(".")[0]))
            visualization_files = view_files
        elif os.path.exists(visualization_path):
            
            visualization_files = [os.path.basename(visualization_path)]
        else:
            print(f"No visualization files found at: {visualization_path}")
            return jsonify({"error": "Visualization files were not created"}), 500

        print(f"Found visualization files: {visualization_files}")
        
        return jsonify({
            "message": "Visualization generated successfully",
            "visualizationPaths": visualization_files,
        })

    except Exception as e:
        print(f"Error in visualize_video: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/video/<path:filename>")
def serve_video(filename):
    try:
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(video_path):
            return jsonify({"error": "Video not found"}), 404
        return send_file(video_path, mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/visualization/<path:filename>")
def serve_visualization(filename):
    try:
        
        vis_path = os.path.join(app.config["UPLOAD_FOLDER"], "visualizations", filename)
        print(f"Attempting to serve visualization from: {vis_path}")
        if not os.path.exists(vis_path):
            print(f"Visualization not found at: {vis_path}")
            return jsonify({"error": "Visualization not found"}), 404
        print(f"Found visualization at: {vis_path}")
        return send_file(vis_path, mimetype="image/png")
    except Exception as e:
        print(f"Error serving visualization: {str(e)}")
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

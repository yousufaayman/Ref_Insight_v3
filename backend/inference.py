import argparse, torch, json
from models.vars import VARS
import torchvision.transforms as T
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from utils import VideoPreprocessor


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = VARS(pretrained=False, pooling_type="attention").to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def process_video(
    video_path: str, model: torch.nn.Module, device: torch.device
) -> Dict[str, Union[List[float], float, str]]:
    """
    Process a single video file using the provided model.

    Args:
        video_path: Path to the video file
        model: The loaded model
        device: The device to run inference on

    Returns:
        Dictionary containing foul probabilities, offense probabilities, and attention scores
    """
    preprocessor = VideoPreprocessor()
    video_tensor, timestamps = preprocessor.preprocess_video(video_path)
    video_tensor = video_tensor.to(device)

    with torch.no_grad():
        foul_probs, offense_probs, attention_scores = model(video_tensor)

    foul_probs = foul_probs.cpu().numpy()
    offense_probs = offense_probs.cpu().numpy()
    attention_scores = attention_scores.cpu().numpy()

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

    return {
        "foulProbabilities": foul_probs.tolist(),
        "offenseProbabilities": offense_probs.tolist(),
        "attentionScore": attention_scores.tolist(),
        "classification": classification,
        "severityRating": float(severity_rating),
        "timestamps": timestamps,
    }


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    # Use VideoPreprocessor to prepare videos for inference
    preprocessor = VideoPreprocessor(frames=args.frames, resolution=args.resolution)
    video_batch = preprocessor.prepare_for_inference(args.videos).to(device)

    with torch.no_grad():
        out = model(video_batch)
        # Get predictions for each view
        foul_probs = torch.softmax(out["foul_logits"], dim=1).cpu().tolist()
        offense_probs = torch.softmax(out["offense_logits"], dim=1).cpu().tolist()
        attention_scores = out["attention_scores"].cpu().tolist()

        # Prepare per-view and aggregated predictions
        predictions = {
            "per_view_predictions": [
                {
                    "video_path": video_path,
                    "foul_probabilities": foul_prob,
                    "offense_probabilities": off_prob,
                    "attention_score": attn,
                }
                for video_path, foul_prob, off_prob, attn in zip(
                    args.videos, foul_probs, offense_probs, attention_scores
                )
            ],
            "aggregated_predictions": {
                "foul_probabilities": (
                    foul_probs[-1] if len(foul_probs) > 1 else foul_probs[0]
                ),
                "offense_probabilities": (
                    offense_probs[-1] if len(offense_probs) > 1 else offense_probs[0]
                ),
                "attention_weights": attention_scores,
            },
        }

        print(json.dumps(predictions))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument(
        "--videos", nargs="+", required=True, help="Paths to each view's video"
    )
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=224)
    args = p.parse_args()
    predict(args)

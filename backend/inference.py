import argparse, torch, json
from models.vars import VARS
import torchvision.transforms as T
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from utils import VideoPreprocessor


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    try:
        print(f"Loading model from: {model_path}")
        print(f"Device: {device}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        print("Creating VARS model")
        model = VARS(pretrained=False, pooling_type="attention").to(device)
        
        print("Loading checkpoint")
        ckpt = torch.load(model_path, map_location=device)
        
        print("Loading state dict")
        model.load_state_dict(ckpt["model_state_dict"])
        
        print("Setting model to eval mode")
        model.eval()
        
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise


def process_video(
    video_paths: Union[str, List[str]], model: torch.nn.Module, device: torch.device
) -> Dict[str, Union[List[float], float, str]]:
    """
    Process multiple video files as different views of the same event using the provided model.

    Args:
        video_paths: List of paths to video files (different views of the same event)
        model: The loaded model
        device: The device to run inference on

    Returns:
        Dictionary containing foul probabilities, offense probabilities, and attention scores
    """
    try:
        print(f"Starting video processing for {len(video_paths)} views")
        print(f"Device: {device}")
        print(f"Model device: {next(model.parameters()).device}")
        
        preprocessor = VideoPreprocessor()
        print("Created VideoPreprocessor")
        
        # Process all videos together as multiple views
        video_tensor = preprocessor.prepare_for_inference(video_paths)
        print(f"Video tensor shape: {video_tensor.shape}")
        print(f"Video tensor stats - min: {video_tensor.min():.3f}, max: {video_tensor.max():.3f}, mean: {video_tensor.mean():.3f}")
        
        video_tensor = video_tensor.to(device)
        print("Moved tensor to device")

        with torch.no_grad():
            print("Running model inference")
            out = model(video_tensor)
            print("Inference completed")

            # Get predictions and handle potential NaN values
            foul_logits = out["foul_logits"]
            offense_logits = out["offense_logits"]
            attention_scores = out["attention_scores"]

            print(f"Foul logits stats - min: {foul_logits.min():.3f}, max: {foul_logits.max():.3f}, mean: {foul_logits.mean():.3f}")
            print(f"Offense logits stats - min: {offense_logits.min():.3f}, max: {offense_logits.max():.3f}, mean: {offense_logits.mean():.3f}")
            print(f"Attention scores stats - min: {attention_scores.min():.3f}, max: {attention_scores.max():.3f}, mean: {attention_scores.mean():.3f}")

            # Apply softmax and handle potential NaN values
            foul_probs = torch.softmax(foul_logits, dim=1)
            offense_probs = torch.softmax(offense_logits, dim=1)
            
            # Replace NaN values with zeros
            foul_probs = torch.nan_to_num(foul_probs, nan=0.0)
            offense_probs = torch.nan_to_num(offense_probs, nan=0.0)
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0)

            # Convert to numpy arrays
            foul_probs = foul_probs.cpu().numpy()
            offense_probs = offense_probs.cpu().numpy()
            attention_scores = attention_scores.cpu().numpy()

            # Normalize probabilities if they sum to zero
            foul_probs = foul_probs / np.maximum(np.sum(foul_probs, axis=1, keepdims=True), 1e-6)
            offense_probs = offense_probs / np.maximum(np.sum(offense_probs, axis=1, keepdims=True), 1e-6)
            attention_scores = attention_scores / np.maximum(np.sum(attention_scores), 1e-6)

        return {
            "foulProbabilities": foul_probs,
            "offenseProbabilities": offense_probs,
            "attentionScore": attention_scores,
        }
    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise


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

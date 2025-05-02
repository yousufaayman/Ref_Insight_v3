import sys
import os


backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

import torch
import json
import cv2
import numpy as np
from models.vars import VARS
from utils.preprocess import VideoPreprocessor
from attention_rollout import visualize_attention_rollout
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

def create_simple_visualization(video_path, attention_scores, output_path):
    """Create a simple visualization using the video frame and attention scores."""
    print("Attempting simple visualization...")
    try:
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {middle_frame} from video")

        
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0  

        
        attention = np.array(attention_scores)
        attention = attention.reshape(-1, 1)  
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        attention_map = cv2.resize(attention, (224, 224))

        
        heatmap = plt.get_cmap('jet')(attention_map)[..., :3]

        
        overlay = 0.7 * frame + 0.3 * heatmap

        
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        print(f"Simple visualization saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error in simple visualization: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def visualize_single_video(video_path, attention_scores, output_path, model_path):
    print(f"\nStarting visualization process...")
    print(f"Video path: {video_path}")
    print(f"Output path: {output_path}")
    print(f"Model path: {model_path}")
    print(f"Attention scores shape: {np.array(attention_scores).shape}")

    try:
        
        print("\nAttempting complex visualization with attention rollout...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        
        try:
            model = VARS(
                num_foul_types=8,
                num_offense_categories=4,
                pretrained=False,
                pooling_type="attention",
            )
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        
        try:
            preprocessor = VideoPreprocessor()
            video_tensor = preprocessor.prepare_for_inference([video_path])
            video_tensor = video_tensor.to(device)
            print(f"Video preprocessed successfully, tensor shape: {video_tensor.shape}")
        except Exception as e:
            print(f"Error preprocessing video: {str(e)}")
            raise

        
        try:
            attention_scores = torch.tensor(attention_scores, device=device)
            print(f"Attention scores converted to tensor, shape: {attention_scores.shape}")
        except Exception as e:
            print(f"Error converting attention scores: {str(e)}")
            raise

        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        
        try:
            print("Generating complex visualization...")
            visualize_attention_rollout(model, video_tensor, save_path=output_path)
            if os.path.exists(output_path) or any(
                os.path.exists(f"{os.path.splitext(output_path)[0]}_view{i}{os.path.splitext(output_path)[1]}")
                for i in range(5)
            ):
                print("Complex visualization succeeded")
                return
            else:
                print("Complex visualization did not create output file")
                raise Exception("No output file created by complex visualization")
        except Exception as e:
            print(f"Error in complex visualization: {str(e)}")
            raise

    except Exception as e:
        print(f"Complex visualization failed: {str(e)}")
        print("Falling back to simple visualization...")
        
        
        if create_simple_visualization(video_path, attention_scores, output_path):
            print("Simple visualization succeeded")
            return
        
        print("Both visualization methods failed")
        raise Exception("Failed to create visualization using either method")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python visualize_single.py <video_path> <attention_scores_json> <output_path> <model_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    attention_scores = json.loads(sys.argv[2])
    output_path = sys.argv[3]
    model_path = sys.argv[4]

    try:
        visualize_single_video(video_path, attention_scores, output_path, model_path)
        print("Visualization completed successfully")
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1) 
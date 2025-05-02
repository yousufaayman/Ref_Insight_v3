import os
import cv2
import numpy as np
import torch
from pathlib import Path
import shutil
import argparse
import json
from typing import List, Tuple, Union, Dict


class VideoPreprocessor:
    def __init__(
        self,
        output_dir: str = "processed_videos",
        frames: int = 16,
        resolution: int = 224,
    ):
        self.output_dir = output_dir
        self.frames = frames
        self.resolution = resolution
        os.makedirs(output_dir, exist_ok=True)

    def process_video(self, video_path: str) -> Tuple[str, List[float]]:
        video_name = Path(video_path).stem
        output_path = os.path.join(self.output_dir, f"{video_name}_processed.mp4")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = np.linspace(0, frame_count - 1, self.frames, dtype=int)
        frame_timestamps = frame_indices / fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, fps, (self.resolution, self.resolution)
        )

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame {idx} from video")

            frame = cv2.resize(frame, (self.resolution, self.resolution))
            out.write(frame)

        cap.release()
        out.release()

        return output_path, frame_timestamps.tolist()

    def prepare_for_inference(self, video_paths: Union[str, List[str]]) -> torch.Tensor:
        """
        Prepare multiple videos for inference by extracting frames and creating a tensor.

        Args:
            video_paths: List of paths to video files (different views of the same event)

        Returns:
            Tensor of shape [batch_size, num_views, channels, frames, height, width]
        """
        if isinstance(video_paths, str):
            video_paths = [video_paths]
            
        print(f"Preparing {len(video_paths)} videos for inference")
        
        
        video_tensors = []
        for video_path in video_paths:
            print(f"Processing video: {video_path}")
            frames = self.extract_frames(video_path)
            print(f"Extracted {len(frames)} frames")
            
            
            processed_frames = []
            for frame in frames:
                frame = self.preprocess_frame(frame)
                print(f"Processed frame shape: {frame.shape}")
                processed_frames.append(frame)
            
            
            video_tensor = torch.stack(processed_frames)
            print(f"Video tensor shape after stacking frames: {video_tensor.shape}")
            video_tensors.append(video_tensor)
        
        
        final_tensor = torch.stack(video_tensors)
        print(f"Final tensor shape after stacking videos: {final_tensor.shape}")
        
        
        
        final_tensor = final_tensor.permute(0, 2, 1, 3, 4)
        print(f"Final tensor shape after permuting: {final_tensor.shape}")
        
        
        final_tensor = final_tensor.unsqueeze(0)
        print(f"Final tensor shape after adding batch dimension: {final_tensor.shape}")
        
        
        final_tensor = final_tensor.contiguous()
        
        return final_tensor

    def validate_video(self, video_path: str) -> bool:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < self.frames:
                return False

            ret, frame = cap.read()
            if not ret:
                return False

            cap.release()
            return True

        except Exception as e:
            print(f"Error validating video: {e}")
            return False

    def cleanup(self, video_path: str):
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            print(f"Error cleaning up video: {e}")

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, cnt - 1, self.frames, dtype=int)
        frames = []

        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, img = cap.read()
            img = cv2.resize(img, (self.resolution, self.resolution))
            frames.append(img)

        cap.release()
        return frames

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        
        frame = frame[..., ::-1] / 255.0
        
        
        tensor = torch.tensor(frame, dtype=torch.float32)
        
        
        tensor = tensor.permute(2, 0, 1)  
        
        return tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video preprocessing utility")
    parser.add_argument("video_path", help="Path to the video file to process")
    parser.add_argument(
        "--output_dir",
        default="processed_videos",
        help="Directory to save processed videos",
    )
    parser.add_argument(
        "--frames", type=int, default=16, help="Number of frames to extract"
    )
    parser.add_argument(
        "--resolution", type=int, default=224, help="Resolution for resizing frames"
    )
    args = parser.parse_args()

    preprocessor = VideoPreprocessor(
        output_dir=args.output_dir, frames=args.frames, resolution=args.resolution
    )

    if preprocessor.validate_video(args.video_path):
        try:
            processed_path, timestamps = preprocessor.process_video(args.video_path)
            print(processed_path)
            print(json.dumps(timestamps))
        except Exception as e:
            print(f"Error processing video: {e}")
    else:
        print("Invalid video file")

import sys

sys.path.append(".")
import torch
import os
import argparse
from models.vars import VARS
from data.dataset import SoccerNetMVFoulDataset
from config.config import Config
from config.classes import EVENT_DICTIONARY
from attention_rollout import visualize_attention_rollout


def parse_args():
    parser = argparse.ArgumentParser(description="VARS Attention Rollout Visualization")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization/rollout",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VARS(
        num_foul_types=8,
        num_offense_categories=4,
        pretrained=False,
        pooling_type="attention",
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    val_dataset = SoccerNetMVFoulDataset(
        root_dir=args.data_root,
        split="val",
        frames=Config.FRAMES,
        resolution=Config.RESOLUTION,
        num_views=None,
    )

    for i in range(min(args.num_samples, len(val_dataset))):
        sample = val_dataset[i]
        videos = sample["video"].unsqueeze(0).to(device)
        action_id = sample["action_id"]

        print(f"\nSample {i} - Action ID: {action_id}")

        save_path = os.path.join(args.output_dir, f"rollout_sample_{action_id}.png")
        visualize_attention_rollout(model, videos, save_path=save_path)


if __name__ == "__main__":
    main()


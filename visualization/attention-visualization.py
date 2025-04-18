import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from models.vars import VARS
from data.dataset import SoccerNetMVFoulDataset
from config.config import Config
from config.classes import EVENT_DICTIONARY
import torchvision.transforms.functional as F


def visualize_attention_weights(model, video_batch, save_path=None):
    model.eval()

    with torch.no_grad():

        outputs = model(video_batch)

        attention_scores = outputs["attention_scores"]

        _, foul_preds = torch.max(outputs["foul_logits"], 1)
        _, offense_preds = torch.max(outputs["offense_logits"], 1)

        attention_np = attention_scores.cpu().numpy()[0]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(attention_np)), attention_np, color="skyblue")
        plt.xlabel("View Index")
        plt.ylabel("Attention Weight")
        plt.title("Attention Weights Across Views")
        plt.xticks(range(len(attention_np)))

        for i, v in enumerate(attention_np):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Attention visualization saved to {save_path}")

        plt.show()

        return attention_np


def visualize_sample_with_attention(model, video_batch, sample_data, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():

        outputs = model(video_batch)

        attention_scores = outputs["attention_scores"][0].cpu().numpy()
        foul_logits = outputs["foul_logits"][0].cpu().numpy()
        offense_logits = outputs["offense_logits"][0].cpu().numpy()

        foul_gt = torch.argmax(sample_data["action_label"]).item()
        offense_gt = torch.argmax(sample_data["offense_severity_label"]).item()
        foul_pred = np.argmax(foul_logits)
        offense_pred = np.argmax(offense_logits)

        foul_classes = list(EVENT_DICTIONARY["action_class"].keys())
        offense_classes = list(EVENT_DICTIONARY["offense_severity"].keys())

        num_views = video_batch.shape[1]
        plt.figure(figsize=(4 * num_views, 8))

        plt.subplot(2, 1, 1)
        bars = plt.bar(range(num_views), attention_scores, color="skyblue")
        plt.xlabel("View Index")
        plt.ylabel("Attention Weight")
        plt.title("Attention Weights Across Views")
        plt.xticks(range(num_views))

        for i, v in enumerate(attention_scores):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

        for i, bar in enumerate(bars):

            bar.set_color(plt.cm.coolwarm(attention_scores[i] / max(attention_scores)))

        plt.subplot(2, 1, 2)

        for i in range(num_views):
            plt.subplot(2, num_views, num_views + i + 1)

            middle_frame_idx = video_batch.shape[2] // 2
            frame = video_batch[0, i, :, middle_frame_idx].cpu()

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame = frame * std + mean

            frame = frame.permute(1, 2, 0).numpy()
            frame = np.clip(frame, 0, 1)

            plt.imshow(frame)
            plt.title(f"View {i}\nAtt: {attention_scores[i]:.3f}")
            plt.axis("off")

        plt.suptitle(
            f"Action ID: {sample_data['action_id']}\n"
            + f"Foul: {foul_classes[foul_gt]} (Pred: {foul_classes[foul_pred]})\n"
            + f"Offense: {offense_classes[offense_gt]} (Pred: {offense_classes[offense_pred]})",
            fontsize=14,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir:
            save_path = os.path.join(
                save_dir, f"attention_sample_{sample_data['action_id']}.png"
            )
            plt.savefig(save_path)
            print(f"Sample visualization saved to {save_path}")

        plt.show()


def run_attention_visualization(model_path, data_root, output_dir, num_samples=5):

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    print(f"Loaded model from {model_path}")

    val_dataset = SoccerNetMVFoulDataset(
        root_dir=data_root,
        split="val",
        frames=Config.FRAMES,
        resolution=Config.RESOLUTION,
        num_views=None,
    )

    num_samples = 5
    for i in range(num_samples):

        sample = val_dataset[i]
        videos = sample["video"].to(device)

        videos = videos.unsqueeze(0)

        sample_data = {
            "action_id": sample["action_id"],
            "action_label": sample["action_label"],
            "offense_severity_label": sample["offense_severity_label"],
        }

        print(f"\nVisualizing sample {i} (Action ID: {sample_data['action_id']})")
        visualize_sample_with_attention(model, videos, sample_data, save_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Attention Visualization for VARS Model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization/attention",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize"
    )

    args = parser.parse_args()

    run_attention_visualization(
        model_path=args.model_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )

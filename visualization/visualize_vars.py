import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from models.vars import VARS
from data.dataset import SoccerNetMVFoulDataset
from config.config import Config
from config.classes import EVENT_DICTIONARY


from gradcam import GradCAM, visualize_vars_gradcam


def parse_args():
    parser = argparse.ArgumentParser(description="VARS Model Visualization")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--viz_type",
        type=str,
        default="both",
        choices=["attention", "gradcam", "both"],
        help="Type of visualization to generate",
    )

    return parser.parse_args()


def visualize_attention_weights(model, video_batch, save_path=None):
    """
    Visualize the attention weights across different views
    """

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
    """
    Visualize a sample with attention weights and sample frames
    """
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


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    attention_dir = os.path.join(args.output_dir, "attention")
    gradcam_dir = os.path.join(args.output_dir, "gradcam")

    if args.viz_type in ["attention", "both"]:
        os.makedirs(attention_dir, exist_ok=True)

    if args.viz_type in ["gradcam", "both"]:
        os.makedirs(gradcam_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VARS(
        num_foul_types=8,
        num_offense_categories=4,
        pretrained=False,
        pooling_type="attention",
    )

    checkpoint = torch.load(args.model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])

    model = model.to(device)
    model.eval()

    print(f"Loaded model from {args.model_path}")

    val_dataset = SoccerNetMVFoulDataset(
        root_dir=args.data_root,
        split="val",
        frames=Config.FRAMES,
        resolution=Config.RESOLUTION,
        num_views=None,
    )

    foul_classes = list(EVENT_DICTIONARY["action_class"].keys())
    offense_classes = list(EVENT_DICTIONARY["offense_severity"].keys())

    for i in range(min(args.num_samples, len(val_dataset))):

        sample = val_dataset[i]
        videos = sample["video"].to(device)

        videos = videos.unsqueeze(0)

        action_label = torch.argmax(sample["action_label"]).item()
        offense_label = torch.argmax(sample["offense_severity_label"]).item()
        action_id = sample["action_id"]

        action_name = foul_classes[action_label]
        offense_name = offense_classes[offense_label]

        print(
            f"\nSample {i} - Action ID: {action_id}, Action: {action_name}, Offense: {offense_name}"
        )

        sample_data = {
            "action_id": action_id,
            "action_label": sample["action_label"],
            "offense_severity_label": sample["offense_severity_label"],
        }

        if args.viz_type in ["attention", "both"]:
            print("Generating attention visualization...")
            visualize_sample_with_attention(
                model, videos, sample_data, save_dir=attention_dir
            )

        if args.viz_type in ["gradcam", "both"]:
            print("Generating Grad-CAM visualizations...")
            grad_cam = GradCAM(model)

            for view_idx in range(min(2, videos.shape[1])):
                save_path = os.path.join(
                    gradcam_dir, f"sample_{i}_view_{view_idx}_foul.png"
                )

                try:
                    visualize_vars_gradcam(
                        model,
                        videos,
                        class_idx=action_label,
                        task="foul",
                        view_idx=view_idx,
                        save_path=save_path,
                    )
                except Exception as e:
                    print(f"Error generating Grad-CAM for foul (view {view_idx}): {e}")

            for view_idx in range(min(2, videos.shape[1])):
                save_path = os.path.join(
                    gradcam_dir, f"sample_{i}_view_{view_idx}_offense.png"
                )

                try:
                    visualize_vars_gradcam(
                        model,
                        videos,
                        class_idx=offense_label,
                        task="offense",
                        view_idx=view_idx,
                        save_path=save_path,
                    )
                except Exception as e:
                    print(
                        f"Error generating Grad-CAM for offense (view {view_idx}): {e}"
                    )

    print(f"\nVisualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()

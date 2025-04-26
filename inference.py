import argparse, torch, json
from models.vars import VARS
import torchvision.transforms as T
import cv2
import numpy as np


def load_model(path, device):
    model = VARS(pretrained=False, pooling_type="attention").to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def preprocess(video_paths, frames=16, resolution=224):

    clips = []
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, cnt - 1, frames, dtype=int)
        frames_list = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, img = cap.read()
            img = cv2.resize(img, (resolution, resolution))
            img = img[..., ::-1] / 255.0
            frames_list.append(img)
        cap.release()
        clip = np.stack(frames_list, axis=0)
        clips.append(clip)

    arr = np.stack(clips, axis=0)
    arr = np.transpose(arr, (0, 1, 4, 0, 2, 3))
    return torch.tensor(arr, dtype=torch.float32)


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    video_batch = preprocess(args.videos, args.frames, args.resolution).to(device)
    with torch.no_grad():
        out = model(video_batch)
        foul_p = torch.softmax(out["foul_logits"], dim=1).cpu().tolist()
        off_p = torch.softmax(out["offense_logits"], dim=1).cpu().tolist()
        attn = out["attention_scores"].cpu().tolist()
    print(
        json.dumps(
            {"foul_probs": foul_p[0], "offense_probs": off_p[0], "attention": attn[0]}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument(
        "--videos", nargs="+", required=True, help="Paths to each view’s video"
    )
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=224)
    args = p.parse_args()
    predict(args)

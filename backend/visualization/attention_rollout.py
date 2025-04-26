import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import math
import cv2

ATTENTION_CACHE = {}


class MultiscaleAttentionWithCapture(torch.nn.Module):
    def __init__(self, original_attn, identifier):
        super().__init__()
        self.original_attn = copy.deepcopy(original_attn)
        self.identifier = identifier
        self.qkv = self.original_attn.qkv
        self.num_heads = self.original_attn.num_heads

        embed_dim = self.original_attn.qkv.out_features
        self.head_dim = embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

    def forward(self, x, thw):
        out, thw_new = self.original_attn.forward(x, thw)
        if hasattr(self, "qkv"):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = qkv.unbind(2)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            ATTENTION_CACHE[self.identifier] = attn.detach()
        return out, thw_new


def patch_attention_layer(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            parent = model
            for sub in name.split(".")[:-1]:
                parent = getattr(parent, sub)
            last = name.split(".")[-1]
            original = getattr(parent, last)
            wrapped = MultiscaleAttentionWithCapture(original, identifier=name)
            setattr(parent, last, wrapped)
            break


def extract_attention_rollout(
    model, video_batch, layer_name="encoder.mvit.blocks.15.attn"
):
    ATTENTION_CACHE.clear()
    patch_attention_layer(model, layer_name)
    _ = model(video_batch)

    if layer_name in ATTENTION_CACHE:
        attn = ATTENTION_CACHE[layer_name]
        avg_attn = attn.mean(dim=1)
        rollout = torch.eye(avg_attn.size(-1), device=avg_attn.device)
        rollout = rollout @ avg_attn[0]
        return rollout
    else:
        print("No attention map captured.")
        return None


def visualize_attention_rollout(model, video_batch, save_path=None):
    """
    Generates and overlays attention rollout heatmaps on the most representative frame (middle frame)
    of each view clip. Saves one image per view.
    """

    B, V, C, T, H, W = video_batch.shape
    mid = T // 2

    for v in range(V):

        single = video_batch[:, v : v + 1, :, :, :]

        rollout = extract_attention_rollout(model, single)
        if rollout is None:
            continue

        rollout_vec = rollout[0, 1:]
        num_tokens = rollout_vec.shape[0]
        size = int(math.sqrt(num_tokens))
        while size * size > num_tokens:
            size -= 1
        heatmap = rollout_vec[: size * size].view(size, size).cpu().numpy()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        frame_t = single[0, 0, :, mid]
        mean = torch.tensor([0.485, 0.456, 0.406], device=frame_t.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=frame_t.device).view(3, 1, 1)
        frame = (frame_t * std + mean).permute(1, 2, 0).cpu().numpy()
        frame = np.clip(frame, 0, 1)

        heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
        colored = plt.get_cmap("jet")(heatmap_resized)[..., :3]

        overlay = 0.4 * frame + 0.6 * colored

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"Attention Overlay - View {v}")
        plt.axis("off")

        if save_path:
            base, ext = os.path.splitext(save_path)
            view_path = f"{base}_view{v}{ext}"
            os.makedirs(os.path.dirname(view_path), exist_ok=True)
            plt.savefig(view_path)
            print(f"Saved overlay to {view_path}")

        plt.show()

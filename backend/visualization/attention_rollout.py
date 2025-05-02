import torch
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
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
    model, video_batch, layer_name="mvit.blocks.15.attn"
):
    layer_names_to_try = [
        layer_name,
        "encoder.mvit.blocks.15.attn",
        "encoder.mvit.blocks.11.attn",
        "encoder.mvit.blocks.7.attn",
        "encoder.mvit.blocks.3.attn",
        "mvit.blocks.15.attn",
        "mvit.blocks.11.attn",
        "mvit.blocks.7.attn",
        "mvit.blocks.3.attn"
    ]

    # First try to get the encoder
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'mvit'):
        base_model = model.encoder.mvit
    else:
        base_model = model

    # Try to find any attention layer
    attention_layer = None
    for name, module in base_model.named_modules():
        if 'attn' in name.lower():
            attention_layer = module
            break

    if attention_layer is None:
        print("Error: No attention layer found in model")
        return None

    for try_layer in layer_names_to_try:
        ATTENTION_CACHE.clear()
        try:
            patch_attention_layer(model, try_layer)
            with torch.no_grad():
                _ = model(video_batch)

            if try_layer in ATTENTION_CACHE:
                attn = ATTENTION_CACHE[try_layer]
                avg_attn = attn.mean(dim=1)
                rollout = torch.eye(avg_attn.size(-1), device=avg_attn.device)
                rollout = rollout @ avg_attn[0]
                return rollout
        except Exception as e:
            print(f"Error trying layer {try_layer}: {str(e)}")
            continue

    # If no layer worked, try using the found attention layer directly
    try:
        ATTENTION_CACHE.clear()
        wrapped = MultiscaleAttentionWithCapture(attention_layer, identifier="direct_attn")
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'mvit'):
            # Replace the attention layer in the MViT model
            for name, module in model.encoder.mvit.named_modules():
                if module == attention_layer:
                    parent_name = '.'.join(name.split('.')[:-1])
                    last_name = name.split('.')[-1]
                    parent = model.encoder.mvit
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    setattr(parent, last_name, wrapped)
                    break
        
        with torch.no_grad():
            _ = model(video_batch)

        if "direct_attn" in ATTENTION_CACHE:
            attn = ATTENTION_CACHE["direct_attn"]
            avg_attn = attn.mean(dim=1)
            rollout = torch.eye(avg_attn.size(-1), device=avg_attn.device)
            rollout = rollout @ avg_attn[0]
            return rollout
    except Exception as e:
        print(f"Error with direct attention layer access: {str(e)}")

    print("Error: No attention map captured from any method.")
    return None


def visualize_attention_rollout(model, video_batch, save_path=None):
    B, V, C, T, H, W = video_batch.shape
    mid = T // 2

    for v in range(V):
        single = video_batch[:, v:v+1, :, :, :]

        rollout = extract_attention_rollout(model, single)
        if rollout is None:
            print(f"Error: No attention rollout for view {v}")
            continue

        rollout_vec = rollout[0, 1:]
        num_tokens = rollout_vec.shape[0]
        size = int(math.sqrt(num_tokens))
        while size * size > num_tokens:
            size -= 1

        try:
            heatmap = rollout_vec[:size * size].view(size, size).cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            frame_t = single[0, 0, :, mid]
            mean = torch.tensor([0.485, 0.456, 0.406], device=frame_t.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=frame_t.device).view(3, 1, 1)
            frame = (frame_t * std + mean).permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame, 0, 1)

            heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
            colored = plt.get_cmap('jet')(heatmap_resized)[..., :3]
            overlay = 0.4 * frame + 0.6 * colored

            plt.figure(figsize=(12, 12))
            plt.imshow(overlay)
            plt.title(f'Attention Overlay - View {v}')
            plt.axis('off')

            if save_path:
                try:
                    base, ext = os.path.splitext(save_path)
                    view_path = f"{base}_view{v}{ext}"
                    os.makedirs(os.path.dirname(view_path), exist_ok=True)
                    plt.savefig(view_path, bbox_inches='tight', pad_inches=0, dpi=300)
                except Exception as e:
                    print(f"Error saving visualization for view {v}: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
            
            plt.close()

        except Exception as e:
            print(f"Error processing visualization for view {v}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue

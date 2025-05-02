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
    print(f"Attempting to extract attention rollout from layer: {layer_name}")
    print("Available layers:")
    for name, _ in model.named_modules():
        print(f"  {name}")

    # Try different layer names based on the actual model structure
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
        print("Found MViT model in encoder")
    else:
        base_model = model
        print("Using base model")

    # Try to find any attention layer
    attention_layer = None
    for name, module in base_model.named_modules():
        if 'attn' in name.lower():
            print(f"Found attention layer: {name}")
            attention_layer = module
            break

    if attention_layer is None:
        print("No attention layer found in model")
        return None

    for try_layer in layer_names_to_try:
        print(f"\nTrying layer: {try_layer}")
        ATTENTION_CACHE.clear()
        try:
            patch_attention_layer(model, try_layer)
            with torch.no_grad():
                _ = model(video_batch)

            if try_layer in ATTENTION_CACHE:
                print(f"Successfully captured attention from layer: {try_layer}")
                attn = ATTENTION_CACHE[try_layer]
                avg_attn = attn.mean(dim=1)
                rollout = torch.eye(avg_attn.size(-1), device=avg_attn.device)
                rollout = rollout @ avg_attn[0]
                return rollout
        except Exception as e:
            print(f"Error trying layer {try_layer}: {str(e)}")
            continue

    # If no layer worked, try using the found attention layer directly
    print("\nTrying direct attention layer access")
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
            print("Successfully captured attention from direct layer access")
            attn = ATTENTION_CACHE["direct_attn"]
            avg_attn = attn.mean(dim=1)
            rollout = torch.eye(avg_attn.size(-1), device=avg_attn.device)
            rollout = rollout @ avg_attn[0]
            return rollout
    except Exception as e:
        print(f"Error with direct attention layer access: {str(e)}")

    print("No attention map captured from any method.")
    return None


def visualize_attention_rollout(model, video_batch, save_path=None):
    print("\nStarting visualization...")
    print(f"Model structure:\n{model}")
    
    B, V, C, T, H, W = video_batch.shape
    mid = T // 2
    print(f"Video batch shape: {video_batch.shape}")

    for v in range(V):
        print(f"\nProcessing view {v}")
        single = video_batch[:, v:v+1, :, :, :]
        print(f"Single view tensor shape: {single.shape}")

        rollout = extract_attention_rollout(model, single)
        if rollout is None:
            print(f"No attention rollout for view {v}")
            continue

        print(f"Rollout shape: {rollout.shape}")
        rollout_vec = rollout[0, 1:]
        print(f"Rollout vector shape: {rollout_vec.shape}")

        num_tokens = rollout_vec.shape[0]
        size = int(math.sqrt(num_tokens))
        while size * size > num_tokens:
            size -= 1
        print(f"Reshaped size: {size}x{size}")

        try:
            heatmap = rollout_vec[:size * size].view(size, size).cpu().numpy()
            print(f"Heatmap shape: {heatmap.shape}")

            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            print(f"Normalized heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

            frame_t = single[0, 0, :, mid]
            print(f"Frame tensor shape: {frame_t.shape}")

            mean = torch.tensor([0.485, 0.456, 0.406], device=frame_t.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=frame_t.device).view(3, 1, 1)
            frame = (frame_t * std + mean).permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame, 0, 1)
            print(f"Processed frame shape: {frame.shape}")

            heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
            print(f"Resized heatmap shape: {heatmap_resized.shape}")

            colored = plt.get_cmap('jet')(heatmap_resized)[..., :3]
            print(f"Colored heatmap shape: {colored.shape}")

            overlay = 0.4 * frame + 0.6 * colored
            print(f"Final overlay shape: {overlay.shape}")

            plt.figure(figsize=(12, 12))
            plt.imshow(overlay)
            plt.title(f'Attention Overlay - View {v}')
            plt.axis('off')

            if save_path:
                try:
                    base, ext = os.path.splitext(save_path)
                    view_path = f"{base}_view{v}{ext}"
                    os.makedirs(os.path.dirname(view_path), exist_ok=True)
                    print(f"Saving visualization to: {view_path}")
                    plt.savefig(view_path, bbox_inches='tight', pad_inches=0, dpi=300)
                    print(f"Successfully saved overlay to {view_path}")
                except Exception as e:
                    print(f"Error saving visualization for view {v}: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
            
            plt.close()  # Close the figure to free memory

        except Exception as e:
            print(f"Error processing visualization for view {v}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue

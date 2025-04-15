import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class GradCAM:
    def __init__(self, model, target_layer_name="mvit.stages.3"):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        
        self._register_hooks()
    
    def _register_hooks(self):
        
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                
                module.register_forward_hook(self._save_activations)
                
                module.register_full_backward_hook(self._save_gradients)
                break
    
    def _save_activations(self, module, input, output):
        self.activations = output
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, video_input, target_class, task='foul'):
        
        self.model.eval()
        
        
        if len(video_input.shape) == 4:
            video_input = video_input.unsqueeze(0)
        
        video_tensor = video_input.clone().detach().requires_grad_(True)
        
        outputs = self.model(video_tensor)
        
        if task == 'foul':
            logits = outputs['foul_logits']
        else:  
            logits = outputs['offense_logits']
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        
        logits.backward(gradient=one_hot, retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        
        for i in range(pooled_gradients.shape[0]):
            self.activations[:, i, :, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze().detach().cpu()
        heatmap = F.relu(heatmap)
        
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
        
        return heatmap

    def visualize(self, video_input, target_class, task='foul', save_path=None, view_idx=0, frame_idx=None):
        
        heatmap = self.generate_cam(video_input, target_class, task)
        
        if frame_idx is None:
            frame_idx = heatmap.shape[0] // 2  
        
        frame = video_input[0, view_idx, :, frame_idx].permute(1, 2, 0).cpu().numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = std * frame + mean
        frame = np.clip(frame, 0, 1)
        
        cam_heatmap = heatmap[frame_idx].numpy()
        
        cam_heatmap = cv2.resize(cam_heatmap, (frame.shape[1], frame.shape[0]))
        
        colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        cam_heatmap_colored = cmap(cam_heatmap)[:, :, :3]
        
        alpha = 0.4
        superimposed_img = (1-alpha) * frame + alpha * cam_heatmap_colored
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title('Original Frame')
        plt.imshow(frame)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('Grad-CAM Heatmap')
        plt.imshow(cam_heatmap, cmap='jet')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('Overlaid Image')
        plt.imshow(superimposed_img)
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return superimposed_img


def visualize_vars_gradcam(model, video_batch, class_idx, task='foul', view_idx=0, frame_idx=None, save_path=None):
    grad_cam = GradCAM(model)
    
    
    superimposed = grad_cam.visualize(
        video_batch, 
        target_class=class_idx,
        task=task,
        view_idx=view_idx,
        frame_idx=frame_idx,
        save_path=save_path
    )
    
    return superimposed
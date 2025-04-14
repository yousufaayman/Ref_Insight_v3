import torch
import torch.nn as nn
from config.config import Config

class VideoEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(VideoEncoder, self).__init__()
        
        if Config.MODEL_VARIANT == 'small':
            from torchvision.models.video import mvit_v2_s            self.mvit = mvit_v2_s(pretrained=pretrained)
        elif Config.MODEL_VARIANT == 'base':
            from torchvision.models.video import mvit_v2_b
            self.mvit = mvit_v2_b(pretrained=pretrained)
        elif Config.MODEL_VARIANT == 'large':
            from torchvision.models.video import mvit_v2_l
            self.mvit = mvit_v2_l(pretrained=pretrained)
        else:
            
            from torchvision.models.video import mvit_v2_s
            self.mvit = mvit_v2_b(pretrained=pretrained)
            print(f"Unknown model variant: {Config.MODEL_VARIANT}, using 'base' as default.")
        
        
        num_features = self.mvit.head.proj.in_features
        self.feature_dim = num_features
        self.mvit.head.proj = nn.Identity()  
        
        print(f"Initialized VideoEncoder with MViTv2-{Config.MODEL_VARIANT}, feature dimension: {self.feature_dim}")
        
    def forward(self, x):
        batch_size, num_views = x.shape[0], x.shape[1]
        
        
        x = x.reshape(-1, *x.shape[2:])  
        
        
        features = self.mvit(x)  
        
        
        features = features.view(batch_size, num_views, -1)  
        
        return features
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from config.config import Config

class VideoEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(VideoEncoder, self).__init__()
        
        
        weights = MViT_V2_S_Weights.DEFAULT
        
        self.mvit = mvit_v2_s(weights=weights)
        self.feature_dim = 768  
        
        if isinstance(self.mvit.head, nn.Sequential):
            
            head_layers = list(self.mvit.head.children())
            if len(head_layers) >= 2 and isinstance(head_layers[-1], nn.Linear):
                
                self.feature_dim = head_layers[-1].in_features
                
                new_head = nn.Sequential(*head_layers[:-1])
                self.mvit.head = new_head
        
        print(f"Initialized VideoEncoder with MViTv2-small, feature dimension: {self.feature_dim}")
        
    def forward(self, x):

        batch_size, num_views = x.shape[0], x.shape[1]
        
        
        x = x.reshape(-1, *x.shape[2:])  
        
        
        features = self.mvit(x)  
        
        
        features = features.view(batch_size, num_views, -1)  
        
        return features
    
    
    
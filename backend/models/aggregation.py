import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanPoolAggregation(nn.Module):    
    def forward(self, features):
        attention = torch.ones(features.shape[0], features.shape[1], device=features.device)
        attention = attention / attention.sum(dim=1, keepdim=True)
        
        aggregated = torch.mean(features, dim=1)
        return aggregated, attention


class MaxPoolAggregation(nn.Module):
    
    def forward(self, features):
        
        aggregated, indices = torch.max(features, dim=1)
        
        
        attention = torch.zeros(features.shape[0], features.shape[1], device=features.device)
        batch_indices = torch.arange(features.shape[0], device=features.device)
        
        
        view_indices = indices[:, 0]  
        attention[batch_indices, view_indices] = 1.0
        
        return aggregated, attention


class AttentionAggregation(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionAggregation, self).__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, feature_dim))
        
        nn.init.orthogonal_(self.W)
        
    def forward(self, features):
        
        transformed = features @ self.W  
        
        
        S = transformed @ transformed.transpose(1, 2)  
        
        
        S = F.relu(S)
        
        
        N = S / (S.sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        
        attention = N.sum(dim=2)  
        
        
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)
        
        
        weighted_features = features * attention.unsqueeze(2)
        aggregated = weighted_features.sum(dim=1)
        
        return aggregated, attention
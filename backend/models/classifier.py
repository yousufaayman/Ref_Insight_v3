import torch
import torch.nn as nn

class FoulClassifier(nn.Module):    
    def __init__(self, input_dim, num_classes=8, dropout_rate=0.5):
        super(FoulClassifier, self).__init__()
        
        
        hidden_dim = min(1024, input_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class OffenseClassifier(nn.Module):
    
    def __init__(self, input_dim, num_classes=4, dropout_rate=0.5):
        super(OffenseClassifier, self).__init__()
        
        
        hidden_dim = min(1024, input_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)
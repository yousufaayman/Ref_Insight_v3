import torch
import torch.nn as nn

from models.encoder import VideoEncoder
from models.aggregation import AttentionAggregation, MeanPoolAggregation, MaxPoolAggregation
from models.classifier import FoulClassifier, OffenseClassifier
from config.config import Config

class VARS(nn.Module):

    
    def __init__(self, num_foul_types=8, num_offense_categories=4, 
                 pretrained=True, pooling_type='attention'):

        super(VARS, self).__init__()
        
        
        self.encoder = VideoEncoder(pretrained=pretrained)
        feature_dim = self.encoder.feature_dim
        
        
        if pooling_type == 'mean':
            self.aggregation = MeanPoolAggregation()
        elif pooling_type == 'max':
            self.aggregation = MaxPoolAggregation()
        else:  
            self.aggregation = AttentionAggregation(feature_dim)
        
        
        self.foul_classifier = FoulClassifier(feature_dim, num_foul_types)
        self.offense_classifier = OffenseClassifier(feature_dim, num_offense_categories)
    
    def forward(self, x):

        
        features = self.encoder(x)  
        
        
        aggregated, attention_scores = self.aggregation(features)  
        
        
        foul_logits = self.foul_classifier(aggregated)
        offense_logits = self.offense_classifier(aggregated)
        
        return {
            'foul_logits': foul_logits, 
            'offense_logits': offense_logits,
            'attention_scores': attention_scores,
            'features': features,
            'aggregated_features': aggregated
        }
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io.video import read_video
import torch.nn.functional as F

from data.data_loader import load_annotations, load_video_paths
from config.config import Config

class SoccerNetMVFoulDataset(Dataset):
    
    def __init__(self, root_dir, split='train', frames=16, resolution=224, num_views=None, temporal_shift_range=4):
        self.root_dir = root_dir
        self.split = split
        self.frames = frames
        self.resolution = resolution
        self.num_views = num_views
        self.temporal_shift_range = temporal_shift_range if split == 'train' else 0
        
        
        self.is_eval = split in ['val', 'test']
        
        if split != 'challenge':
            annotations = load_annotations(root_dir, split, num_views)
            
            self.offense_severity_labels = annotations['offense_severity_labels']
            self.action_labels = annotations['action_labels']
            self.offense_severity_distribution = annotations['offense_severity_distribution']
            self.action_distribution = annotations['action_distribution']
            self.included_action_ids = annotations['included_action_ids']
            
            
            self.offense_severity_weights = 1.0 / (self.offense_severity_distribution / len(self.offense_severity_labels))
            self.action_weights = 1.0 / (self.action_distribution / len(self.action_labels))
            
            
            self.offense_severity_weights = self.offense_severity_weights / self.offense_severity_weights.sum()
            self.action_weights = self.action_weights / self.action_weights.sum()
            
            self.video_paths = load_video_paths(root_dir, split, annotations['excluded_actions'], num_views)
        else:
            self.video_paths = load_video_paths(root_dir, split, [], num_views)
            self.included_action_ids = [str(i) for i in range(len(self.video_paths))]
        
        
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.video_paths)} samples for {split} split")
    
    def __len__(self):
        return len(self.video_paths)
    
    
    def get_class_weights(self):
        """Return weights for weighted loss calculation with proper handling of edge cases"""
        if self.split == 'challenge':
            return None, None
        
        
        offense_distribution = self.offense_severity_distribution.clone()
        
        
        min_count = 1.0  
        offense_distribution = torch.clamp(offense_distribution, min=min_count)
        
        
        offense_weights = 1.0 / offense_distribution
        
        
        if offense_weights.sum() > 0:
            offense_weights = offense_weights / offense_weights.sum()
        else:
            offense_weights = torch.ones_like(offense_weights) / len(offense_weights)
        
        
        action_distribution = self.action_distribution.clone()
        
        
        action_distribution = torch.clamp(action_distribution, min=min_count)
        
        
        action_weights = 1.0 / action_distribution
        
        
        if action_weights.sum() > 0:
            action_weights = action_weights / action_weights.sum()
        else:
            action_weights = torch.ones_like(action_weights) / len(action_weights)
        
        
        if torch.isnan(offense_weights).any():
            print("Warning: NaN values found in offense_weights, using uniform weights")
            offense_weights = torch.ones_like(offense_weights) / len(offense_weights)
        
        if torch.isnan(action_weights).any():
            print("Warning: NaN values found in action_weights, using uniform weights")
            action_weights = torch.ones_like(action_weights) / len(action_weights)
        
        return offense_weights, action_weights
    
    def __getitem__(self, index):
        videos = []
        views_to_use = []
        
        
        if self.split == 'train' and self.num_views is not None and len(self.video_paths[index]) > self.num_views:
            view_indices = random.sample(range(len(self.video_paths[index])), self.num_views)
            views_to_use = [self.video_paths[index][i] for i in view_indices]
        else:
            views_to_use = self.video_paths[index]
        
        
        for video_path in views_to_use:
            
            try:
                video_tensor, _, _ = read_video(video_path, pts_unit='sec')  
            except Exception as e:
                print(f"Error reading video {video_path}: {e}")
                
                video_tensor = torch.zeros(30, 720, 1280, 3, dtype=torch.uint8)
            
            
            center_frame = video_tensor.shape[0] // 2
            
            
            if self.split == 'train' and self.temporal_shift_range > 0:
                shift = random.randint(-self.temporal_shift_range, self.temporal_shift_range)
                center_frame = max(0, min(video_tensor.shape[0] - 1, center_frame + shift))
            
            
            half_frames = self.frames // 2
            start_frame = max(0, center_frame - half_frames)
            end_frame = min(video_tensor.shape[0], center_frame + half_frames)
            
            
            frames = video_tensor[start_frame:end_frame]
            
            
            if frames.shape[0] < self.frames:
                
                pad_size = self.frames - frames.shape[0]
                if pad_size > 0:
                    if frames.shape[0] > 0:  
                        padding = frames[[0]].repeat(pad_size, 1, 1, 1)
                        frames = torch.cat([padding, frames], dim=0)
                    else:
                        
                        frames = torch.zeros(self.frames, 720, 1280, 3, dtype=torch.uint8)
                
            
            frames = frames[:self.frames]
            
            
            frames = frames.float() / 255.0
            
            
            frames = frames.permute(0, 3, 1, 2)
            
            
            frames = F.interpolate(
                frames, size=(self.resolution, self.resolution), 
                mode='bilinear', align_corners=False
            )
            
            
            for i in range(frames.shape[0]):
                frames[i] = self.transform(frames[i])
            
            
            videos.append(frames)
        
        
        video_tensor = torch.stack(videos)  
        
        
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
        
        if self.split != 'challenge':
            return {
                'offense_severity_label': self.offense_severity_labels[index],
                'action_label': self.action_labels[index],
                'video': video_tensor,
                'action_id': self.included_action_ids[index]
            }
        else:
            
            return {
                'offense_severity_label': torch.zeros(4),
                'action_label': torch.zeros(8),
                'video': video_tensor,
                'action_id': self.included_action_ids[index]
            }

def create_data_loaders(config):
    
    train_num_views = 2  
    eval_num_views = None  
    
    
    if hasattr(config, 'NUM_VIEWS'):
        train_num_views = config.NUM_VIEWS
        eval_num_views = config.NUM_VIEWS
    
    
    train_dataset = SoccerNetMVFoulDataset(
        root_dir=config.DATA_ROOT,
        split='train',
        frames=config.FRAMES,
        resolution=config.RESOLUTION,
        num_views=train_num_views
    )
    
    
    val_dataset = SoccerNetMVFoulDataset(
        root_dir=config.DATA_ROOT,
        split='val',
        frames=config.FRAMES,
        resolution=config.RESOLUTION,
        num_views=eval_num_views  
    )
    
    
    test_dataset = SoccerNetMVFoulDataset(
        root_dir=config.DATA_ROOT,
        split='test',
        frames=config.FRAMES,
        resolution=config.RESOLUTION,
        num_views=eval_num_views  
    )
    
    
    offense_weights, action_weights = train_dataset.get_class_weights()
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'offense_weights': offense_weights,
        'action_weights': action_weights
    }
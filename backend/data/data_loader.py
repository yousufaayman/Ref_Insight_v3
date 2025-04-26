import os
import torch
import json
import numpy as np
from config.classes import EVENT_DICTIONARY, SEVERITY_MAPPING, NUM_FOUL_TYPES, NUM_OFFENSE_CATEGORIES

def load_annotations(folder_path, split, num_views=None):
    path_annotations = os.path.join(folder_path, split, "annotations.json")
    
    action_dict = EVENT_DICTIONARY['action_class']

    if not os.path.exists(path_annotations):
        raise FileNotFoundError(f"Annotations file not found: {path_annotations}")
    
    with open(path_annotations) as f:
        annotations_data = json.load(f)

    excluded_actions = []
    included_action_ids = []
    
    
    action_labels = []
    offense_severity_labels = []
    
    total_distribution = torch.zeros(NUM_OFFENSE_CATEGORIES, NUM_FOUL_TYPES)
    action_distribution = torch.zeros(NUM_FOUL_TYPES)
    offense_severity_distribution = torch.zeros(NUM_OFFENSE_CATEGORIES)

    for action_id, action_data in annotations_data['Actions'].items():
        action_class = action_data.get('Action class', '')
        offense_class = action_data.get('Offence', '')
        severity_class = action_data.get('Severity', '')
        
        
        if action_class == '' or action_class == 'Dont know':
            excluded_actions.append(action_id)
            continue
            
        if (offense_class == '' or offense_class == 'Between') and action_class != 'Dive/Simulation':
            excluded_actions.append(action_id)
            continue
            
        if (severity_class == '' or severity_class == '2.0' or severity_class == '4.0') and \
            action_class != 'Dive/Simulation' and offense_class not in ['No offence', 'No Offence']:
            excluded_actions.append(action_id)
            continue
        
        
        if offense_class == '' or offense_class == 'Between':
            offense_class = 'Offence'
            
        if severity_class == '' or severity_class == '2.0' or severity_class == '4.0':
            severity_class = '1.0'
        
        
        if num_views == 1:  
            for i in range(len(action_data.get('Clips', []))):
                offense_severity_idx = process_offense_severity(offense_class, severity_class)
                
                offense_severity_tensor = torch.zeros(NUM_OFFENSE_CATEGORIES)
                offense_severity_tensor[offense_severity_idx] = 1
                offense_severity_labels.append(offense_severity_tensor)
                offense_severity_distribution[offense_severity_idx] += 1
                
                action_idx = action_dict.get(action_class, 0)
                action_tensor = torch.zeros(NUM_FOUL_TYPES)
                action_tensor[action_idx] = 1
                action_labels.append(action_tensor)
                action_distribution[action_idx] += 1
                
                total_distribution[offense_severity_idx][action_idx] += 1
        else:  
            offense_severity_idx = process_offense_severity(offense_class, severity_class)
            
            offense_severity_tensor = torch.zeros(NUM_OFFENSE_CATEGORIES)
            offense_severity_tensor[offense_severity_idx] = 1
            offense_severity_labels.append(offense_severity_tensor)
            offense_severity_distribution[offense_severity_idx] += 1
            
            action_idx = action_dict.get(action_class, 0)
            action_tensor = torch.zeros(NUM_FOUL_TYPES)
            action_tensor[action_idx] = 1
            action_labels.append(action_tensor)
            action_distribution[action_idx] += 1
            
            included_action_ids.append(action_id)
            total_distribution[offense_severity_idx][action_idx] += 1
    
    return {
        'offense_severity_labels': offense_severity_labels,
        'action_labels': action_labels,
        'offense_severity_distribution': offense_severity_distribution,
        'action_distribution': action_distribution,
        'excluded_actions': excluded_actions,
        'included_action_ids': included_action_ids,
        'total_distribution': total_distribution
    }


def process_offense_severity(offense_class, severity_class):
    """Helper function to get offense severity index"""
    if offense_class in ['No Offence', 'No offence']:
        return 0
    elif offense_class == 'Offence':
        if severity_class == '1.0':
            return 1
        elif severity_class == '3.0':
            return 2
        elif severity_class == '5.0':
            return 3
    return 0  


def load_video_paths(folder_path, split, excluded_actions=None, num_views=None):
    if excluded_actions is None:
        excluded_actions = []
        
    path_clips = os.path.join(folder_path, split)
    
    if not os.path.exists(path_clips):
        raise FileNotFoundError(f"Clips directory not found: {path_clips}")
    
    
    folders = 0
    for _, dirnames, _ in os.walk(path_clips):
        folders += len(dirnames)
        break  
    
    clip_paths = []
    
    for i in range(folders):
        action_id = str(i)
        
        if action_id in excluded_actions:
            continue
        
        action_path = os.path.join(path_clips, f"action_{action_id}")
        
        if num_views == 1:  
            for view_idx in range(4):  
                clip_path = os.path.join(action_path, f"clip_{view_idx}.mp4")
                
                if os.path.exists(clip_path):
                    clip_paths.append([clip_path])
        else:  
            views = []
            
            for view_idx in range(4):  
                clip_path = os.path.join(action_path, f"clip_{view_idx}.mp4")
                
                if os.path.exists(clip_path):
                    views.append(clip_path)
            
            if views:
                clip_paths.append(views)
    
    return clip_paths
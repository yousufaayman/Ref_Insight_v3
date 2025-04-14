import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_accuracy(outputs, targets):
    """
    Calculate classification accuracy
    
    Args:
        outputs: Predicted logits tensor of shape [batch_size, num_classes]
        targets: One-hot encoded target tensor of shape [batch_size, num_classes]
        
    Returns:
        float: Accuracy in range [0, 1]
    """
    _, pred_indices = torch.max(outputs, 1)
    _, target_indices = torch.max(targets, 1)
    correct = (pred_indices == target_indices).float().sum()
    return correct.item() / targets.size(0)


def calculate_balanced_accuracy(outputs, targets):
    _, pred_indices = torch.max(outputs, 1)
    _, target_indices = torch.max(targets, 1)
    
    
    y_true = target_indices.cpu().numpy()
    y_pred = pred_indices.cpu().numpy()
    
    
    cm = confusion_matrix(y_true, y_pred)
    
    
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    
    
    return np.mean(per_class_acc)


def calculate_metrics(foul_outputs, foul_targets, offense_outputs, offense_targets):
    metrics = {}
    
    
    metrics['foul_accuracy'] = calculate_accuracy(foul_outputs, foul_targets)
    metrics['offense_accuracy'] = calculate_accuracy(offense_outputs, offense_targets)
    
    
    metrics['foul_balanced_accuracy'] = calculate_balanced_accuracy(foul_outputs, foul_targets)
    metrics['offense_balanced_accuracy'] = calculate_balanced_accuracy(offense_outputs, offense_targets)
    
    
    metrics['combined_accuracy'] = (metrics['foul_accuracy'] + metrics['offense_accuracy']) / 2
    metrics['combined_balanced_accuracy'] = (metrics['foul_balanced_accuracy'] + metrics['offense_balanced_accuracy']) / 2
    
    return metrics


def log_metrics(metrics, step, mode='train', logger=None):
    
    print(f"{mode.capitalize()} Metrics - Step {step}:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    
    if logger is not None:
        for name, value in metrics.items():
            logger.add_scalar(f"{mode}/{name}", value, step)
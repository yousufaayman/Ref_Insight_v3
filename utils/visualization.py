import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from config.classes import EVENT_DICTIONARY

def visualize_attention_scores(video_tensor, attention_scores, save_path=None):
    num_views = video_tensor.shape[0]
    
    
    fig, axes = plt.subplots(1, num_views, figsize=(4*num_views, 4))
    if num_views == 1:
        axes = [axes]
    
    
    attention_scores = attention_scores.cpu().numpy()
    attention_normalized = (attention_scores / attention_scores.max()) * 100
    
    
    for i in range(num_views):
        
        center_frame_idx = video_tensor.shape[2] // 2
        frame = video_tensor[i, :, center_frame_idx].permute(1, 2, 0).cpu().numpy()
        
        
        frame = frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        frame = np.clip(frame, 0, 1)
        
        
        axes[i].imshow(frame)
        axes[i].axis('off')
        axes[i].set_title(f"View {i+1}\nAttention: {attention_normalized[i]:.1f}%")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(all_targets, all_preds, class_names, title, save_path=None):
    
    cm = confusion_matrix(all_targets, all_preds)
    
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  
    
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def visualize_multi_view_prediction(video_tensor, attention_scores, foul_probs, offense_probs, 
                                    true_foul=None, true_offense=None, save_path=None):
    num_views = video_tensor.shape[0]
    
    
    foul_names = list(EVENT_DICTIONARY['action_class'].keys())
    offense_names = list(EVENT_DICTIONARY['offense_severity'].keys())
    
    
    attention_scores = attention_scores.cpu().numpy()
    foul_probs = torch.softmax(foul_probs, dim=0).cpu().numpy()
    offense_probs = torch.softmax(offense_probs, dim=0).cpu().numpy()
    
    
    pred_foul = np.argmax(foul_probs)
    pred_offense = np.argmax(offense_probs)
    
    
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, num_views + 1)
    
    
    for i in range(num_views):
        
        center_frame_idx = video_tensor.shape[2] // 2
        frame = video_tensor[i, :, center_frame_idx].permute(1, 2, 0).cpu().numpy()
        
        
        frame = frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        frame = np.clip(frame, 0, 1)
        
        
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frame)
        ax.axis('off')
        
        
        att_score = attention_scores[i] / np.sum(attention_scores) * 100
        ax.set_title(f"View {i+1}\nAttention: {att_score:.1f}%")
    
    
    ax_foul = fig.add_subplot(gs[0, -1])
    y_pos = np.arange(len(foul_names))
    ax_foul.barh(y_pos, foul_probs, align='center')
    ax_foul.set_yticks(y_pos)
    ax_foul.set_yticklabels(foul_names)
    ax_foul.invert_yaxis()  
    ax_foul.set_xlabel('Probability')
    ax_foul.set_title('Foul Type Classification')
    
    
    ax_foul.get_children()[pred_foul].set_color('red')
    if true_foul is not None:
        ax_foul.axhline(y=true_foul, color='green', linestyle='--', alpha=0.7)
    
    
    ax_offense = fig.add_subplot(gs[1, :])
    y_pos = np.arange(len(offense_names))
    ax_offense.barh(y_pos, offense_probs, align='center')
    ax_offense.set_yticks(y_pos)
    ax_offense.set_yticklabels(offense_names)
    ax_offense.invert_yaxis()  
    ax_offense.set_xlabel('Probability')
    ax_offense.set_title('Offense Severity Classification')
    
    
    ax_offense.get_children()[pred_offense].set_color('red')
    if true_offense is not None:
        ax_offense.axhline(y=true_offense, color='green', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
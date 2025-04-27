import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from config.config import Config
from data.dataset import create_data_loaders
from models.vars import VARS
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, visualize_multi_view_prediction
from config.classes import EVENT_DICTIONARY

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VARS model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, help='Path to dataset root')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--pooling', type=str, default='attention', 
                      choices=['mean', 'max', 'attention'], help='Pooling method')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--num_visualizations', type=int, default=10, help='Number of predictions to visualize')
    
    return parser.parse_args()

def evaluate(model, data_loader, device, save_dir=None, visualize=False, num_visualizations=10):
    model.eval()
    
    all_metrics = []
    all_foul_targets = []
    all_foul_preds = []
    all_offense_targets = []
    all_offense_preds = []
    action_ids = []
    
    
    visualizations_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            
            offense_targets = batch['offense_severity_label'].to(device)
            action_targets = batch['action_label'].to(device)
            videos = batch['video'].to(device)
            action_id = batch['action_id']
            
            
            outputs = model(videos)
            foul_logits = outputs['foul_logits']
            offense_logits = outputs['offense_logits']
            attention_scores = outputs['attention_scores']
            
            
            _, foul_preds = torch.max(foul_logits, 1)
            _, offense_preds = torch.max(offense_logits, 1)
            _, foul_targets_idx = torch.max(action_targets, 1)
            _, offense_targets_idx = torch.max(offense_targets, 1)
            
            
            batch_metrics = calculate_metrics(foul_logits, action_targets, offense_logits, offense_targets)
            all_metrics.append(batch_metrics)
            
            
            all_foul_targets.extend(foul_targets_idx.cpu().numpy())
            all_foul_preds.extend(foul_preds.cpu().numpy())
            all_offense_targets.extend(offense_targets_idx.cpu().numpy())
            all_offense_preds.extend(offense_preds.cpu().numpy())
            action_ids.extend(action_id)
            
            
            if visualize and save_dir and visualizations_count < num_visualizations:
                if batch_metrics['combined_accuracy'] > 0.5:  
                    vis_dir = os.path.join(save_dir, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    vis_path = os.path.join(vis_dir, f'prediction_{action_id[0]}.png')
                    
                    visualize_multi_view_prediction(
                        videos[0].cpu(),  
                        attention_scores[0].cpu(),
                        foul_logits[0].cpu(),
                        offense_logits[0].cpu(),
                        true_foul=foul_targets_idx[0].item(),
                        true_offense=offense_targets_idx[0].item(),
                        save_path=vis_path
                    )
                    
                    visualizations_count += 1
    
    
    agg_metrics = {}
    for key in all_metrics[0].keys():
        agg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        
        foul_names = list(EVENT_DICTIONARY['action_class'].keys())
        plot_confusion_matrix(
            all_foul_targets, all_foul_preds, foul_names,
            'Foul Type Confusion Matrix',
            os.path.join(save_dir, 'foul_confusion_matrix.png')
        )
        
        
        offense_names = list(EVENT_DICTIONARY['offense_severity'].keys())
        plot_confusion_matrix(
            all_offense_targets, all_offense_preds, offense_names,
            'Offense Severity Confusion Matrix',
            os.path.join(save_dir, 'offense_confusion_matrix.png')
        )
        
        
        with open(os.path.join(save_dir, 'evaluation_metrics.txt'), 'w') as f:
            for key, value in agg_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
    
    return agg_metrics, all_foul_targets, all_foul_preds, all_offense_targets, all_offense_preds, action_ids

def main():
    args = parse_args()
    
    
    if args.data_root:
        Config.DATA_ROOT = args.data_root
    
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    data_loaders = create_data_loaders(Config)
    eval_loader = data_loaders[args.split]
    
    
    model = VARS(
        num_foul_types=8, 
        num_offense_categories=4,
        pretrained=False,  
        pooling_type=args.pooling
    )
    
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from {args.checkpoint}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    
    
    metrics, foul_targets, foul_preds, offense_targets, offense_preds, action_ids = evaluate(
        model, eval_loader, device, args.save_dir, args.visualize, args.num_visualizations
    )
    
    
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    
    if args.save_dir:
        results_path = os.path.join(args.save_dir, 'detailed_results.csv')
        with open(results_path, 'w') as f:
            f.write("action_id,true_foul,pred_foul,true_offense,pred_offense\n")
            
            foul_names = list(EVENT_DICTIONARY['action_class'].keys())
            offense_names = list(EVENT_DICTIONARY['offense_severity'].keys())
            
            for i in range(len(action_ids)):
                true_foul_name = foul_names[foul_targets[i]]
                pred_foul_name = foul_names[foul_preds[i]]
                true_offense_name = offense_names[offense_targets[i]]
                pred_offense_name = offense_names[offense_preds[i]]
                
                f.write(f"{action_ids[i]},{true_foul_name},{pred_foul_name},{true_offense_name},{pred_offense_name}\n")
        
        print(f"Detailed results saved to {results_path}")

if __name__ == "__main__":
    main()
import os
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from data.dataset import create_data_loaders
from models.vars import VARS
from utils.metrics import calculate_metrics, log_metrics

def create_loss_functions(offense_weights, action_weights, device):
    
    if action_weights is not None and not torch.isnan(action_weights).any() and not (action_weights == 0).all():
        
        valid_indices = torch.where(action_weights > 0)[0]
        if len(valid_indices) > 0:
            valid_weights = action_weights[valid_indices]
            valid_weights = valid_weights / valid_weights.sum()
            
            
            valid_weights = torch.sqrt(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()
            
            print(f"Using weighted loss for actions with weights: {valid_weights}")
            
            full_weights = torch.ones(action_weights.size(0), device=device)
            full_weights[valid_indices] = valid_weights
            action_criterion = nn.CrossEntropyLoss(weight=full_weights)
        else:
            print("Warning: No valid action weights found, using unweighted loss")
            action_criterion = nn.CrossEntropyLoss()
    else:
        print("Warning: Invalid action weights, using unweighted loss")
        action_criterion = nn.CrossEntropyLoss()
    
    
    if offense_weights is not None and not torch.isnan(offense_weights).any() and not (offense_weights == 0).all():
        
        offense_weights = offense_weights / offense_weights.sum()
        
        
        offense_weights = torch.sqrt(offense_weights)
        offense_weights = offense_weights / offense_weights.sum()
        
        print(f"Using weighted loss for offense with weights: {offense_weights}")
        offense_criterion = nn.CrossEntropyLoss(weight=offense_weights)
    else:
        print("Warning: Invalid offense weights, using unweighted loss")
        offense_criterion = nn.CrossEntropyLoss()
    
    return action_criterion, offense_criterion

def save_checkpoint(path, epoch, model, optimizer, scheduler, train_loss, val_loss, train_metrics, val_metrics, config):    
    
    processed_train_metrics = {}
    processed_val_metrics = {}
    
    for key, value in train_metrics.items():
        if hasattr(value, 'item'):  
            processed_train_metrics[key] = value.item() if hasattr(value, 'item') else float(value)
        else:
            processed_train_metrics[key] = value
            
    for key, value in val_metrics.items():
        if hasattr(value, 'item'):  
            processed_val_metrics[key] = value.item() if hasattr(value, 'item') else float(value)
        else:
            processed_val_metrics[key] = value
    
    
    serializable_config = {}
    for k, v in config.__dict__.items():
        if not k.startswith('__') and not callable(v) and not isinstance(v, staticmethod):
            serializable_config[k] = v
    
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_metrics': processed_train_metrics,
        'val_metrics': processed_val_metrics,
        'config': serializable_config
    }
    
    
    try:
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, path)
            print(f"Fallback: Model state dict saved to {path}")
            return True
        except Exception as e2:
            print(f"Critical error: Could not save model: {e2}")
            return False

def parse_args():
    parser = argparse.ArgumentParser(description='Train VARS model')
    parser.add_argument('--data_root', type=str, help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--pooling', type=str, default='attention', 
                      choices=['mean', 'max', 'attention'], help='Pooling method')
    parser.add_argument('--frames', type=int, default=Config.FRAMES, help='Number of frames per clip')
    parser.add_argument('--resolution', type=int, default=Config.RESOLUTION, help='Frame resolution')
    parser.add_argument('--pretrained', action='store_true', default=True, 
                       help='Use pretrained MViTv2 weights')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                       help='Do not use pretrained weights')
    
    return parser.parse_args()

def train_one_epoch(model, train_loader, optimizer, criterions, device, epoch):
    """
    Train the model for one epoch with improved training strategy
    
    Args:
        model: the VARS model
        train_loader: data loader for training data
        optimizer: optimizer
        criterions: loss functions for foul and offense classification
        device: torch device
        epoch: current epoch number
    
    Returns:
        epoch_loss: average loss for the epoch
        epoch_metrics: evaluation metrics for the epoch
        epoch_time: time taken for the epoch
    """
    model.train()
    
    
    action_criterion, offense_criterion = criterions
    
    epoch_loss = 0
    epoch_metrics = {
        'foul_accuracy': 0,
        'offense_accuracy': 0,
        'foul_balanced_accuracy': 0,
        'offense_balanced_accuracy': 0,
        'combined_accuracy': 0,
        'combined_balanced_accuracy': 0
    }
    
    valid_batches = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        
        offense_targets = batch['offense_severity_label'].to(device)
        action_targets = batch['action_label'].to(device)
        videos = batch['video'].to(device)
        
        
        outputs = model(videos)
        foul_logits = outputs['foul_logits']
        offense_logits = outputs['offense_logits']
        
        try:
            
            action_loss = action_criterion(foul_logits, action_targets)
            offense_loss = offense_criterion(offense_logits, offense_targets)
            
            
            if torch.isnan(action_loss) or torch.isinf(action_loss):
                action_loss = torch.tensor(0.0, device=device)
                print(f"Warning: NaN/Inf in action_loss at batch {batch_idx}")
            
            if torch.isnan(offense_loss) or torch.isinf(offense_loss):
                offense_loss = torch.tensor(0.0, device=device)
                print(f"Warning: NaN/Inf in offense_loss at batch {batch_idx}")
            
            
            
            
            if epoch < 5:
                action_weight = 1.5  
                offense_weight = 0.5
            else:
                action_weight = 1.0
                offense_weight = 1.0
                
            
            total_loss = action_weight * action_loss + offense_weight * offense_loss
            
            
            if not torch.isnan(total_loss) and not torch.isinf(total_loss) and total_loss > 0:
                
                optimizer.zero_grad()
                
                
                total_loss.backward()
                
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                
                optimizer.step()
                
                
                batch_metrics = calculate_metrics(foul_logits, action_targets, offense_logits, offense_targets)
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
                
                epoch_loss += total_loss.item()
                valid_batches += 1
                
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {total_loss.item():.4f} | "
                          f"Foul Acc: {batch_metrics['foul_accuracy']:.4f} | "
                          f"Offense Acc: {batch_metrics['offense_accuracy']:.4f}")
            else:
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: NaN/Inf (skipped)")
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    
    if valid_batches > 0:
        epoch_loss /= valid_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= valid_batches
    else:
        epoch_loss = float('nan')
        for key in epoch_metrics:
            epoch_metrics[key] = 0
    
    epoch_time = time.time() - start_time
    
    return epoch_loss, epoch_metrics, epoch_time

def validate(model, val_loader, criterions, device, epoch):
    """Validate model"""
    model.eval()
    
    
    action_criterion, offense_criterion = criterions
    
    val_loss = 0
    val_metrics = {
        'foul_accuracy': 0,
        'offense_accuracy': 0,
        'foul_balanced_accuracy': 0,
        'offense_balanced_accuracy': 0,
        'combined_accuracy': 0,
        'combined_balanced_accuracy': 0
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            
            offense_targets = batch['offense_severity_label'].to(device)
            action_targets = batch['action_label'].to(device)
            videos = batch['video'].to(device)
            
            
            outputs = model(videos)
            foul_logits = outputs['foul_logits']
            offense_logits = outputs['offense_logits']
            
            
            action_loss = action_criterion(foul_logits, action_targets)
            offense_loss = offense_criterion(offense_logits, offense_targets)
            total_loss = action_loss + offense_loss
            
            
            batch_metrics = calculate_metrics(foul_logits, action_targets, offense_logits, offense_targets)
            for key in val_metrics:
                val_metrics[key] += batch_metrics[key]
            
            val_loss += total_loss.item()
    
    
    val_loss /= len(val_loader)
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)
    
    return val_loss, val_metrics

def main():
    args = parse_args()
    
    
    if args.data_root:
        Config.DATA_ROOT = args.data_root
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.frames:
        Config.FRAMES = args.frames
    if args.resolution:
        Config.RESOLUTION = args.resolution

    Config.create_dirs()
    
    
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    data_loaders = create_data_loaders(Config)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    
    offense_weights = data_loaders.get('offense_weights', None)
    action_weights = data_loaders.get('action_weights', None)
    
    if offense_weights is not None and action_weights is not None:
        offense_weights = offense_weights.to(device)
        action_weights = action_weights.to(device)
        print(f"Using weighted loss with offense weights: {offense_weights}")
        print(f"Using weighted loss with action weights: {action_weights}")
    
    
    model = VARS(
        num_foul_types=8, 
        num_offense_categories=4,
        pretrained=args.pretrained,
        pooling_type=args.pooling
    )
    model = model.to(device)
    
    
    print(f"Model created with MViTv2-small backbone and {args.pooling} pooling")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    

    criterions = create_loss_functions(offense_weights, action_weights, device)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    
    
    writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, Config.EXPERIMENT_NAME))
    
    
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(Config.NUM_EPOCHS):
        
        train_loss, train_metrics, epoch_time = train_one_epoch(
            model, train_loader, optimizer, criterions, device, epoch
        )
        
        
        val_loss, val_metrics = validate(model, val_loader, criterions, device, epoch)
        
        
        scheduler.step()
        
        
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Val Metrics: {val_metrics}")
        
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        for key, value in train_metrics.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        
        val_combined_acc = val_metrics['combined_accuracy']
        if val_combined_acc > best_val_acc:
            best_val_acc = val_combined_acc
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, 
                Config.EXPERIMENT_NAME, 
                f"best_model_epoch{epoch+1}_acc{val_combined_acc:.4f}.pth"
            )
            save_checkpoint(
                checkpoint_path, epoch, model, optimizer, scheduler,
                train_loss, val_loss, train_metrics, val_metrics, Config
            )

        
        latest_path = os.path.join(
            Config.CHECKPOINT_DIR, 
            Config.EXPERIMENT_NAME, 
            "latest_model.pth"
        )
        save_checkpoint(
            latest_path, epoch, model, optimizer, scheduler,
            train_loss, val_loss, train_metrics, val_metrics, Config
        )
        
        print(f"Latest model saved to {latest_path}")
        print("-" * 80)
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()
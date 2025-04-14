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
    """Train model for one epoch"""
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
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        
        offense_targets = batch['offense_severity_label'].to(device)
        action_targets = batch['action_label'].to(device)
        videos = batch['video'].to(device)
        
        
        outputs = model(videos)
        foul_logits = outputs['foul_logits']
        offense_logits = outputs['offense_logits']
        
        
        action_loss = action_criterion(foul_logits, action_targets)
        offense_loss = offense_criterion(offense_logits, offense_targets)
        total_loss = action_loss + offense_loss
        
        
        optimizer.zero_grad()
        total_loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        
        batch_metrics = calculate_metrics(foul_logits, action_targets, offense_logits, offense_targets)
        for key in epoch_metrics:
            epoch_metrics[key] += batch_metrics[key]
        
        epoch_loss += total_loss.item()
        
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {total_loss.item():.4f} | "
                  f"Foul Acc: {batch_metrics['foul_accuracy']:.4f} | "
                  f"Offense Acc: {batch_metrics['offense_accuracy']:.4f}")
    
    
    epoch_loss /= len(train_loader)
    for key in epoch_metrics:
        epoch_metrics[key] /= len(train_loader)
    
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
    
    
    if offense_weights is not None and action_weights is not None:
        action_criterion = nn.CrossEntropyLoss(weight=action_weights)
        offense_criterion = nn.CrossEntropyLoss(weight=offense_weights)
    else:
        action_criterion = nn.CrossEntropyLoss()
        offense_criterion = nn.CrossEntropyLoss()
    
    criterions = (action_criterion, offense_criterion)
    
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
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': {k: v for k, v in vars(Config).items() if not k.startswith('__')}
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
        
        latest_path = os.path.join(
            Config.CHECKPOINT_DIR, 
            Config.EXPERIMENT_NAME, 
            "latest_model.pth"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': {k: v for k, v in vars(Config).items() if not k.startswith('__')}
        }, latest_path)
        
        print(f"Latest model saved to {latest_path}")
        print("-" * 80)
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()
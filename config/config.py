import os
from datetime import datetime

class Config:
    
    DATA_ROOT = "/path/to/soccernet_mvfoul_dataset"  
    FRAMES = 16  
    RESOLUTION = 224  
    FPS = 16  
    TEMPORAL_SHIFT_RANGE = 4  
    
    
    PRETRAINED = True  
    MODEL_VARIANT = "small"  
    FEATURE_DIM = 768  
    
    
    BATCH_SIZE = 6  
    LEARNING_RATE = 5e-5  
    WEIGHT_DECAY = 1e-4  
    LR_STEP_SIZE = 3  
    LR_GAMMA = 0.3  
    NUM_EPOCHS = 7  
    
    
    DEVICE = "cuda"  
    NUM_WORKERS = 4  
    
    
    EXPERIMENT_NAME = f"VARS_MViTv2Small_K700_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    
    @staticmethod
    def create_dirs():
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.join(Config.CHECKPOINT_DIR, Config.EXPERIMENT_NAME), exist_ok=True)
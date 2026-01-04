import torch
import os

# Paths
DATA_DIR = "/content/isic_workspace"
MODEL_SAVE_PATH = "/content/drive/MyDrive/ISIC_Model"
CHECKPOINT_PATH = "/content/drive/MyDrive/ISIC_Checkpoint"
LOG_PATH = "/content/drive/MyDrive/ISIC_Log_Path"
GENERATED_IMAGES_PATH = "/content/drive/MyDrive/ISIC_generative_images"

# Hyperparameters
lr = 0.0002
beta1 = 0.5
num_epochs = 50
batch_size = 32
image_size = 256
nz = 100
ngf = 32
ndf = 32
nc = 3

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

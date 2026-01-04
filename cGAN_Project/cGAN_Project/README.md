# ISIC 2019 Conditional GAN Project

## Project Overview
This project implements a Conditional Generative Adversarial Network (cGAN) to synthesize skin lesion images based on the ISIC 2019 dataset. The model generates 256x256 images conditioned on diagnosis, age, sex, and anatomical site.

## Usage
1. **Train the Model**:
   Run the trainer script to start training the cGAN.
   ```bash
   python trainer.py
   ```

2. **Generate Synthetic Data**:
   Run the generation script to create a balanced dataset of synthetic images and metadata.
   ```bash
   python generate_balanced_metadata.py
   ```

## Outputs
By default (configured in `config.py`), outputs are saved to Google Drive:
- **Trained Models**: `/content/drive/MyDrive/ISIC_Model`
- **Checkpoints**: `/content/drive/MyDrive/ISIC_Checkpoint`
- **Training Logs**: `/content/drive/MyDrive/ISIC_Log_Path`
- **Generated Images**: `/content/drive/MyDrive/ISIC_generative_images`
- **Balanced Synthetic Dataset**: `/content/drive/MyDrive/ISIC_synthetic_balanced`

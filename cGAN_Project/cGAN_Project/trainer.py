import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms
import config
from utils import ISICDataset, weights_init
from generator import Generator
from discriminator import Discriminator

def main():
    # ================= Data Preprocessing =================
    print("Loading Data...")
    meta_path = os.path.join(config.DATA_DIR, "ISIC_2019_Training_Metadata.csv")
    gt_path = os.path.join(config.DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")
    img_dir = os.path.join(config.DATA_DIR, "ISIC_2019_Training_Input")

    df_meta = pd.read_csv(meta_path)
    df_gt = pd.read_csv(gt_path)
    df = pd.merge(df_meta, df_gt, on='image')

    # Handle missing values
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].mean())
    df['anatom_site_general'] = df['anatom_site_general'].fillna('unknown')

    # One-Hot Encoding
    site_dummies = pd.get_dummies(df['anatom_site_general'], prefix='site')
    df = pd.concat([df, site_dummies], axis=1)

    # Normalize age
    df['age_approx'] = df['age_approx'] / df['age_approx'].max()

    # Define Condition Columns
    label_cols = [col for col in df_gt.columns if col != 'image']
    meta_cols = ['age_approx'] + list(site_dummies.columns)
    condition_cols = label_cols + meta_cols
    n_conditions = len(condition_cols)
    print(f"Conditions ({n_conditions}): {condition_cols}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset & DataLoader
    dataset = ISICDataset(df, img_dir, condition_cols, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    # ================= Model Initialization =================
    print("Initializing Models...")
    netG = Generator(config.nz, n_conditions, config.ngf, config.nc).to(config.device)
    netG.apply(weights_init)

    netD = Discriminator(config.nc, n_conditions, config.ndf).to(config.device)
    netD.apply(weights_init)

    # Optimizers & Loss
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # Fixed Noise for visualization
    fixed_noise = torch.randn(config.batch_size, config.nz, device=config.device)
    
    # Get fixed conditions from the first batch
    _, real_batch_conditions = next(iter(dataloader))
    fixed_conditions = real_batch_conditions.to(config.device)

    # ================= Training Loop =================
    print("Starting Training...")
    G_losses = []
    D_losses = []

    for epoch in range(config.num_epochs):
        for i, (data, conditions) in enumerate(dataloader):
            # (1) Update D network
            netD.zero_grad()
            real_cpu = data.to(config.device)
            cond_data = conditions.to(config.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1.0, dtype=torch.float, device=config.device)

            output = netD(real_cpu, cond_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, config.nz, device=config.device)
            fake = netG(noise, cond_data)
            label.fill_(0.0)
            output = netD(fake.detach(), cond_data).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network
            netG.zero_grad()
            label.fill_(1.0)
            output = netD(fake, cond_data).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if i % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, config.num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save generated images
        with torch.no_grad():
            fake = netG(fixed_noise, fixed_conditions).detach().cpu()
        vutils.save_image(fake, f"{config.GENERATED_IMAGES_PATH}/fake_samples_epoch_{epoch}.png", normalize=True)

        # Save Checkpoints
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.num_epochs:
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
            }, f"{config.CHECKPOINT_PATH}/checkpoint_epoch_{epoch}.pth")

    # Save Final Model
    torch.save(netG.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'ISIC_generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'ISIC_discriminator_final.pth'))
    print("Training Complete and Models Saved.")

if __name__ == '__main__':
    main()

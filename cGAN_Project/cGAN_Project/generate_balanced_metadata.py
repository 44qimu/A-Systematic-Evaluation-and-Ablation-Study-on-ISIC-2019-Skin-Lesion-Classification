import os
import torch
import pandas as pd
import numpy as np
import torchvision.utils as vutils
import config
from generator import Generator

def main():
    # Define output directory
    output_dir = "/content/drive/MyDrive/ISIC_synthetic_balanced"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # ================= Data Preprocessing (Must match Training) =================
    print("Loading Data...")
    meta_path = os.path.join(config.DATA_DIR, "ISIC_2019_Training_Metadata.csv")
    gt_path = os.path.join(config.DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")
    
    if not os.path.exists(meta_path) or not os.path.exists(gt_path):
        print("Data not found. Please ensure data is in config.DATA_DIR")
        return

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

    # ================= Inverse Frequency Sampling =================
    print("Calculating class weights...")
    class_counts = df[label_cols].sum()
    valid_classes = class_counts[class_counts > 0].index.tolist()
    valid_counts = class_counts[valid_classes]
    weights = 1.0 / valid_counts
    probs = weights / weights.sum()
    
    num_samples = 5000
    print(f"Sampling {num_samples} conditions...")
    sampled_class_names = np.random.choice(valid_classes, size=num_samples, p=probs)

    # Prepare Conditions
    synthetic_conditions_list = []
    synthetic_metadata_list = []

    for cls_name in sampled_class_names:
        real_samples = df[df[cls_name] == 1.0]
        if len(real_samples) > 0:
            random_row = real_samples.sample(1)
            cond_values = random_row[condition_cols].values[0].astype('float32')
            synthetic_conditions_list.append(cond_values)
            
            meta_entry = {
                'diagnosis': cls_name,
                'age_approx': random_row['age_approx'].values[0],
                'anatom_site_general': random_row['anatom_site_general'].values[0],
                'sex': random_row['sex'].values[0]
            }
            synthetic_metadata_list.append(meta_entry)

    synthetic_conditions = torch.tensor(np.array(synthetic_conditions_list)).to(config.device)

    # ================= Generate Images =================
    print("Loading Generator...")
    netG = Generator(config.nz, n_conditions, config.ngf, config.nc).to(config.device)
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'ISIC_generator_final.pth')
    
    if os.path.exists(model_path):
        netG.load_state_dict(torch.load(model_path, map_location=config.device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}")
        return

    netG.eval()
    print("Generating images...")
    
    with torch.no_grad():
        for i in range(0, num_samples, config.batch_size):
            batch_conds = synthetic_conditions[i : i + config.batch_size]
            current_batch_size = batch_conds.size(0)
            noise = torch.randn(current_batch_size, config.nz, device=config.device)
            
            fake_images = netG(noise, batch_conds)
            
            for j, img in enumerate(fake_images):
                global_idx = i + j
                if global_idx >= num_samples: break
                
                cls_label = sampled_class_names[global_idx]
                filename = f"syn_img_{global_idx:05d}_{cls_label}.png"
                save_path = os.path.join(output_dir, filename)
                
                vutils.save_image(img, save_path, normalize=True)
                synthetic_metadata_list[global_idx]['image'] = filename.replace('.png', '')
            
            if (i + config.batch_size) % 1000 < config.batch_size:
                 print(f"Generated {min(i + config.batch_size, num_samples)} / {num_samples}")

    # ================= Save Metadata =================
    df_syn = pd.DataFrame(synthetic_metadata_list)
    cols_order = ['image', 'diagnosis', 'age_approx', 'anatom_site_general', 'sex']
    df_syn = df_syn[cols_order]
    
    csv_path = os.path.join(output_dir, 'synthetic_metadata.csv')
    df_syn.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")
    print("Generation Complete.")

if __name__ == '__main__':
    main()

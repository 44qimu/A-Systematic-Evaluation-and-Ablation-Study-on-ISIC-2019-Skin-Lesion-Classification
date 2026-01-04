import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

def weights_init(m):
    """
    Custom weights initialization called on netG and netD.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ISICDataset(Dataset):
    """
    Custom Dataset for ISIC 2019 images and conditions.
    """
    def __init__(self, dataframe, image_dir, condition_cols, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.condition_cols = condition_cols

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image name and path
        img_name = self.dataframe.iloc[idx]['image']
        img_path = os.path.join(self.image_dir, img_name + '.jpg')

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get conditions
        # Ensure we convert boolean columns to float if any exist
        conditions = self.dataframe.iloc[idx][self.condition_cols].values.astype('float32')
        conditions = torch.tensor(conditions)

        return image, conditions

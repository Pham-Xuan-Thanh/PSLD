import os
import io
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
from util import data_scaler, register_module
import torchvision.transforms as transforms


@register_module(category="datasets", name="fcvn")
class FCVNDataset(Dataset):
    """Implementation of the CelebA dataset.
    Downloaded from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """
    def __init__(self, root, norm=True, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform
        self.norm = norm

        self.images = []

        for img in tqdm(os.listdir(root)):
            self.images.append(os.path.join(self.root, img))

        # Subsample the dataset (if enabled)
        if subsample_size is not None:
            self.images = self.images[:subsample_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)

        # Normalize
        img = data_scaler(img, norm=self.norm)

        # TODO: Add the functionality to return labels to enable
        # guidance-based generation.
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.images)






# class FCVNDataset(Dataset):
#     def __init__(
#         self,
#         parquet_file,
#         norm=True,
#         transform=None,
#         subsample_size=None,
#         **kwargs,
#     ):
#         if not os.path.exists(parquet_file):
#             raise ValueError(f"The specified parquet file: {parquet_file} does not exist")
        
#         if subsample_size is not None:
#             assert isinstance(subsample_size, int)
        
#         self.parquet_file = parquet_file
#         self.norm = norm
#         self.subsample_size = subsample_size
#         # Load the parquet file into a DataFrame
#         self.dataset = pd.read_parquet(self.parquet_file)
#         if transform is None:
#             self.transform = transforms.Compose([
#                 transforms.Resize((64, 64)),
#                 transforms.ToTensor()
#             ])
#         else:
#             self.transform = transform
#         # Optionally subsample the dataset
#         if self.subsample_size is not None:
#             self.dataset = self.dataset.sample(n=self.subsample_size).reset_index(drop=True)
#         # Print the resolution of the first image sample
#         if not self.dataset.empty:
#             first_image_bytes = self.dataset.iloc[0][kwargs.get("image_column", "image_data")]
#             first_image = Image.open(io.BytesIO(first_image_bytes)).convert("RGB")
#             print(f"First image sample resolution: {first_image.size}")
#         print("get kwargs:", kwargs.get("image_column", "image_data"))
#         # Specify the column that contains image data (assumed to be binary)
#         self.image_column = kwargs.get("image_column", "image_data")

#     def __getitem__(self, idx):
#         row = self.dataset.iloc[idx]
        
#         # Retrieve the image bytes and convert to a PIL image in RGB format
#         image_bytes = row[self.image_column]
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
#         # Apply transformation if provided
#         img_transformed = self.transform(image) if self.transform is not None else image
        
#         # Normalize using data_scaler
#         img_scaled = data_scaler(img_transformed, norm=self.norm)
        
#         # Convert to torch tensor and adjust dimensions (HxWxC -> CxHxW)
#         if not torch.is_tensor(img_scaled):
#             img_tensor = torch.tensor(img_scaled)
#         else:
#             img_tensor = img_scaled
        
#         if img_tensor.ndim == 3 and img_tensor.shape[-1] in [1, 3]:
#             img_tensor = img_tensor.permute(2, 0, 1).float()
        
#         return img_tensor

#     def __len__(self):
#         return len(self.dataset)

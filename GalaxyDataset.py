import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import transforms
from PIL import Image

class GalaxyDataset1(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        
        
        img_name = self.data_frame.iloc[idx, 0]
        img_path = f"{self.root_dir}/{str(img_name)[0:6]}/{str(img_name)}.jpg"
        image = io.imread(img_path)
        image = Image.fromarray(image)

        label = torch.tensor([self.data_frame.iloc[idx, 9], self.data_frame.iloc[idx, 10]])   
                
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

class GalaxyDataset2(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_samples=100):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Limit the dataset size to max_samples
        num_samples = min(len(self.data_frame), max_samples)

        # Load and transform images into memory
        self.images = []
        for idx in range(num_samples):
            img_name = self.data_frame.iloc[idx, 0]
            img_path = os.path.join(root_dir, str(img_name)[:6], f"{str(img_name)}.jpg")
            image = io.imread(img_path)

            # Apply the transformation before storing the image
            if self.transform:
                image = self.transform(Image.fromarray(image))

            self.images.append(image)

            # Print a message every 1000 images
            if idx % 1000 == 999:
                print(f"Loaded {idx + 1} images into self.images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].clone()  # Use a copy to avoid modifying the original image

            # Convert the label to a NumPy array with a consistent data type
        label = torch.tensor([self.data_frame.iloc[idx, 9], self.data_frame.iloc[idx, 10]])

        # If needed, you can further convert the label to torch tensor
        label = torch.tensor(label)

        return image, label
        


def main():

    transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                ])

    dataset = GalaxyDataset2(csv_file='GZ1_dataset_new.csv', root_dir="images", transform=transform)

    # Splitting the dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    data, targets = next(iter(train_loader))

    print(f"Data shape: {data.shape}")
    print(f"Targets shape: {targets}")

if __name__ == '__main__':
    main()
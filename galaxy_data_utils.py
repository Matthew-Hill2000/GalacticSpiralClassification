# from galaxy_datasets.pytorch.galaxy_dataset import GalaxyDataset
import matplotlib.pyplot as plt
# from galaxy_datasets.pytorch import GZ2
import pandas as pd
from torch.utils.data import *
from PIL import Image
import os
from torchvision import transforms
import torch

class GalaxyDataset(Dataset):
    def __init__(self, data_frame, transform=transforms.ToTensor()):
        super(GalaxyDataset, self).__init__() 
        self.data_frame = data_frame
        self.transform = transform
        self.images, self.labels = self.get_images_labels(data_frame)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label
    
    def get_images_labels(self, df):
        images= []
        labels = []
        for idx in range(len(df)):
            img_name = df.iloc[idx, 0]
            print(idx)

            path = f"gz2/images/{str(img_name)[0:6]}/{str(img_name)}.jpg"
            
            image = Image.open(path)
            if self.transform:
                image = self.transform(image)

            images.append(image)
            labels.append(torch.tensor([self.data_frame.iloc[idx, 9], self.data_frame.iloc[idx, 10]]))
        

            if len(images) % 1000 == 0 and len(images) > 0:
                print(f"Processed {len(images)} images")
            
            if len(images) == 4000:
                break

        print(f"Found {len(images)} images")
        print(images.shape)
        
        return images, labels
    

def delete_rows_without_images(new_spreadsheet_path, spreadsheet_path, image_folder_path):
    # Read the spreadsheet into a DataFrame
    df = pd.read_csv(spreadsheet_path)

    # Create a set to store image names in the folder and its subfolders

    # Iterate through each row in the DataFrame
    rows_to_delete = []
    for index, row in df.iterrows():
        # Get the image name from the first column of the row
        img_name = row.iloc[0]
        img_path = f"{image_folder_path}/{str(img_name)[0:6]}/{str(img_name)}.jpg"
        print(img_path)
        
        if not os.path.exists(img_path):
            rows_to_delete.append(index)

        # Print the current row number every 10,000 rows
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1} rows")

    # Delete the rows marked for deletion
    df = df.drop(rows_to_delete)

    # Save the modified DataFrame back to the spreadsheet
    df.to_csv(new_spreadsheet_path, index=False)
    

def main():

    # spreadsheet_path = 'GalaxyZoo1_DR_table2.csv'
    # new_spreadsheet_path = 'GZ1_dataset_new.csv'
    # image_folder_path = 'images'
    # image_extension = 'jpg'
    # delete_rows_without_images(new_spreadsheet_path, spreadsheet_path, image_folder_path)


    ## Galaxy Zoo
    transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                ])

    df = pd.read_csv('GZ1_dataset.csv')
    dataset = GalaxyDataset(data_frame=df, transform=transform)
    
    dataset_size = len(dataset)
    train_size = int(0.6 * dataset_size)  # Adjust the split ratio as needed
    test_size = int(0.2 * dataset_size)
    valid_size = dataset_size - train_size - test_size
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, valid_size])


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, num_workers=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, num_workers=8, shuffle=False) 
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=32, num_workers=8, shuffle=False)
    

    # Check the lengths of the data loaders
    print(f"Train loader size: {len(train_loader)}")
    print(f"Test loader size: {len(test_loader)}")
    print(f"Validation loader size: {len(valid_loader)}")

    # Sample a few batches from the loaders to inspect the data
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if batch_idx == 0:
            # Print the first batch from the training loader
            print("Sample batch from the training loader:")
            print(f"Batch Size: {inputs.size(0)}")
            print(f"Input Shape: {inputs.shape}")
            print(f"Label Shape: {labels.shape}")

    for batch_idx, (inputs, labels) in enumerate(test_loader):
        if batch_idx == 0:
            # Print the first batch from the test loader
            print("Sample batch from the test loader:")
            print(f"Batch Size: {inputs.size(0)}")
            print(f"Input Shape: {inputs.shape}")
            print(f"Label Shape: {labels.shape}")

    for batch_idx, (inputs, labels) in enumerate(valid_loader):
        if batch_idx == 0:
            # Print the first batch from the validation loader
            print("Sample batch from the validation loader:")
            print(f"Batch Size: {inputs.size(0)}")
            print(f"Input Shape: {inputs.shape}")
            print(f"Label Shape: {labels.shape}")

if __name__ == '__main__':
    main()
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from lenet import *
from lenet_ge import *
import tensorboard
from galaxy_data import GalaxyDataset
import pandas as pd

LEARNING_RATE = 0.0001
BATCH_SIZE = 128
NUM_EPOCHS = 120
NUM_CLASSES = 2

class Trainer():
    def __init__(self, learning_rate, batch_size, num_epochs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.test_batch_size = 256
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.scheduler = None

    def model_summary(self):
        print(summary(self.model, (3, 28, 28)))

    def load_data(self, data):
        
        if data=="CIFAR10":
            ## CIFAR10
            train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = torchvision.datasets.CIFAR10(root='data', train=False,  transform=transforms.ToTensor())                                               
            
            validation_size = 0.2
            num_samples = len(train_dataset)
            num_valid = int(num_samples*validation_size)
            num_train = num_samples - num_valid

            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_valid])
            
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)
            self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)                         

        elif data=="GalaxyZoo":
            ## Galaxy Zoo
            transform = transforms.Compose([
                        transforms.Resize((160, 160)),
                        transforms.ToTensor(),
                        ])

            df = pd.read_csv('GZ1_dataset.csv')
            dataset = GalaxyDataset(data_frame=df, transform=transform)
            
            dataset_size = len(dataset)
            train_size = int(0.2 * dataset_size)  # Adjust the split ratio as needed
            test_size = int(0.75 * dataset_size)
            valid_size = dataset_size - train_size - test_size
            train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, valid_size])

            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False) 
            self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)

            print(f"Train loader size: {len(self.train_loader)}")
            print(f"Test loader size: {len(self.test_loader)}")
            print(f"Validation loader size: {len(self.valid_loader)}")

    def load_model(self):

        self.model = LeNet(num_classes=NUM_CLASSES).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        self.model.train()
        n_total_steps = len(self.train_loader)
        start_time = time.time()
        validation_loss_min = np.Inf
        for epoch in range(self.num_epochs):
            self.model.train()
            print(f"Learning Rate: ", self.optimizer.param_groups[0]['lr'])
            
            for i, (images, labels) in enumerate(self.train_loader):
                
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass --> Backward Pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                #Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not i % 50:
                    print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                            %(epoch+1, NUM_EPOCHS, i, len(self.train_loader), loss))
                    
            validation_loss = self.validate()
            print(f'Validation Loss after epoch {epoch + 1}: {validation_loss:.4f}')          
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            
            if validation_loss < validation_loss_min:
                validation_loss_min = validation_loss
                torch.save(self.model, "model.ckpt")  

    
        print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    def validate(self):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, labels in (self.valid_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        validation_loss = total_loss / total_samples

        return validation_loss


    def test(self, data):
        self.model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            
            for i, (images, labels) in enumerate(data):
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(len(labels)):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1
            acc = 100.0 * n_correct / n_samples
            return acc


def main():
    ResNet_model = Trainer(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
    ResNet_model.load_data("GalaxyZoo")
    ResNet_model.load_model()
    ResNet_model.train()
    print('Test accuracy: %.2f%%' % (ResNet_model.test(ResNet_model.test_loader)))

if __name__ == '__main__':
    main()

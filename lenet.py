import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(LeNet, self).__init__()

        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6*input_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6*input_channels, out_channels=16*input_channels, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        

        self.fc1 = nn.Linear(16*37*37*input_channels, 120*input_channels)
        self.fc2 = nn.Linear(120*input_channels, 84*input_channels)
        self.fc3 = nn.Linear(84*input_channels, num_classes)

    def forward(self, x):
        
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        
        x = torch.flatten(x, 1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def main():
    model = LeNet()
    model.eval()
    x = torch.randn(2, 3, 160, 160)
    
    y = model(x).to('cuda')
    print(y.shape)


if __name__ == '__main__':
    main()
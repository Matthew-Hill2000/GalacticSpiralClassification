import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



class block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.expansion = 4

    

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, 
                               stride=1, padding=0,  bias=False)
        
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, 
                          stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels*self.expansion))
        

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x += self.downsample(identity)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        
        self.in_channels = 64

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        # self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        # self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        # self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * 4, num_classes)

        print("MODEL TOPOLOGY:")
        for i, (name, mod) in enumerate(self.named_modules()):
            print(f"\t{i} - {name}")

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)

        x = self.layer1(x)
        print(x.shape)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)
        # x = self.fc(x)


        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        layers = []

        layers.append(block(self.in_channels, out_channels, stride))

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = out_channels*4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=10):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=10):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=10):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


def main():
    model = ResNet50()
    model.eval()
    x = torch.randn(2, 3, 160, 160)
    
    y = model(x).to('cuda')


if __name__ == '__main__':
    main()
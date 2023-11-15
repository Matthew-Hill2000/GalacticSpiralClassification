import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn


class LeNet_GE(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, rotation_order=8):
        super(LeNet_GE, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(rotation_order)

        in_type = enn.FieldType(self.r2_act, input_channels*[self.r2_act.trivial_repr])
        self.input_type = in_type

        out_type1 = enn.FieldType(self.r2_act, input_channels*6*[self.r2_act.regular_repr])
        out_type2 = enn.FieldType(self.r2_act, input_channels*16*[self.r2_act.regular_repr])
        

        self.conv1 = enn.R2Conv(in_type, out_type1, kernel_size=5)
        self.relu1= enn.ReLU(out_type1)
        self.maxpool1 = enn.PointwiseMaxPool(out_type1, kernel_size=2, stride=2)
        self.conv2 = enn.R2Conv(out_type1, out_type2, kernel_size=5)
        self.relu2 = enn.ReLU(out_type2)
        self.maxpool2 = enn.PointwiseMaxPool(out_type2, kernel_size=2, stride=2)

        self.fc1 = nn.Linear(input_channels*rotation_order*16*5*5, 120*input_channels)
        self.fc2 = nn.Linear(120*input_channels, 84*input_channels)
        self.fc3 = nn.Linear(84*input_channels, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = enn.GeometricTensor(x, self.input_type)
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.relu1(x)
        print(x.shape)
        x = self.maxpool1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.relu2(x)
        print(x.shape)
        x = self.maxpool2(x)
        print(x.shape)
        x = x.tensor

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def main():
    model = LeNet_GE()
    model.eval()
    x = torch.randn(2, 3, 160, 160)
    
    y = model(x).to('cuda')
    print(y.shape)


if __name__ == '__main__':
    main()
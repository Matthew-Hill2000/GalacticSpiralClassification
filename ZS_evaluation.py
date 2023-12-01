import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from lenet import *
from lenet_ge import *
from skimage import io

def main():
    # Create an instance of the model
    model = LeNet(num_classes=2)

    transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                ])

    # Load the model state dictionary directly (without using torch.load)
    model_state_dict = torch.load('model.pth', map_location='cuda')
    model.load_state_dict(model_state_dict)

    # Set the model to evaluation mode
    model.eval()

    image_path = "images\\587722\\587722981741625520.jpg"
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Step 7: Forward pass
    with torch.no_grad():
        output = model(input_tensor)

             
    print(output)

if __name__ == '__main__':
    main()
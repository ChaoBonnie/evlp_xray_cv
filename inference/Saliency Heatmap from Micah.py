Model = CNN()
weights = "weights.ckpt"

import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

# Class Activation Mapping
def load_file(path):
    return np.load(path).astype(np.float32)

x = sums
y = sums_squared # use values from normalization of initial dataset (0.49, 0.248)

val_transforms = transforms.Compose([
    transforms.ToTensor()
    transforms.Normalize(x,y) 
])

val_dataset = torchvision.datasets.DatasetFolder("INSERT HERE", loader = load_file,
                                                  extensions = "npy", transform = val_transforms)

# Multiply output of last convolutional layer with the weights
temp_model = torchvision.models.resnet50()
temp_model

torch.nn.Sequential(*list(temp_model.children())[:-2]) # create a model without last two layers to capture last conv layer

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50()
        # not sure if there was one input in conv1 in your model?
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Assuming 3 features in the last fc layer due to 3 classes?
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)

        self.feature_map = torch.nn.Sequential(*List(self.model.children())[:-2])

    def forward(self, data):
        feature_map = self.feature_map(data)
        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(input = feature_map, output_size = (1,1))
        avg_output_flattened = torch.flatten(avg_pool_output)
        pred = self.model.fc(avg_output_flattened)
        return pred, feature_map
    
model = Model.load_from_checkpoint(weights, strict = False)
model.eval();

def cam(model, img):
    with torch.no_grad():
        pred, features = model(img.unsqueeze(0))
        features = features.reshape(512,49) # not sure what desired shape is
        weight_params = list(model.model.fc.parameters())[0]
        weight = weight_params[0].detach()

        cam = torch.matmul(weight,features)
        cam_img = cam.reshape(7,7).cpu()
        return cam_img, torch.sigmoid(pred)
    
def visualize(img, cam, pred):
    img = img [0]
    cam = transforms.functional.resize(cam.unsqueeze(0), (224,224))[0]

    fig, axis = plt.subplots(1,2)
    axis[0].imshow(img, cmap = "bone")
    axis[1].imshow(img, cmap = "bone")
    axis[1].imshow(cam, alpha = 0.5, cmap = "jet")
    plt.title(pred>0.5)

    # Visualize!
    img = val_dataset[][0]
    activation_map, pred = cam(model, img)
    visualize(img, activation_map, pred)



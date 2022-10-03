import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch   
import os
import torchvision.transforms as tt #To apply transformations to the dataset, augmenting it and transforming it to a tensor.
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder 
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm.notebook import tqdm
from resnetmodel import *

from PIL import Image
import torchvision.transforms as transforms

data_dir = r'C:\Users\hp\Downloads\dermnet'
test_img = (data_dir + '/test/Atopic Dermatitis Photos/1IMG011.jpg')
image = Image.open(test_img)

transform = transforms.ToTensor()
test_ds = transform(image)

model = to_device(Resnet34(), device)
predict_image(test_ds,model,test_ds)
test_tf= tt.Compose([   
    tt.Resize(size=(150,150)),
    tt.ToTensor(),
])

# test_dir = (data_dir + '/test')
# test_ds = ImageFolder(test_dir,test_tf)

# test_dl = DataLoader(
#     test_ds,
#     batch_size=batch_size,
#     num_workers=3,
#     shuffle=False,
#     pin_memory=True
# )
# test_dl = DeviceDataLoader(test_dl, device)  
# history = [evaluate(model, test_dl)]
# print(history)
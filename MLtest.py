import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt
import pandas as pd

class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
                
        image = item[0:].values.astype(np.uint8).reshape((28, 28))
        label = '?'
        if self.transform is not None:
            image = self.transform(image)
            label = '?'
        return image, label

BATCH_SIZE = 100
VALID_SIZE = 0.00 # процент данных для проверки

transform_train = transforms.Compose([
    transforms.ToPILImage(),
   
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

transform_valid = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 10),
        )
                
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = Net()

model.load_state_dict(torch.load('../ML/model_mtl_mnist.pt'))

LEARNING_RATE = 0.2

# функция ошибки и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)

def pencilsketch(inp_img):
    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)
    final_img = dodgeV2(img_gray, img_smoothing)
    return(final_img)

def image2df(inpimg):
    img_array = cv2.cvtColor(inpimg, cv2.COLOR_BGR2GRAY)
    img_pil = Image.fromarray(img_array)
    img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
    img_array = (img_28x28.flatten())

    #print(img_array.shape)
    img_array  = (img_array.reshape(-1,1).T-255) *-1
    df = pd.DataFrame(img_array)
    return df


def data2ML(df):
    dataset = df
    train_data = DatasetMNIST(dataset, transform=transform_train)
    valid_data = DatasetMNIST(dataset, transform=transform_valid)
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(VALID_SIZE * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = train_idx
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    #valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, sampler=valid_sampler)
    model.eval()
    for data, target in train_loader:
    
        output = model(data)
    _, pred = torch.max(output, 1)  
    return int(pred)


st.title("Digit recognizer")
st.write("This Web App is to help predict your handwritten digits")

file_image = st.sidebar.file_uploader("Upload your Photos", type=['jpeg','jpg','png'])

if file_image is None:
    st.write("You haven't uploaded any image file")

else:
    input_img = Image.open(file_image)
    dataf = image2df(np.array(input_img))
    predx = data2ML(dataf)
    final_sketch = pencilsketch(np.array(input_img))
    st.write("**Input Photo**")
    st.image(input_img, use_column_width=True)
    st.write(f"Model Prediction {predx}")
    #st.image(final_sketch, use_column_width=True)
    
    # if st.button("Download Images"):
    #     im_pil = Image.fromarray(input_img)
    #     im_pil.save(f"{predx}.jpeg")
    #     st.write('Download completed')
   

---
title: VGGNet implementation issues & learning
layout: default
parent: Image Classification
grand_parent: Paper Review
nav_order: 4
---
date: 2022-10-05

I faced several issues and learned several things when implementing VGGNet by using PyTorch. 

### 1. model.eval()

Dropout layer and batch normalization is automatically turned off when this function is called. So, there is no need to separate model architecture with dropout layer version and without dropout layer version.

In contrast, when `model.train()` is executed, dropout layer and batch normalization is turned on. 

### 2. torch-summary module issue 

When using torch-summary, the batch size is automatically set to 2. 

So, when flattening the tensor, we need to take care of batch size. 

### 3. Tensor.view(-1) vs torch.flatten()

Unlike Tensor.view(-1), torch.flatten() can designate which tensor dimension to flatten. So it can ignore the batch size and can manipulate image-related dimensions from 2nd to 4th dimension.

When we use Tensor.view(-1) to flatten the feature map, this code can be used.

`out = out.view(-1,512*7*7) # (batchsize, 512*7*7)` 

-1 is used to set batch size, and 512\*7\*7 is used to represent the size when feature map is flattened. 

When we use torch.flatten() to flatten the feature map, this code can be used. 

`out = torch.flatten(out, start_dim=1)`

`start_dim=1` is used to designate 1th to 3rd dimension of the tensor to be flattened without 0th dimension.

### 4. module initialize related issue 

I first implemented VGGNet by making ReLU layer, MaxPooling layer, and classification layers (fully-connected layers) as properties of the class and used this properties to define the module in the model class. However, this led to unappropriate redundancy of the layer properties. For instance, the first ReLU layer that is supposed to be only one layer after the first convolutional layer was defined as multiple layers. 



---



My VGGNet implementation has same amount of parameters, which is about 138 million. 

This is described in [this link](https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96).

This is the final VGG layer (using configuration D of original VGG paper) that I implemented using PyTorch. 

```python
import torch 
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, dropout_ratio=0.5):
        super(VGG16, self).__init__()
        # input is 224x224 RGB image 3@224x224
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_ratio = dropout_ratio
        
        # convolutional layers (with ReLU and MaxPooling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # 64@224x224
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64@224x224
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)) # 64@112x112
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 128@112x112
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 128@112x112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)) #128@56x56
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride=1, padding=1), # 256@56x56
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 256@56x56
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 256@56x56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)) # 256@28x28
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512@28x28
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512@28x28
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512@28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)) #512@14x14
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512@14x14
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512@14x14
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512@14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)) # 512@7x7
        
        
        # fully connected module
        self.fc_module = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.Dropout(p=self.dropout_ratio),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.Dropout(p=self.dropout_ratio),
            nn.ReLU(),
            nn.Linear(4096,1000)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        
        out = out.view(-1,512*7*7) # (batchsize, -1)
        # out = torch.flatten(out, start_dim=1)
        out = self.fc_module(out)        
        out = F.softmax(out, dim=1)
        return out
```



And this is the output I got from using torch-summary module using the VGG model I implemented.

```tex
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]          36,928
              ReLU-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 128, 112, 112]          73,856
              ReLU-7        [-1, 128, 112, 112]               0
            Conv2d-8        [-1, 128, 112, 112]         147,584
              ReLU-9        [-1, 128, 112, 112]               0
        MaxPool2d-10          [-1, 128, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]         295,168
             ReLU-12          [-1, 256, 56, 56]               0
           Conv2d-13          [-1, 256, 56, 56]         590,080
             ReLU-14          [-1, 256, 56, 56]               0
           Conv2d-15          [-1, 256, 56, 56]         590,080
             ReLU-16          [-1, 256, 56, 56]               0
        MaxPool2d-17          [-1, 256, 28, 28]               0
           Conv2d-18          [-1, 512, 28, 28]       1,180,160
             ReLU-19          [-1, 512, 28, 28]               0
           Conv2d-20          [-1, 512, 28, 28]       2,359,808
             ReLU-21          [-1, 512, 28, 28]               0
           Conv2d-22          [-1, 512, 28, 28]       2,359,808
             ReLU-23          [-1, 512, 28, 28]               0
        MaxPool2d-24          [-1, 512, 14, 14]               0
           Conv2d-25          [-1, 512, 14, 14]       2,359,808
             ReLU-26          [-1, 512, 14, 14]               0
           Conv2d-27          [-1, 512, 14, 14]       2,359,808
             ReLU-28          [-1, 512, 14, 14]               0
           Conv2d-29          [-1, 512, 14, 14]       2,359,808
             ReLU-30          [-1, 512, 14, 14]               0
        MaxPool2d-31            [-1, 512, 7, 7]               0
           Linear-32                 [-1, 4096]     102,764,544
          Dropout-33                 [-1, 4096]               0
             ReLU-34                 [-1, 4096]               0
           Linear-35                 [-1, 4096]      16,781,312
          Dropout-36                 [-1, 4096]               0
             ReLU-37                 [-1, 4096]               0
           Linear-38                 [-1, 1000]       4,097,000
================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.59
Params size (MB): 527.79
Estimated Total Size (MB): 746.96
----------------------------------------------------------------

```
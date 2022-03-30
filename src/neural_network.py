from audioop import mul
from torch import nn
from torch.nn.functional import pad
import torch


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.conv_model = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,padding=1,stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1,stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,padding=1,stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=3,padding=1,stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,10,kernel_size=3,padding=1,stride=2, bias=True),
        )
        self.final_activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
    def pad_to_multiple(self,x,multiple=32):
        h,w = x.shape[-2:]
        h_pad,w_pad = 0,0
        if h%multiple != 0:
            h_pad = multiple - (h%multiple)
        if w%multiple != 0:
            w_pad = multiple - (w%multiple)
        l = w_pad//2
        r = w_pad - l
        t =  h_pad//2
        b = h_pad - t
        return pad(x,[l,r,t,b])

    def forward(self, x):
        x = self.pad_to_multiple(x,32)
        logits = self.conv_model(x)
        return logits.squeeze()

    def activate(self,x):
        return self.final_activation(x)
    
    def calc_loss(self,preds,target):
        return self.loss(preds,target)
    
    def parse_predictions(self,x):
        activated = self.activate(x)
        return torch.argmax(activated,dim=1)

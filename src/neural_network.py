from torch import nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.final_activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def activate(self,x):
        return self.final_activation(x)
    
    def calc_loss(self,preds,target):
        return self.loss(preds,target)
    
    def parse_predictions(self,x):
        activated = self.activate(x)
        return torch.argmax(activated,dim=1)

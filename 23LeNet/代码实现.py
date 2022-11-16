import torch
from torch import nn
from d2l import torch as d2l

class Reshape(torch.nn.Module):   # 可以用在Sequential里
    def forward(self,x):
        return x.view(-1,1,28,28)  #批量数不变通道数变成1

net = torch.nn.Sequential(
    Reshape(),nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),nn.Flatten(),  #flatten()只保留第一维度。
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10)
)
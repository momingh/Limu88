import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# Pytorch不会隐式地调整输入的形状
# 因此，我们定义了展平层（flatten）在线性层前调整网络输入的形状
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))  # nn.Flatten()将任何维度的向量变为二维向量，其中第一个维度保留
# nn.Linear(inputs,outputs)输入和输出维度

def init_weights(m): # m代表当前layer
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

loss = nn.CrossEntropyLoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()
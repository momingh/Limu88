import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w,true_b,1000)

def load_array(data_arrays,batch_size,is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size=10
data_iter=load_array((features,labels),batch_size)

next(iter(data_iter))

# 模型
# 'nn'是神经网络的缩写
from torch import nn

net=nn.Sequential(nn.Linear(2,1))  # nn.Linear(输入维度,输出维度)，Sequential是一个层的容器。

# 初始化模型参数（torch会自动初始化模型，也可以人为设置）
net[0].weight.data.normal_(0,0.01) # 使用正态分布替换掉data中的值
net[0].bias.data.fill_(0)

# 计算均方误差使用的是MSELoss类，也称为平方L2范数
loss=nn.MSELoss()

# 实力化SGD实例
trainer=torch.optim.SGD(net.parameters(),lr=0.03)


# 训练过程
num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        # loss(net(X), y).backward() # 同样也可以梯度下降
        l.backward()
        trainer.step()   # 更新参数
    l=loss(net(features),labels)
    print(f'epoch {epoch+1}, loss {l:f}')
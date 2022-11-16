import torch
from torch import nn

def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape) # !!!!!!!!!!!!! 加入通道数和批量大小
    Y=conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1)
x=torch.randn(size=(8,8))
print(comp_conv2d(conv2d,x).shape)

# 填充不同的高度和宽度吗
conv2d=nn.Conv2d(1,1,kernel_size=(5,3),padding=(2,1))
print(comp_conv2d(conv2d,x).shape)

# 将高度和宽度的步幅设置为2
conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1,stride=2)
print(comp_conv2d(conv2d,x).shape)

# 稍微复杂点的例子
conv2d=nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=(3,4))
print(comp_conv2d(conv2d,x).shape)

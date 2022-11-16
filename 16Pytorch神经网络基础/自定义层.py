import torch
import torch.nn.functional as F
from torch import nn

# 自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self,X):
        return X-X.mean()

layer=CenteredLayer() # 实例化
print(layer(torch.FloatTensor([1,2,3,4,5])))

# 将层作为组件合并到构建更复杂的模型中
net=nn.Sequential(nn.Linear(8,128),CenteredLayer())

Y=net(torch.rand(4,8))
print(Y.mean())

# 带参数的图层
class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super(MyLinear, self).__init__()
        self.weight=nn.Parameter(torch.randn(in_units,units))
        self.bias=nn.Parameter(torch.randn(units,))

    def forward(self,X):
        linear=torch.matmul(X,self.weight.data)+self.bias
        return F.relu(linear)

dense=MyLinear(5,3)
print(dense.weight)

# 使用自定义层执行正向传播计算
print(dense(torch.rand(2,5)))
# 使用自定义层构建模型
net=nn.Sequential(MyLinear(64,8),MyLinear(8,1))
net(torch.rand(2,64))
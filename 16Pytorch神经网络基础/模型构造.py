import torch
from torch import nn
from torch.nn import functional as F
X=[]
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(20,256)
        self.out=nn.Linear(256,10)

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))

net=MLP() # 实例化MLP类
net(X) # 调用__call()__方法，__call()__调用了forward函数



class MySequential(nn.Module):   # 这样的好处是：可以执行许多自定义的计算
    def __init__(self,*args):
        super(MySequential, self).__init__()
        for block in args:
            self._modules[block]=block  # _modules是一个特殊的容器。

    def forward(self,X):
        for block in self._modules.values():
            X=block(X)
        return X

net=MySequential(nn.Linear(20,256),nn.ReLU())



class FixedHiddenMLP(nn.Module):   # 灵活的定义方式（代码没有实际意义）
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        self.rand_weight=torch.rand((20,20),requires_grad=False)
        self.linear=nn.Linear(20,20)

    def forward(self,X):
        X=self.linear(X)
        X=F.relu(torch.mm(X,self.rand_weight)+1)
        X=self.linear(X)
        while X.abs().sum()>1:
            X/=2
        return X.sum()


# 混合搭配各种组合快的方法
class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net=nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                               nn.Linear(64,32),nn.ReLU())
        self.linear=nn.Linear(32,16)

    def forward(self,X):
        return self.linear(self.net(X))

chimera=nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
chimera(X)


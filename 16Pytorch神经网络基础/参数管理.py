import torch
from torch import nn

net =nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X=torch.rand(size=(2,4))
net(X)

# 参数访问
print(net[2].state_dict())

# 直接访问目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad==None) # 还没做反向计算，所以没梯度

# 一次性访问所有参数
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
print(*[(name,param.shape) for name,param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)








# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())

def block2():
    net=nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}',block1())
    return net

rgnet=nn.Sequential(block2(),nn.Linear(4,1))
print(rgnet(X))
print(rgnet)


# 内置初始化
def init_normal(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01) # _表示一个替换操作，直接换掉weight的值，不需要返回值（原地操作）
        nn.init.zeros_(m.bias)

net.apply(init_normal)  #对与net中的所有layer，去调用init_normal。即遍历一遍
print(net[0].weight.data[0])
print(net[0].bias.data[0])

def init_constant(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print(net[0].weight.data[0])
print(net[0].bias.data[0])

# 对不同的块运用不同的初始化方法
def xavier(m):
    if type(m)==nn.Linear:
        nn.init.xavier_normal_(m.weight)

def init_42(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[0].bias.data[0])


# 自定义初始化
def my_init(m):
    if type(m)==nn.Linear:
        print("Init",*[(name,param.shape) for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10) # 均匀初始化
        m.weight.data*=m.weight.data.abs()>=5 # 权重小于5的赋0

net.apply(my_init)
print(net[0].weight[:2])


# 参数绑定(共享某些层的权重)
shared=nn.Linear(9,8)
net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1)) # 第二层、第三层权共享
net(X)
print(net[2].weight.data[0]==net[4].weight.data[0])
net[2].weight.data[0,0]=100
print(net[2].weight.data[0]==net[4].weight.data[0])
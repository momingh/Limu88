import torch
from torch import nn
from torch.nn import functional as F
x=torch.arange(4)
torch.save(x,'x-file')  # 把x存入x-file

x2=torch.load("x-file") #读x-file文件
print(x2)


# 存储一个张量列表，然后把它们读回内存
y = torch.zeros(4)
torch.save([x,y],'x-files')
x2, y2 = torch.load('x-files') #读x-file文件
print((x2, y2))

# 写入或读取从字符串映射到张量的字典
mydict = {'x':x, 'y':y}
torch.save(mydict,'mydict')
mydict2 =torch.load('mydict')
print(mydict2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden=nn.Linear(20,256)
        self.output=nn.Linear(20,256)

    def forward(self,X):
        return self.output(F.relu(self.hidden(X)))

net=MLP()
X=torch.randn(size=(2,20))
Y=net(X)
   # 将参数保存在一个叫做"mlp.params"的文件中
torch.save(net.state_dict(),'mlp.params')
'''
 state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
(注意,只有那些参数可以训练的layer才会被保存到模型的state_dict中,如卷积层,线性层等等)
优化器对象Optimizer也有一个state_dict,它包含了优化器的状态以及被使用的超参数(如lr, momentum,weight_decay等)
state_dict是在定义了model或optimizer之后pytorch自动生成的,可以直接调用.常用的保存state_dict的格式是".pt"或'.pth'的文件
'''
  # 实例化了原始多层感知机模型的一个备份。直接读取文件中存储的参数
clone=MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

Y_clone=clone(X)
print(Y_clone==Y)
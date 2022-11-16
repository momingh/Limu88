import random
import torch
import matplotlib.pyplot as plt

from d2l import torch as d2l

# 生成一个人工数据集
def synthetic_data(w,b,num_examples):
    """生成y=Xw+b+噪声。"""
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)  #给y加噪音
    return X,y.reshape((-1,1))   # -1表示自动计算有多少行，1表示固定一列（既返回一个列向量）

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels=synthetic_data(true_w, true_b, 1000)

print('features:',features[0],'\nlabel:',labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().
                numpy(),1)
plt.show()


# 定义一个data_iter函数（可迭代函数），该函数接受批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的批大小
def data_iter(batch_size,features,labels):
    num_examples=len(features)  #len取featrues第一维度的大小
    indices=list(range(num_examples))
    #这些样本是随机读取的，没有特定顺序
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        #d=indices[i:min(i+batch_size,num_examples)]    # 不一定要转换成张量，也可以
        yield features[batch_indices], labels[batch_indices]
        #yield features[d],labels[d]   # 用yield代替return，返回时记住代码的执行位置，再次调用这个函数，从上次的位置开始执行。

batch_size=10

for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break

w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X,w)+b

def squared_loss(y_hat,y):
    """均方损失"""
    return (y_hat - y.reshape(y.shape))**2 / 2

# 定义优化算法
def sgd(params,lr,batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():      #参数更新的时候不需要进行梯度计算。
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 训练过程
lr=0.07
num_epochs=3
net=linreg
loss=squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y)
        #因为'1'形状是（'batch_size'，1），而不是一个标量。'l'中所有元素求和
        #并以此计算关于['w','b']的梯度
        l.sum().backward()
        sgd([w,b],lr,batch_size)  #使用参数的梯度更新参数
    with torch.no_grad():    #查看当前损失
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')


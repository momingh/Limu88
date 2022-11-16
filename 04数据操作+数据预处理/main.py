import torch
x=torch.arange(4.0,requires_grad=True)
#x.grad
y=2*torch.dot(x,x)
y.backward()
print('1',x.grad==4*x)


# 在默认情况下，Pytorch会累计梯度，我们需要清除之前的值
x.grad.zero_()
print('2',x.grad)
y=x.sum()
y.backward()
print('3',x.grad)


# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y=x*x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

# 将某些计算移动到记录的计算图之外
x.grad.zero_()
y=x*x
u=y.detach()
z=u*x

z.sum().backward()
print(x.grad==u)


# 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。
def f(a):
    b=a*2
    while b.norm()<1000:#norm是L2范数
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c

a=torch.randn(size=(),requires_grad=True)
d=f(a)
d.backward()

print(a.grad==d/a)
import numpy as np
#权重向量与输入向量赋值
w1=np.array([0.5,0.25,-0.75])
x=np.array([1,0.5,-1])
#定义relu函数
def relu(z):
    return np.maximum(0, z)
z=w1*x
y=relu (z) #激活函数处理
print(y)     
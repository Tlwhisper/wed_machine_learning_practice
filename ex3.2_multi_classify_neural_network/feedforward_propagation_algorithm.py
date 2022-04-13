# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio #读取mat文件
import scipy.optimize as opt

data = scio.loadmat('ex3data1.mat')
weights = scio.loadmat('ex3weights.mat')
x = data['X']
y = data['y']
theta1 = weights['Theta1']
theta2 = weights['Theta2']

'''向前传播'''
#sigmoid函数作为激活函数
def g(x):
    return 1/(1+np.exp(-x))

#前面加一列1的函数
def plus1(x):
    return np.column_stack((np.ones(x.shape[0]), x))


#前馈函数
def forward_pro(x, theta1, theta2): #如果多层可添加theta
    b1 = x  #(5000, 400)
    for i in range(1,3): #如果为n层网络，则这里的3改为n即可
        locals()['a'+str(i)] = plus1(locals()['b'+str(i)])
        locals()['z'+str(i+1)] = locals()['a'+str(i)]@locals()['theta'+str(i)].T
        locals()['b'+str(i+1)] = g(locals()['z'+str(i+1)])
        if i+1 == 3:  #如果为n层网络，则这里的3改为n即可
            b3 = g(locals()['z'+str(i+1)])
    return b3  #(5000, 10)  如果为n层网络，这样输出也是an


'''预测与评价'''
#预测的y值
def predict(prob):
    y_predict = np.zeros((prob.shape[0],1))
    for i in range(prob.shape[0]):
        #查找第i行的最大值并返回它所在的位置,再加1就是对应的类别
        y_predict[i] = np.unravel_index(np.argmax(prob[i,:]), prob[i,:].shape)[0]+1
    return y_predict
        
#精度
def accuracy(y_predict, y=y):
    m = y.size
    count = 0
    for i in range(y.shape[0]):
        if y_predict[i] == y[i]:
            j = 1 
        else:
            j = 0
        count = j+count #计数预测值和期望值相等的项
    return count/m
    
#预测的概率矩阵 (5000, 10)        
prob = forward_pro(x, theta1, theta2)
y_predict = predict(prob)
accuracy(y_predict)
print ('accuracy = {0}%'.format(accuracy(y_predict) * 100))


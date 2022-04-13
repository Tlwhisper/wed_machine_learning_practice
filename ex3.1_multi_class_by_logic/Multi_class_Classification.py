
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio #读取mat文件
import scipy.optimize as opt

data = scio.loadmat('ex3data1.mat')
x = data['X']
y = data['y']


'''数据可视化'''
s = np.random.permutation(x) #随机重排，但不打乱x中的顺序
a = s[:100,:] #选前100行,(100, 400)

#定义可视化函数
def displayData(x):
    plt.figure()
    n = np.round(np.sqrt(x.shape[0])).astype(int)
    #定义10*10的子画布
    fig, a = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(6, 6))
    #在每个子画布中画出一个数字
    for row in range(n):
        for column in range(n):
            a[row, column].imshow(x[10 * row + column].reshape(20,20).T, cmap='gray')
    plt.xticks([]) #去掉坐标轴
    plt.yticks([])       
    plt.show()

displayData(a)


'''计算代价函数和梯度'''
#sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

#原本代价函数
def cost_func(theta, x, y):
    m = y.size
    return -1/m*(y@np.log(sigmoid(x@theta))+(1-y)@np.log(1-sigmoid(x@theta)))

#正则化的代价函数，不惩罚第一项theta[0]
def cost_reg(theta, x, y, l=0.1):
    m = y.size
    theta_ = theta[1:] #选取第二项以后的
    return cost_func(theta, x, y) + l/(2*m)*np.sum(theta_*theta_)

#原本的梯度
def gradient_func(theta, x, y):
    m = y.size
    return 1/m*((sigmoid(x@theta))-y).T@x

#正则化的的梯度，不惩罚第一项theta[0]
def gradient_reg(theta, x, y, l=0.1):
    theta_ = l/(y.size)*theta
    theta_[0] = 0 #第一项不惩罚设为0
    return gradient_func(theta, x, y) + theta_


'''一对多分类'''
def one_vs_all(x, y, l, K=10):
    all_theta = np.zeros((x.shape[1], K)) #应该是(10, 401)
    for i in range(K):
        iteration_y = np.array([1 if j==i+1 else 0 for j in y]) #第0列到第9列分别对应类别1到10
        p = opt.fmin_ncg(f=cost_reg, fprime=gradient_reg, x0=all_theta[:, i:i+1], args=(x, iteration_y), maxiter=400)
        all_theta[:, i:i+1] = np.mat(p).T
    return all_theta
    
#为x添加了一列常数项 1 ，以计算截距项（常数项）
x = np.column_stack((np.ones(x.shape[0]), x))
lmd = 0.1
all_theta = one_vs_all(x, y, l=lmd)


'''预测与评价'''
#预测的概率矩阵 (5000, 10)
def probability(x, theta):
    return sigmoid(x@theta)

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
    
        
prob = probability(x, all_theta)
y_predict = predict(prob)
accuracy(y_predict)
print ('accuracy = {0}%'.format(accuracy(y_predict) * 100))



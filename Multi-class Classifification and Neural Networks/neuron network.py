import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report
from scipy.optimize import minimize

def sample_show(X):
    sample_idx = np.random.choice(np.arange(X.shape[0]),100)
    sample_image = X[sample_idx,:]
    fig,ax_array = plt.subplots(nrows = 10,ncols = 10,sharey = True,sharex = True,figsize = (12,12))
    for r in range(10):
        for c in range(10):
            ax_array[r,c].matshow(np.array(sample_image[10 * r + c].reshape((20,20))).T,cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def costfun(theta,X,y,learningRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    first = np.multiply(-y , np.log(sigmoid(X*theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
    return np.sum(first-second)/len(X) + (learningRate/(2*len(X)))*np.sum(np.power(theta[:,1:],2))

def gradient(theta,X,y,learningRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    error = sigmoid(X * theta.T) - y
    grad = ((X.T*error).T) / len(X) + ((learningRate/len(X))*theta)
    grad[0,0] = np.sum(np.multiply(error,X[:,0]))/len(X)
    return np.array(grad).ravel()

def onevsall(X, y, num_label, learningRate):
    para = X.shape[1]
    all_theta = np.zeros((num_label,para))
    for i in range(1 , num_label+1):
        theta = np.zeros(para)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i,(X.shape[0],1))
        fmin = minimize(fun=costfun, x0=theta, args=(X, y_i, learningRate), method='TNC', jac=gradient)
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        all_theta[i-1,:] = fmin.x
    return all_theta

def predict_all(X,all_theta):
    X = np.mat(X)
    all_theta = np.mat(all_theta)
    h = sigmoid(X*all_theta.T)
    return np.argmax(h, axis = 1) + 1

def predict(theta1, theta2, X):
    X = np.mat(X)
    theta1 = np.mat(theta1)
    theta2 = np.mat(theta2)
    z1 = sigmoid(X*theta1.T)
    z1 = np.insert(z1,0,1,axis = 1)
    h = sigmoid(z1*theta2.T)
    return np.argmax(h, axis = 1) + 1
    
data = loadmat('Y:\exercise\machine-learning-ex3\ex3data1.mat')
theta = loadmat('Y:\exercise\machine-learning-ex3\ex3weights.mat')
theta1 = theta['Theta1']
theta2 = theta['Theta2']

X = np.insert(data['X'],0,1,axis = 1)
y = data['y']


alltheta = onevsall(X, y, 10, 1)
y_pred = predict_all(X, alltheta)
print(classification_report(data['y'], y_pred))
y_pred2 = predict(theta1, theta2, X)
print(classification_report(y, y_pred2))
"""
classification_report简介
sklearn.metrics.classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
sklearn中的classification_report函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。
主要参数:
y_true：1维数组，或标签指示器数组/稀疏矩阵，目标值。
y_pred：1维数组，或标签指示器数组/稀疏矩阵，分类器返回的估计值。
labels：array，shape = [n_labels]，报表中包含的标签索引的可选列表。
target_names：字符串列表，与标签匹配的可选显示名称（相同顺序）。
sample_weight：类似于shape = [n_samples]的数组，可选项，样本权重。
digits：int，输出浮点值的位数
"""


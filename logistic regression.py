import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def computeCost(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(computeSigmod(X * theta.T)))
    second = np.multiply((1-y),np.log(1-computeSigmod(X * theta.T)))
    return np.sum(first - second)/len(X)

def computeSigmod(z):
    return 1 / (1 + np.exp(-z))

def predict(theta, X):
    probability = computeSigmod(X*theta.T)
    return [1 if x>=0.5 else 0 for x in probability]
    
    
def hfunc1(theta,X):
    theta = np.matrix(theta)
    X = np.matrix(X)
    return computeSigmod(theta*X.T)

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = computeSigmod(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad


    
path1 = 'Y:\exercise\logistic regression\ex2data1.txt'
data1 = pd.read_csv(path1,names=['Exam 1','Exam 2','admitted'])
data1.insert(0, 'Ones', 1)
cols = data1.shape[1]
X = data1.iloc[:,0:cols-1]
y = data1.iloc[:,cols-1:cols]
theta = np.zeros(3)

# 转换X，y的类型
X = np.array(X.values)
y = np.array(y.values)

result = opt.fmin_tnc(func=computeCost, x0=theta, fprime=gradient, args=(X, y))

predictions = predict(np.mat(result[0]),X)
correct = [1 if(a == 0 and b == 0)or(a == 1 and b == 1)else 0 for (a,b) in zip(predictions,y)]
accuracy = np.sum(correct)/len(correct)
print("accuracy = %d%%"%(accuracy*100))

pos = data1[data1['admitted'].isin([1])]
neg = data1[data1['admitted'].isin([0])]
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = ( - result[0][0] - result[0][1] * plotting_x1) / result[0][2]
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Prediction')
ax.scatter(pos['Exam 1'],pos['Exam 2'],s = 50,c ='b',marker = 'o',label = 'Admitted')
ax.scatter(neg['Exam 1'],neg['Exam 2'],s = 50,c ='r',marker = 'x',label = 'Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()


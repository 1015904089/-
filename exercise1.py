import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return sum(inner)/(len(X)*2)

def boundary(X,y):
    x1 = np.linspace(-1,1.5,150)
    x2 = np.linspace(-1,1.5,150)
    z = np.zeros((len(x1),len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            t = np.mat([x1,x2])
            z[i][j] = computeCost(X,y,t)
    return x1,x2,z

def gridentdecent(X,y,theta,alpha,iter):
    para = X.shape[1]
    temp = theta
    cost = np.zeros(iter)
    for i in range(iter):
        error = (X * theta.T) - y
        for j in range(para):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - alpha/len(X)*np.sum(term)
        theta = temp
        cost[i] = computeCost(X,y,theta)
    return theta,cost

def nomalization(X,y):
    return ((np.linalg.inv(X.T*X))*X.T*y)

path1 = 'Y://exercise/Linear Regression/ex1data1.txt'
data1 = pd.read_csv(path1,names = ['Pop','profit'])
data1.insert(0,'Ones',1)
col1 = data1.shape[1]
X1 = data1.iloc[:,0:col1-1]
y1 = data1.iloc[:,col1-1:col1]
X1 = np.mat(X1.values)
y1 = np.mat(y1.values)
theta1 = np.mat(np.zeros(X1.shape[1]))
X = np.linspace(data1.Pop.min(), data1.Pop.max(), 100)
f = g[0, 0] + (g[0, 1] * x)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(X, f, 'r', label='Prediction')
ax.scatter(data1.Pop, data1.profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()




path2 = 'Y://exercise\Linear Regression\ex1data2.txt'
data2 = pd.read_csv(path2,names = ['Size', 'Bedrooms', 'Price'])
data2 = (data2-data2.mean())/data2.std()
data2.insert(0,'Ones',1)
col2 = data2.shape[1]
X2 = data2.iloc[:,0:col2-1]
y2 = data2.iloc[:,col2-1:col2]
X2 = np.mat(X2.values)
y2 = np.mat(y2.values)
theta2 = np.mat(np.zeros(X2.shape[1]))

alpha = 0.01
iter = 30000

g2,cost2 = gridentdecent(X2,y2,theta2,alpha,iter)
g1,cost1 = gridentdecent(X1,y1,theta1,alpha,iter)
print(g1)

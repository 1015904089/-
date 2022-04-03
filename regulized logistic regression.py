import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def func(x1,x2,theta):
    res = theta[0]
    place = 1
    for i in range(1, 7):
        for j in range(0, i+1):
            res += (x1**(i-j))*(x2**j) * theta[place]
            place += 1
    return res

def boundary(theta):
    x1 = np.linspace(-1,1.5,200)
    x2 = np.linspace(-1,1.5,200)
    z = np.zeros((len(x1),len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            z[i][j] = func(x1[i],x2[j],theta);
    return x1,x2,z
    
    
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost(theta,X,y,learningrate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)
    n = theta.shape[1]
    first = np.multiply(y,np.log(sigmoid(X * theta.T)))
    second = np.multiply(1-y,np.log(1-sigmoid(X * theta.T)))
    reg = learningrate/(m*2)*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(-first-second)/m + reg

def gredient(theta,X,y,learningrate):
    grad = theta
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    error = sigmoid(X*theta.T) - y
    for i in range(theta.shape[1]):
        term = np.multiply(error,X[:,i])
        if(i == 0):
            grad[i] = np.sum(term)/len(X)
        else:
            grad[i] = np.sum(term)/len(X)+((learningrate / len(X)) * theta[:,i])
    return grad
    
def predict(theta, X):
    probability = sigmoid(X*theta.T)
    return [1 if x >=0.5 else 0 for x in probability]


    
path1 = 'Y:\exercise\logistic regression\ex2data2.txt'
data = pd.read_csv(path1, names=['Test 1','Test 2','admitted'])
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.zeros(cols -1)

pos  = data[data['admitted'].isin([1])]
neg  = data[data['admitted'].isin([0])]
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(pos['Test 1'],pos['Test 2'],s = 50, c = 'b', marker = '*', label = 'Accepted')
ax.scatter(neg['Test 1'],neg['Test 2'],s = 50, c = 'r', marker = 'o', label = 'Accepted')
ax.legend()
ax.set_xlabel('Test 1')
ax.set_ylabel('Test 2')



data2 = data
data2.insert(3, 'Ones', 1)
x1 = data['Test 1']
x2 = data['Test 2']
degree = 6
for i in range(1,degree +1):
    for j in range(0,i+1):
        data2['F'+str(i-j)+str(j)] = np.multiply(np.power(x1,i-j),np.power(x2,j))
data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)
col2 = data2.shape[1]
y = np.array(data2.iloc[:,0:1])
X = np.array(data2.iloc[:,1:col2])
theta = np.zeros(col2-1)
learningrate = 0
result2 = opt.fmin_tnc(func=cost, x0=theta, fprime=gredient, args=(X, y, learningrate))
predictions = predict(np.mat(result2[0]),X)
theta2 = result2[0]
correct = [1 if(a == 0 and b == 0)or(a == 1 and b == 1)else 0 for (a,b) in zip(predictions,y)]
accuracy = np.sum(correct)/len(correct)
x1,x2,z = boundary(theta2)
ax.contour(x1,x2,z,[-0.01,0.01])
plt.show()


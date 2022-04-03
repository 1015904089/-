import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from PIL import Image
from scipy.optimize import minimize
import PIL.ImageOps


def unroll(m1, m2):
    return np.concatenate((np.ravel(m1), np.ravel(m2)))


def get_train_pattern():
    curdir = "Y://exercise/Back propagation/mnist训练"
    train = loadmat(curdir + "/mnist_train.mat")["mnist_train"]
    train_label = loadmat(curdir + "/mnist_train_labels.mat")["mnist_train_labels"]
    train = np.where(train > 180, 1, 0)  # 二值化
    return train, train_label


def get_test_pattern():
    curdir = "Y://exercise/Back propagation/mnist训练/mnist_test"
    test_img = []
    test_label = []
    for i in range(10):
        img_url = os.listdir(curdir + '/' + str(i))
        for url in img_url:
            img = Image.open(curdir + '/' + str(i) + "/" + url).convert('1')
            img_array = np.asarray(img, 'i')
            img_vector = img_array.reshape(img_array.shape[0] * img_array.shape[1])
            test_img.append(img_vector)
            test_label.append(i)

    return np.array(test_img), np.array(test_label)


def reshapen(theta, input_size, hidden_size, output_size):
    theta1 = np.reshape(theta[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1))
    theta2 = np.reshape(theta[hidden_size * (input_size + 1):], (output_size, hidden_size + 1))
    return np.mat(theta1), np.mat(theta2)


def sample_show(X):
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_image = X[sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(np.array(sample_image[10 * r + c].reshape((28, 28))), cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()


def sigmiod(z):
    return 1 / (1 + np.exp(-z))


def dsigmiod(z):
    return np.multiply(sigmiod(z), (1 - sigmiod(z)))


def randomtheta(lin, lout, epsilon):
    return np.random.random((lout, lin + 1)) * 2 * epsilon - epsilon


def forward_propagation(X, theta, input_size, hidden_size, output_size):
    theta1, theta2 = reshapen(theta, input_size, hidden_size, output_size)
    a1 = np.mat(X)
    z2 = a1 * theta1.T
    a2 = sigmiod(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = a2 * theta2.T
    h = sigmiod(z3)
    return a1, z2, a2, z3, h


def cost(theta, X, y, learningRate, input_size, hidden_size, output_size):
    theta1, theta2 = reshapen(theta, input_size, hidden_size, output_size)
    a1, z2, a2, z3, h = forward_propagation(X, theta, input_size, hidden_size, output_size)
    J = 0
    for i in range(len(X)):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J /= len(X)
    reg = learningRate / (2 * len(X)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return J + reg


def backpropagation(theta, X, y, learningRate, input_size, hidden_size, output_size):
    m = len(X)
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y)
    y = np.mat(y)
    X = np.mat(X)
    a1, z2, a2, z3, h = forward_propagation(X, theta, input_size, hidden_size, output_size)
    theta1, theta2 = reshapen(theta, input_size, hidden_size, output_size)
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    J = cost(theta, X, y, learningRate, input_size, hidden_size, output_size)
    d3 = h - y  # (5000, 10)

    z2 = np.insert(z2, 0, 1, axis=1)  # (5000, 26)
    d2 = np.multiply((theta2.T * d3.T).T, dsigmiod(z2))  # (5000, 26)

    delta1 = delta1 + (d2[:, 1:]).T * a1  # (25,401)
    delta2 = delta2 + d3.T * a2  # (10,26)

    delta1 = delta1 / len(X)
    delta2 = delta2 / len(X)

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learningRate) / len(X)
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learningRate) / len(X)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


def computeNumericalGradient(theta, X, y, learningRate, input_size, hidden_size, output_size):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for i in range(len(theta)):
        perturb[i] = e
        loss1 = cost(theta - perturb, X, y, learningRate, input_size, hidden_size, output_size)
        loss2 = cost(theta + perturb, X, y, learningRate, input_size, hidden_size, output_size)
        numgrad[i] = (loss2 - loss1) / (2 * e)
        perturb[i] = 0
    return numgrad


def result_show(theta, X, y, input_size, hidden_size, output_size):
    a1, z2, a2, z3, h = forward_propagation(X, theta, input_size, hidden_size,
                                            output_size)
    yp = np.array((np.argmax(h, axis=1)))
    print(classification_report(y, yp))


def scan(picture, theta):
    img = Image.open(picture)
    img = img.resize((28, 28))  #修改图片分辨率
    img = img.convert('L')
    img = PIL.ImageOps.invert(img)  #翻转颜色
    img = img.convert('1')  
    img_array = np.asarray(img, 'i')
    img_vector = img_array.reshape(1,img_array.shape[0] * img_array.shape[1])
    img_vector = np.insert(img_vector, 0, 1, axis=1)
    a1, z2, a2, z3, h = forward_propagation(img_vector, theta, input_size, hidden_size,
                                            output_size)
    print("您写的数字为", np.argmax(h), "\n")


if __name__ == "__main__":
    dataX, datay = get_train_pattern()
    idx = np.random.choice(np.arange(dataX.shape[0]), 15000)#由于60000个训练太慢了所以随机抽15000个样本作为训练集
    X = dataX[idx]
    y = datay[idx]
    

    #初始化参数
    input_size = X.shape[1]
    hidden_size = 25
    output_size = 10
    learningRate = 0.1
    epsilon = 0.12
    theta1 = randomtheta(X.shape[1], 25, epsilon)
    theta2 = randomtheta(25, 10, epsilon)
    theta = unroll(theta1, theta2)
    # sample_show(X)
    X = np.insert(X, 0, 1, axis=1)
    
    
    #训练
    result1 = minimize(fun=backpropagation, x0=theta,
                       args=(X, y, learningRate, input_size,
                             hidden_size, output_size),
                       method='TNC', jac=True, options={'maxiter': 250})
    
    
    # 测试
    # test_img, test_label = get_test_pattern()
    # test_img = np.insert(test_img, 0, 1, axis=1)
    # result_show(result1.x, test_img, test_label, input_size, hidden_size, output_size)
    '''
             precision    recall  f1-score   support
          0       0.93      0.97      0.95       980
          1       0.97      0.96      0.97      1135
          2       0.90      0.92      0.91      1032
          3       0.87      0.89      0.88      1010
          4       0.91      0.92      0.91       982
          5       0.88      0.87      0.87       892
          6       0.93      0.93      0.93       958
          7       0.94      0.91      0.92      1028
          8       0.89      0.87      0.88       974
          9       0.89      0.89      0.89      1009
avg / total       0.91      0.91      0.91     10000
    '''
    
    
    # 检测图像并输出
    img1 = 'Y://exercise/Back propagation/mnist训练/3.png'
    img2 = 'Y://exercise/Back propagation/mnist训练/2.png'

    scan(img1, result1.x)

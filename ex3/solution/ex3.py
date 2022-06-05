# 逻辑回归判断数字
import scipy.io as sio
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


def plot_an_image(X):
    # 随机选一行
    pick_one = np.random.randint(0, 5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(image.reshape(20, 20).T, cmap='gray_r')
    # 去除刻度
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_100_images(X):
    sample_idx = np.random.choice(len(X), 100)
    sample_img = X[sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))
    for row in range(10):
        for col in range(10):
            ax_array[row, col].imshow(sample_img[10 * row + col].reshape(20, 20).T, cmap='gray_r')
    plt.xticks([])
    plt.xticks([])
    plt.show()


def load_data(path):
    # 读取mat文件
    data = sio.loadmat(path)
    X = data['X']
    y = data['y']
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def regularized_cost(theta, X, y, lamda):
    A = sigmoid(X @ theta)
    first = y * np.log(A)
    second = (1 - y) * np.log(1 - A)

    # 一维数组时 @是做内积，即平方和
    reg = theta[1:] @ theta[1:] * (lamda / (2 * len(X)))
    return -np.sum(first + second) / len(X) + reg


def gradient_reg(theta, X, y, lamda):
    reg = theta[1:] * (lamda / len(X))
    reg = np.insert(reg, 0, values=0, axis=0)

    first = (X.T @ (sigmoid(X @ theta) - y)) / len(X)
    return first + reg


def one_vs_all(X, y, lamda, K):
    n = X.shape[1]
    theta_all = np.zeros((K, n))

    for i in range(1, K + 1):
        theta_i = np.zeros(n,)
        res = minimize(fun=regularized_cost,
                       x0=theta_i,
                       args=(X, y == i, lamda),
                       method='TNC',
                       jac=gradient_reg)
        theta_all[i - 1, :] = res.x
    return theta_all


def predict(X, theta_final):
    # X shape(5000, 401)
    # theta_final shape(10, 401)
    # h shape(5000, 10)
    h = sigmoid(X @ theta_final.T)
    h_argmax = np.argmax(h, axis=1)
    return h_argmax + 1


def main():
    print('Loading and Visualizing Data ...')
    X, y = load_data('ex3data1.mat')
    # X shape (5000, 400), y shape(5000, 1)
    # 400维特征， 20*20像素点， 5000条样例
    print(X.shape, y.shape)

    # plot_an_image(X, y)
    # plot_100_images(X)

    X = np.insert(X, 0, 1, axis=1)
    # 为了之后的运算，去掉y的一个维度
    y = y.flatten()
    lamda = 1
    K = 10
    theta_final = one_vs_all(X, y, lamda, K)
    # (10, 401) 10个数字，401维特征
    print(theta_final.shape)

    y_pred = predict(X, theta_final)
    acc = np.mean(y_pred == y)
    print(f'prediction accuracy: {acc}')


if __name__ == '__main__':
    main()
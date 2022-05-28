import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost(X, y, theta):
    inner = np.power((X * theta.T - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradient_descent(X, y, theta, alpha, epoch):
    cost = np.zeros(epoch)
    m = X.shape[0]
    for i in range(epoch):
        theta = theta - (alpha / m) * (X * theta.T - y).T * X
        cost[i] = compute_cost(X, y, theta)
    return theta, cost


def main():
    data = pd.read_csv('ex1data2.txt', sep=',', header=None, names=['Size', 'Bedrooms', 'Price'])
    print(data.head())
    print('Normalizing Features ...')
    # 归一化
    data = (data - data.mean()) / data.std()
    # 新增1列，后续计算用，用于表示常量项 x_0
    data.insert(0, 'Ones', 1)

    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:]

    print('Running gradient descent ...')
    X = np.mat(X.values)
    y = np.mat(y.values)
    theta = np.mat([0, 0, 0])
    alpha = 0.01
    epoch = 400

    f_theta, cost = gradient_descent(X, y, theta, alpha, epoch)
    print(f_theta)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(epoch), cost, 'r', label='Iterations')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pause():
    input("Program paused. Press the <ENTER> key to continue...")


def warm_up_exercise():
    # 创建单位矩阵
    A = np.identity(5, dtype=int)
    print(A)


def plot_data(data):
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(8, 5), marker='x')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


def compute_cost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradient_descent(X, y, theta, alpha, epoch):
    cost = np.zeros(epoch)
    m = X.shape[0]
    for i in range(epoch):
        # https://zhuanlan.zhihu.com/p/58182806
        theta = theta - (alpha / m) * (X * theta.T - y).T * X
        cost[i] = compute_cost(X, y, theta)
    return theta, cost


def main():
    print('Running warmUpExercise ...')
    print('5x5 Identity Matrix:')
    warm_up_exercise()
    pause()

    print('Plotting Data ...')
    data = pd.read_csv('ex1data1.txt', sep=',', header=None, names=['Population', 'Profit'])
    # print(data.head())
    # print(data.describe())
    plot_data(data)
    pause()

    # 该列用于后续计算，使得X矩阵变为[1, x_value]
    data.insert(0, 'Ones', 1)
    # 获取列数
    cols = data.shape[1]
    # 获取X数据
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:]

    # 将数据转化为矩阵
    X = np.mat(X.values)
    y = np.mat(y.values)
    # θ参数，此处是线性，所以预设两个
    theta = np.mat([0, 0])
    # α学习率
    alpha = 0.01
    # 训练总轮数
    epoch = 1500
    print('Testing the cost function ...')
    J = compute_cost(X, y, theta)
    print('With theta = [0 ; 0], Cost computed={:.2f}'.format(J))
    print('Expected cost value (approx) 32.07')

    print('Running Gradient Descent ...')
    f_theta, cost = gradient_descent(X, y, theta, alpha, epoch)
    print('Theta found by gradient descent:')
    print(f_theta)
    print('Expected theta values (approx)')
    print('-3.6303, 1.1664')

    # 设置横坐标值
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    # 求取纵坐标值
    y = f_theta[0, 0] + (f_theta[0, 1] * x)
    fig, ax = plt.subplots(figsize=(6, 4))
    # 绘制线性回归曲线
    ax.plot(x, y, 'r', label='Prediction')
    ax.scatter(data['Population'], data['Profit'], label='Training Data', marker='x')
    ax.legend(loc=2)
    ax.set_xlabel('Population of City in 10,000s')
    ax.set_ylabel('Profit in $10,000s')
    ax.set_title('Training data with linear regression fit')
    plt.show()

    # 绘制cost曲线
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(epoch), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Training Epoch')
    plt.show()


if __name__ == '__main__':
    main()

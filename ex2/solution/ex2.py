# 逻辑回归函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report


def sigmod(z):
    # sigmod / logistic函数
    return 1 / (1 + np.exp(-z))


def compute_cost(theta, X, y):
    # 利用@算子可以在两个ndarray中做矩阵运算
    first = y * np.log(sigmod(X @ theta))
    second = (1 - y) * np.log(1 - sigmod(X @ theta))
    return np.mean(- first - second)


def gradient(theta, X, y):
    return (X.T @ (sigmod(X @ theta) - y)) / len(X)


def predict(theta, X):
    probability = sigmod(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]


def main():
    print('Plotting Data ...')
    data = pd.read_csv('ex2data1.txt', sep=',', header=None, names=['exam1', 'exam2', 'admitted'])
    data.insert(0, 'Ones', 1)
    print(data.head())

    # 利用isin过滤数据
    positive = data[data.admitted.isin([1])]
    negative = data[data.admitted.isin([0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(positive['exam1'], positive['exam2'], c='r', marker='x', label='Admitted')
    ax.scatter(negative['exam1'], negative['exam2'], s=50, c='y', label='Not Admitted')
    # get_position 获取图像位置， set_position 重新设置位置
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # loc:位置  bbox_to_anchor:相对位置  ncol:列数
    ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=2)
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    plt.show()

    # X 类型是numpy.ndarray
    # X.shape (100, 3)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    # theta 类型是numpy.ndarray
    # theta.shape (3,)
    theta = np.zeros(X.shape[1])

    # 测试 theta 为零时结果
    initial_cost = compute_cost(theta, X, y)
    print('Initial Cost: {}'.format(initial_cost))
    initial_gradient = gradient(theta, X, y)
    print('Initial Gradient: {}'.format(initial_gradient))

    # 测试 theta 非零时结果
    test_theta = np.array([-24, 0.2, 0.2], dtype=float)
    test_cost = compute_cost(test_theta, X, y)
    print('Test Cost: {}'.format(test_cost))
    test_gradient = gradient(test_theta, X, y)
    print('Test Gradient: {}'.format(test_gradient))

    # result[0] 是 theta 信息 https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_tnc.html
    result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=gradient, args=(X, y), messages=0)
    optimized_theta = result[0]
    print('Optimized Theta: {}'.format(optimized_theta))
    optimized_cost = compute_cost(result[0], X, y)
    print('Optimized Cost: {}'.format(optimized_cost))

    # 测试准确度
    predictions = predict(optimized_theta, X)
    correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
    accuracy = sum(correct) / len(X)
    print('Predictions accuracy is {}'.format(accuracy))

    # sklearn 方法检验
    print(classification_report(predictions, y))

    # 决策边界 利用θTx = 0
    x1 = np.arange(130, step=0.1)
    # theta[0] * 1 + theta[1] * x1 + theta[2] * x2 = 0 推得
    x2 = -(optimized_theta[0] + optimized_theta[1] * x1) / optimized_theta[2]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(positive['exam1'], positive['exam2'], c='r', marker='x', label='Admitted')
    ax.scatter(negative['exam1'], negative['exam2'], s=50, c='y', label='Not Admitted')
    ax.plot(x1, x2)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 130)
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    ax.set_title('Decision Boundary')
    plt.show()


if __name__ == '__main__':
    main()
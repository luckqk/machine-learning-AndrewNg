# 逻辑回归函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmod(z):
    # sigmod / logistic函数
    return 1 / (1 + np.exp(-z))


def compute_cost(theta, X, y):
    # 利用@算子可以在两个
    first = y * np.log(sigmod(X @ theta))
    second = (1 - y) * np.log(1 - sigmod(X @ theta))
    return np.mean(- first - second)


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
    result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=gradient)



if __name__ == '__main__':
    main()
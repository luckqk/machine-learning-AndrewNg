# 逻辑回归函数 正则化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report


def feature_mapping(x1, x2, power):
    # 创建更多的特征
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data[f"f{i - p}{p}"] = np.power(x1, i - p) * np.power(x2, p)
    return pd.DataFrame(data)


def sigmod(z):
    # sigmod / logistic函数
    return 1 / (1 + np.exp(-z))


def compute_cost(theta, X, y):
    # 利用@算子可以在两个ndarray中做矩阵运算
    first = y * np.log(sigmod(X @ theta))
    second = (1 - y) * np.log(1 - sigmod(X @ theta))
    return np.mean(- first - second)


def cost_reg(theta, X, y, l=1):
    # 正则化的代价函数
    # θ 不惩罚第一项
    _theta = theta[1:]
    reg = (l / (2 * len(X))) * (_theta @ _theta)
    return compute_cost(theta, X, y) + reg


def gradient(theta, X, y):
    return (X.T @ (sigmod(X @ theta) - y)) / len(X)


def gradient_reg(theta, X, y, l=1):
    # 正则化梯度
    reg = (l / len(X)) * theta
    reg[0] = 0
    return gradient(theta, X, y) + reg


def predict(theta, X):
    probability = sigmod(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]


def main():
    print('Plotting Data ...')
    data = pd.read_csv('ex2data2.txt', sep=',', header=None, names=['test1', 'test2', 'accepted'])
    data.insert(0, 'Ones', 1)
    print(data.head())

    # 利用isin过滤数据
    positive = data[data.accepted.isin([1])]
    negative = data[data.accepted.isin([0])]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(positive['test1'], positive['test2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['test1'], negative['test2'], s=50, c='r', marker='x', label='Rejected')
    # get_position 获取图像位置， set_position 重新设置位置
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # loc:位置  bbox_to_anchor:相对位置  ncol:列数
    ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=2)
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')
    plt.show()

    x1 = data['test1'].values
    x2 = data['test2'].values
    # feature mapping
    _data = feature_mapping(x1, x2, power=6)
    print(_data.head())

    X = _data.values
    y = data.accepted.values
    theta = np.zeros(X.shape[1])

    # 测试 theta 为零时结果
    initial_cost = cost_reg(theta, X, y, 1)
    print('Initial Cost: {}'.format(initial_cost))
    initial_gradient = gradient_reg(theta, X, y, 1)
    print('Initial Gradient: {}'.format(initial_gradient[0:5]))

    result = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=gradient_reg, args=(X, y, 1), messages=0)
    optimized_theta = result[0]

    # 测试准确度
    predictions = predict(optimized_theta, X)
    correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
    accuracy = sum(correct) / len(X)
    print('Predictions accuracy is {}'.format(accuracy))

    # sklearn 方法检验
    print(classification_report(predictions, y))

    # 计算决策边界，即θTx = 0, 因为x维数过多，无法简单通过曲线去描述符合这个条件的曲线
    # 选取大量采样点，设立阈值，去寻找那些满足θTx < 阈值的点
    density = 1000
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)
    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power=6)
    # 求取 θTx
    inner_product = mapped_cord.values @ optimized_theta
    threshold = 2 * 10 ** -3
    # 选取满足要求的x, y
    decision = mapped_cord[np.abs(inner_product) < threshold]
    x_bound, y_bound = decision.f01, decision.f10

    # 绘制决策边界
    positive = data[data.accepted.isin([1])]
    negative = data[data.accepted.isin([0])]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(positive['test1'], positive['test2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['test1'], negative['test2'], s=50, c='r', marker='x', label='Rejected')
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
    ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=2)
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')
    ax.scatter(x_bound, y_bound, s=10, c='g')
    plt.show()
    # 注意: 修改l的值会影响optimized_theta, 进而影响决策边界


if __name__ == '__main__':
    main()
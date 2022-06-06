import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def one_hot_encoder(raw_y):
    result = []
    for i in raw_y:
        y_temp = np.zeros(10)
        y_temp[i - 1] = 1
        result.append(y_temp)

    return np.array(result)


# 序列化权重参数
def serialize(a, b):
    return np.append(a.flatten(), b.flatten())


# 解序列化权重参数
def deserialize(theta_serialize):
    theta1 = theta_serialize[:25*401].reshape(25, 401)
    theta2 = theta_serialize[25*401:].reshape(10, 26)
    return theta1, theta2


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def feed_forward(theta_serialize, X):
    theta1, theta2 = deserialize(theta_serialize)
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=1, axis=1)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


# 不带正则化的损失函数
def cost(theta_serialize, X, y):
    a1, z2, a2, z3, h = feed_forward(theta_serialize, X)
    J = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / len(X)
    return J


# 带正则化的损失函数
def reg_cost(theta_serialize, X, y, lamda):
    theta1, theta2 = deserialize(theta_serialize)
    sum1 = np.sum(np.power(theta1[:, 1:], 2))
    sum2 = np.sum(np.power(theta2[:, 1:], 2))
    reg = (sum1 + sum2) * lamda / (2 * len(X))
    return reg + cost(theta_serialize, X, y)


# 无正则化的梯度
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def gradient(theta_serialize, X, y):
    theta1, theta2 = deserialize(theta_serialize)
    a1, z2, a2, z3, h = feed_forward(theta_serialize, X)
    d3 = h - y
    d2 = d3 @ theta2[:, 1:] * sigmoid_gradient(z2)
    D2 = (d3.T @ a2) / len(X)
    D1 = (d2.T @ a1) / len(X)
    return serialize(D1, D2)


# 带正则化的梯度
def reg_gradient(theta_serialize, X, y, lamda):
    D = gradient(theta_serialize, X, y)
    D1, D2 = deserialize(D)

    theta1, theta2 = deserialize(theta_serialize)
    D1[:, 1:] = D1[:, 1:] + theta1[:, 1:] * lamda / len(X)
    D2[:, 1:] = D2[:, 1:] + theta2[:, 1:] * lamda / len(X)

    return serialize(D1, D2)


# 无正则化的优化, 容易产生过拟合
def nn_training(X, y):
    init_theta = np.random.uniform(-0.5, 0.5, 10285)
    res = minimize(fun=cost,
                   x0=init_theta,
                   args=(X, y),
                   method='TNC',
                   jac=gradient,
                   options= {'maxiter': 300})

    return res


# 带正则化的优化
def reg_nn_training(X, y, lamda):
    init_theta = np.random.uniform(-0.5, 0.5, 10285)
    res = minimize(fun=reg_cost,
                   x0=init_theta,
                   args=(X, y, lamda),
                   method='TNC',
                   jac=reg_gradient,
                   options= {'maxiter': 300})

    return res


def plot_hidden_layer(theta):
    theta1, _ = deserialize(theta)
    hidden_layer = theta1[:, 1:]

    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8, 8))
    for row in range(5):
        for col in range(5):
            ax_array[row, col].imshow(hidden_layer[5 * row + col].reshape(20, 20).T, cmap='gray_r')
    plt.xticks([])
    plt.xticks([])
    plt.show()


def main():
    data = sio.loadmat('ex4data1.mat')
    raw_X = data['X']
    raw_y = data['y']

    X = np.insert(raw_X, 0, values=1, axis=1)
    print(f'X shape: {X.shape}')

    y = one_hot_encoder(raw_y)

    theta = sio.loadmat('ex4weights.mat')
    # theta1 shape(25, 401)
    # theta2 shape(10, 26)
    theta1, theta2 = theta['Theta1'], theta['Theta2']
    theta_serialize = serialize(theta1, theta2)

    curt_cost = cost(theta_serialize, X, y)
    print(f'cost: {curt_cost}')
    lamda = 1
    curt_reg_cost = reg_cost(theta_serialize, X, y, lamda)
    print(f'reg cost: {curt_reg_cost}')

    lamda = 10
    res = reg_nn_training(X, y, lamda)
    raw_y = data['y'].reshape(5000,)
    _, _, _, _, h = feed_forward(res.x, X)
    y_pred = np.argmax(h, axis=1) + 1
    acc = np.mean(y_pred == raw_y)
    print(f'accuracy: {acc}')

    plot_hidden_layer(res.x)


if __name__ == '__main__':
    main()
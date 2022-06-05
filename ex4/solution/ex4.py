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
    theta1, theta2 = deserialize(theta_serialize)
    print(theta1.shape)
    print(theta2.shape)



if __name__ == '__main__':
    main()
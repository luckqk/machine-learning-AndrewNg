# 神经网络
import numpy as np
import scipy.io as sio


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def main():
    data = sio.loadmat('ex3data1.mat')
    raw_X = data['X']
    raw_y = data['y']

    X = np.insert(raw_X, 0, values=1, axis=1)
    y = raw_y.flatten()
    print(y.shape)

    theta = sio.loadmat('ex3weights.mat')
    theta1 = theta['Theta1']
    theta2 = theta['Theta2']

    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)

    a2 = np.insert(a2, 0, values=1, axis=1)

    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)

    y_pred = np.argmax(a3, axis=1)
    y_pred = y_pred + 1

    acc = np.mean(y_pred == y)
    print(f'prediction accuracy {acc}')


if __name__ == '__main__':
    main()
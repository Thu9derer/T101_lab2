from random import random

import numpy as np
import matplotlib.pyplot as plt
from time import time
# generates x and y numpy arrays for
# y = a*x + b + a * noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# vizualizes it and unloads to csv
def generate_linear(a, b, noise, filename, size=100):
    print('Generating random data y = a*x + b')
    x = 2 * np.random.rand(size, 1) - 1
    y = a * x + b + noise * a * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')
    return (x, y)


# thats an example of linear regression using polyfit
def linear_regression_numpy(filename):
    # now let's read it back
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    # split to initial arrays
    x, y = np.hsplit(data, 2)
    # printing shapes is useful for debugging
    print(np.shape(x))
    print(np.shape(y))
    # our model
    time_start = time()
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")
    # our hypothesis for give x
    h = model[0] * x + model[1]

    # and check if it's ok
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    print(model)
    return model


def linear_regression_exact(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    one_line = np.ones((100, 1))
    new_x = np.hstack([one_line, x])  # добавление столбца с еденицами
    tran_x = new_x.transpose()
    time_start = time()
    mass_theta = np.linalg.pinv(tran_x.dot(new_x)).dot(tran_x).dot(y)
    time_end = time()

    print(f"pinv in {time_end - time_start} seconds")
    h = mass_theta[1] * x + mass_theta[0]

    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    print("Ex1: your code here - exact solution usin invert matrix")
    mass_theta_t = mass_theta.transpose()
    mass_theta_t[:, [1, 0]] = mass_theta_t[:, [0, 1]]  # замена местами коэффицентов т.к. check ругается
    return mass_theta_t[0]


def check(model, ground_truth):
    if len(model) != len(ground_truth):
        print("Model is inconsistent")
        return False
    else:
        r = np.dot(model - ground_truth, model - ground_truth) / (np.dot(ground_truth, ground_truth))
        print(r)
        if r < 0.0001:
            return True
        else:
            return False


# Ex1: make the same with polynoms

# generates x and y numpy arrays for
# y = a_n*X^n + ... + a2*x^2 + a1*x + a0 + noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# vizualizes it and unloads to csv
def generate_poly(a, n, noise, filename, size=100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size, 1))
    print(np.shape(x))
    print(np.shape(y))
    if len(a) != (n + 1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) must be the same as polynomial degree {n}')
        return
    for i in range(0, n + 1):
        y = y + a[i] * np.power(x, i) + noise * (np.random.rand(size, 1) - 0.5)
    print(np.shape(x))
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')


def polynomial_regression_numpy(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    # split to initial arrays
    x, y = np.hsplit(data, 2)

    # printing shapes is useful for debugging
    print(np.shape(x))
    print(np.shape(y))
    # our model
    time_start = time()
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 2)
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")
    # our hypothesis for give x
    # and check if it's ok
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    x = np.sort(x, axis=0)
    h = model[0] * x * x + model[1] * x + model[2]
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return model


# Ex.2 gradient descent for linear regression without regularization

# find minimum of function J(theta) using gradient descent
# alpha - speed of descend
# theta - vector of arguments, we're looking for the optimal ones (shape is 1 х N)
# J(theta) function which is being minimizing over theta (shape is 1 x 1 - scalar)
# dJ(theta) - gradient, i.e. partial derivatives of J over theta - dJ/dtheta_i (shape is 1 x N - the same as theta)
# x and y are both vectors

def gradient_descent_step(dJ, theta, alpha, size):
    theta_new = theta - alpha * 1/size * dJ
    return theta_new


# get gradient over all xy dataset - gradient descent
def get_dJ(x, y, theta):
    h = theta.dot(x.transpose())
    dJ = (h - y).dot(x)
    return dJ


# get gradient over all minibatch of size M of xy dataset - minibatch gradient descent
def get_dJ_minibatch(x, y, theta):
    h = theta.dot(x.transpose())
    dJ = (h - y).dot(x)
    return dJ


# get gradient over all minibatch of single sample from xy dataset - stochastic gradient descent
def get_dJ_sgd(x, y, theta):
    h = theta.dot(x.transpose())
    dJ = (h - y).dot(x)
    return dJ


# try each of gradient decsent (complete, minibatch, sgd) for varius alphas
# L - number of iterations
# plot results as J(i)
def minimize(x, y, L, degree, size):
    alpha = 0.205
    theta = np.ones((1, degree + 1))  # you can try random initialization
    for i in range(0, L):
        dJ = get_dJ(x, y, theta)  # here you should try different gradient descents
        theta = gradient_descent_step(dJ, theta, alpha, size)
        alpha -= 0.00005
        h = theta.dot(x.transpose())
        J = 1/ (2 * size) * (np.square(h - y)).sum(axis=1)  # here you should calculate it properly
        plt.plot(i, J, "b.")
    plt.legend()
    plt.show()
    return theta


def minimize_minibatch(x, y, L, M, degree, size):  # M-size minibatch
    alpha = 0.15
    theta = np.ones((1, degree + 1))  # you can try random initialization
    size_minib = np.shape(x)[0] / M
    x = np.vsplit(x, size_minib)
    y = np.hsplit(y, size_minib)
    for i in range(0, L):
        index = int(size_minib * random())
        x_minib = x[index]
        y_minib = y[index]
        dJ = get_dJ_minibatch(x_minib, y_minib, theta)  # here you should try different gradient descents
        theta = gradient_descent_step(dJ, theta, alpha, size)
        alpha -= 0.00002
        h = theta.dot(x_minib.transpose())
        J = 1 / (2 * size) * (np.square(h - y_minib)).sum(axis=1)  # here you should calculate it properly
        plt.plot(i, J, "b.")
    plt.legend()
    plt.show()
    return theta


def minimize_sgd(x, y, L, degree, size):
    alpha = 0.20
    theta = np.ones((1, degree + 1))  # you can try random initialization
    for iter in range(0, L):
        index = int(size * random())
        line_x = np.reshape(x[index], (1, degree + 1))
        one_y = np.reshape(y[0][index], (1, 1))
        dJ = get_dJ_sgd(line_x, one_y, theta)  # here you should try different gradient descents
        theta = gradient_descent_step(dJ, theta, alpha, size)
        alpha -= 0.00002
        h = theta.dot(line_x.transpose())
        J = 1/(2 * size) * (np.square(h - one_y))  # here you should calculate it properly
        plt.plot(iter, J, "b.")
    plt.legend()
    plt.show()
    return theta


if __name__ == "__main__":
    # generate_linear(1, -3, 1, 'linear.csv', 100)
    # model_np = linear_regression_numpy("linear.csv")
    # print(f"Is model correct?\n{check(model_np, np.array([1, -3]))}")
    # mass_theta_np = linear_regression_exact("linear.csv")
    # print(f"Is model correct?\n{check(mass_theta_np, np.array([1, -3]))}")
    # # ex1 . - exact solution
    # model_exact = linear_regression_exact("linear.csv")
    # check(model_exact, np.array([-3,1]))
    #
    # # ex1. polynomial with numpy
    # generate_poly([1, 2, 3], 2, 0.5, 'polynomial.csv')
    # polynomial_regression_numpy("polynomial.csv")

    # ex2. find minimum with gradient descent

    # 0. generate date with function above
    size = 100
    generate_linear(3, -5, 1, 'linear.csv', size)

    # 1. shuffle data into train - test - valid
    with open('linear.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    size_train_data = int(size / 10 * 8)
    size_test_data = int(size / 10)
    train_data = data[:size_train_data]
    test_data = data[size_train_data:size_train_data+size_test_data:1]
    valid_data = data[size_train_data+size_test_data::1]
    x, y = np.hsplit(train_data, 2)
    one_col = np.ones((size_train_data, 1))
    x = np.hstack([one_col, x])
    # 2. call minuimize(...) and plot J(i)
    y = y.transpose()
    degree = 1
    print(minimize(x, y, 150, degree, size_train_data))

    print(minimize_sgd(x, y, 50, degree, size_train_data))

    print(minimize_minibatch(x, y, 60, 5, degree, size_train_data))
    # 3. call check(theta1, theta2) to check results for optimal theta

    # ex3. polinomial regression
    # 0. generate date with function generate_poly for degree=3, use size = 10, 20, 30, ... 100
    size = 100
    degree = 3
    generate_poly([1, 2, 3, 4], degree, 0.5, 'polynomial.csv', 100)
    with open('polynomial.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')

    x, y = np.hsplit(data, 2)
    full_x = x
    for i in range(2, degree + 1):
        full_x = np.hstack([full_x, x**i])
    one_col = np.ones((size, 1))
    x = np.hstack([one_col, full_x])
    y = y.transpose()
    print(minimize(x, y, 3000, degree, size))
    print(minimize_sgd(x, y, 8000, degree, size))
    print(minimize_minibatch(x, y, 800, 5, degree, size))
    # for each size:
    # 1. shuffle data into train - test - valid
    # Now we're going to try different degrees of model to aproximate our data, set degree=1 (linear regression)
    # 2. call minimize(...) and plot J(i)
    # 3. call check(theta1, theta2) to check results for optimal theta
    # 4. plot min(J_train), min(J_test) vs size: is it overfit or underfit?
    #
    # repeat 0-4 for degres = 2,3,4

    # ex3* the same with regularization
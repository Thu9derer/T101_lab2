import numpy as np
import matplotlib.pyplot as plt
from time import time
from sympy import *


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

def gradient_descent_step(dJ, theta, alpha):
    theta_new = theta - alpha * 1/60 * dJ
    return theta_new


# get gradient over all xy dataset - gradient descent
def get_dJ(x, y, theta):
    h = (theta.transpose()).dot(x.transpose()).transpose()
    dJ = ((h - y).transpose()).dot(x)
    return dJ


# get gradient over all minibatch of size M of xy dataset - minibatch gradient descent
def get_dJ_minibatch(x, y, theta, M):
    theta_new = theta
    print("your code goes here - calculate new theta")
    return theta_new


# get gradient over all minibatch of single sample from xy dataset - stochastic gradient descent
def get_dJ_sgd(x, y, theta):
    theta_new = theta
    print("your code goes here - calculate new theta")
    return theta_new


# try each of gradient decsent (complete, minibatch, sgd) for varius alphas
# L - number of iterations
# plot results as J(i)
def minimize(x, y, L):
    alpha = 0.01
    # n - number of samples in learning subset, m - ...
    n = 2  # <-- calculate it properly!
    theta = np.ones((n, 1))  # you can try random initialization
    for i in range(0, L):
        dJ = get_dJ(x, y, theta)  # here you should try different gradient descents
        gradient_descent_step(dJ, theta, alpha)
        J = 0  # here you should calculate it properly
    # and plot J(i)
    print("your code goes here")
    return


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
    generate_linear(1, -3, 1, 'linear.csv', 100)

    # 1. shuffle data into train - test - valid
    with open('linear.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    train_data = data[:60]
    test_data = data[60::1]
    valid_data = data[80::1]
    x, y = np.hsplit(train_data, 2)
    one_col = np.ones((60, 1))
    x = np.hstack([one_col, x])
    # 2. call minuimize(...) and plot J(i)

    minimize(x, y, 5)
    # 3. call check(theta1, theta2) to check results for optimal theta

    # ex3. polinomial regression
    # 0. generate date with function generate_poly for degree=3, use size = 10, 20, 30, ... 100
    # for each size:
    # 1. shuffle data into train - test - valid
    # Now we're going to try different degrees of model to aproximate our data, set degree=1 (linear regression)
    # 2. call minimize(...) and plot J(i)
    # 3. call check(theta1, theta2) to check results for optimal theta
    # 4. plot min(J_train), min(J_test) vs size: is it overfit or underfit?
    #
    # repeat 0-4 for degres = 2,3,4

    # ex3* the same with regularization
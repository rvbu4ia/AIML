import sys
import numpy as np

x1 = np.array([60, 67, 71, 75, 78])
x2 = np.array([22, 24, 15, 20, 16])
y = np.array([140, 159, 192, 200, 212])

w0, w1, w2 = (0, 1, 1)

alpha = 0.0002

def hypothesis (x1, x2, w0, w1, w2):
    return (w0 + w1*x1 + w2*x2)

def mse_cost_func (y, y_pred):
    return np.mean((y - y_pred) ** 2)/2

def gradient_descent (x1, x2, y, w0, w1, w2, alpha, iter = 0):
    w0_new, w1_new, w2_new = w0, w1, w2
    i = 0
    while True:
        y_pred = hypothesis(x1, x2, w0_new, w1_new, w2_new)
        prev_mse = mse_cost_func(y, y_pred)
        w0_new += alpha * np.mean(y - y_pred)
        w1_new += alpha * np.mean((y - y_pred) * x1)
        w2_new += alpha * np.mean((y - y_pred) * x2)
        new_mse = mse_cost_func(y, hypothesis(x1, x2, w0_new, w1_new, w2_new))
        print(f"Values of w0, w1, w2 and MSE after iter {i+1} are:\n\tw0 = {w0_new}\n\tw1 = {w1_new}\n\tw2 = {w2_new}\n\tMSE = {new_mse}")
        if (i + 1 == iter) or (round(float(prev_mse - new_mse), 4) <= 0.0001) :
            break
        i += 1
    print("Final prediction of y: ", hypothesis(x1, x2, w0_new, w1_new, w2_new))
    print("Final MSE: ", new_mse)

if __name__ == '__main__':
    iter = int(input("Enter no of preferred iterations. Enter '0' if no specific preference: "))
    print("Initial prediction of y: ", hypothesis(x1, x2, w0, w1, w2))
    print("Initial MSE: ", mse_cost_func(y, hypothesis(x1, x2, w0, w1, w2)))
    gradient_descent(x1, x2, y, w0, w1, w2, alpha, iter)

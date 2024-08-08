import sys

x1 = [60, 67, 71, 75, 78]
x2 = [22, 24, 15, 20, 16]
y = [140, 159, 192, 200, 212]

w0, w1, w2 = (0, 1, 1)

alpha = 0.0002

def hypothesis (x1, x2, w0, w1, w2):
    y_pred = []
    for i in range(len(y)):
        y_pred.append(w0 + w1 * x1[i] + w2 * x2[i])
    return y_pred

def mse_cost_func (y, y_pred):
    mse = 0
    for i in range(len(y)):
        mse += (y[i] - y_pred[i]) ** 2
    return (mse / (2*len(y)))

def gradient_descent (x1, x2, y, w0, w1, w2, alpha, iter = 0):
    w0_new, w1_new, w2_new = w0, w1, w2
    i = 0
    while True:
        y_new = hypothesis(x1, x2, w0_new, w1_new, w2_new)
        prev_mse = mse_cost_func(y, y_new)
        for j in range(len(y)):
            w0_new += alpha * (y[j] - y_new[j]) / len(y)
            w1_new += alpha * (y[j] - y_new[j]) * x1[j] / len(y)
            w2_new += alpha * (y[j] - y_new[j]) * x2[j] / len(y)
        new_mse = mse_cost_func(hypothesis(x1, x2, w0_new, w1_new, w2_new), y)
        print(f"Values of w0, w1, w2 and MSE after iter {i+1} are:\n\tw0 = {w0_new}\n\tw1 = {w1_new}\n\tw2 = {w2_new}\n\tMSE = {new_mse}")
        if round(float(prev_mse - new_mse), 4) <= 0.0001 or (i + 1 == iter):
            break
        i += 1
    print("Final prediction of y: ", hypothesis(x1, x2, w0_new, w1_new, w2_new))
    print("Final MSE: ", mse_cost_func(y, hypothesis(x1, x2, w0_new, w1_new, w2_new)))

if __name__ == '__main__':
    iter = int(input("Enter no of preferred iterations. Enter '0' if no specific preference: "))
    print("Initial prediction of y: ", hypothesis(x1, x2, w0, w1, w2))
    print("Initial MSE: ", mse_cost_func(y, hypothesis(x1, x2, w0, w1, w2)))
    gradient_descent(x1, x2, y, w0, w1, w2, alpha, iter)

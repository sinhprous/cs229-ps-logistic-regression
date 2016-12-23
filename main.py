import numpy as np
import matplotlib.pyplot as plt

def load(file_name):
    vectors = []
    file = open(file_name)
    lines = file.readlines()
    for line in lines:
        items = line.split()
        if (len(items) == 1):
            vectors.append(float(items[0]))
        elif (len(items) > 1):
            vector = []
            for item in items:
                vector.append(float(item))
            vectors.append(vector)
    return np.array(vectors)

def g(z):
    return 1.0/(1+np.exp(-1*z))


x_train = load("logistic_x")
y_train = load("logistic_y")
#bias = [[1]]*x_train.shape[0]
#x_train = np.hstack((bias,x_train))

'''
w = np.array([0.0 for i in range(len(x_train[1]))])

for num_train in range(1000):
    hessian = np.array([[np.mean([g(y_train[i]* w.dot(x_train[i]))*(1-g(y_train[i]* w.dot(x_train[i])))*x_train[i][j]*x_train[i][k] for i in range(x_train.shape[0])]) for j in range(x_train.shape[1])]for k in range(x_train.shape[1])])
    gradient = -1*np.mean(np.array([[np.exp(-y_train[i]* w.dot(x_train[i]))*g(y_train[i]* w.dot(x_train[i]))*y_train[i]*x_train[i][k] for k in range(x_train.shape[1])] for i in range(len(x_train))]), axis=0)
    w -= np.linalg.inv(hessian).dot(gradient)
print(w)
'''

w = np.array([-2.6205116, 0.76037154, 1.17194674])
x = np.arange(-10, 10, 0.01)
y = -1/w[2] * (w[1]*x+w[0])
for i in range(len(x_train)):
    if y_train[i] == 1:
        plt.plot(x_train[i,0], x_train[i,1], 'ro', color='yellow')
    else:
        plt.plot(x_train[i,0], x_train[i,1], 'ro', color='red')
print(len(x_train))
plt.plot(x, y)
plt.show()

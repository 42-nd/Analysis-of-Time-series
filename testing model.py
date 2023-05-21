import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import tensorflow as tf
import math as m

pi = tf.constant(m.pi, dtype=tf.float64)


def f(x, tet):
    return tet[0] + tet[1] * x + tet[2] * tf.math.pow(x, 2) + tet[3] * tf.math.sin((((2 * pi * x) / tet[4]) + tet[5]))


def loss_f(tet, x, y):
    return tf.reduce_sum(tf.math.pow(f(x, tet) - y, 2))


def Sa0(y_pred, y):
    return 2 * np.sum((y_pred - y))


def Sa1(y_pred, x, y):
    return 2 * np.sum((y_pred - y) * x)


def Sa2(y_pred, x, y):
    return 2 * np.sum((y_pred - y) * x ** 2)

def Sa3(y_pred, tet,x, y):
    return 2 * np.sum((y_pred - y)*tf.math.sin(((2*pi*x)/tet[4])+tet[5]))

def Sa4(y_pred, tet,x, y):
    return 2 * np.sum((y_pred - y)* tet[3] * tf.math.cos(((2*pi*x)/tet[4])+tet[5]) * ((-2*pi*x)/(tet[4]*tet[4])))

def Sa5(y_pred, tet,x, y):
    return 2 * np.sum((y_pred - y) * tet[3] * tf.math.cos(((2 * pi*x) / tet[4]) + tet[5]))

def gradient_descent(x, y):
    results = {}
    n_points = np.linspace(1, 10.0, num=4)
    for j in n_points:
        tet_curr = np.ones(6)
        tet_curr[4] = j
        tet_curr = tf.Variable(tet_curr, dtype=tf.float64)
        dtet = np.zeros(6)
        iterations = 2000
        learning_rate = 0.0001
        for i in range(iterations):
            y_pred = f(x, tet_curr)
            dtet[0] = Sa0(y_pred, y)
            dtet[1] = Sa1(y_pred, x, y)
            dtet[2] = Sa2(y_pred, x, y)
            dtet[3] = Sa3(y_pred, tet_curr,x, y)
            dtet[4] = Sa4(y_pred, tet_curr,x, y)
            dtet[5] = Sa5(y_pred, tet_curr,x, y)

            loss = loss_f(tet_curr, x, y)
            tet_curr.assign_sub(learning_rate * dtet)

        results[loss.numpy()] = [tet_curr]
        print("------------------------------")
        for key, value in results.items():
            print("{0}: {1}".format(key,value))
    return results[min(list(results.keys()))][0]


def gradient_descent_with_AD(x, y):
    results = {}
    n_points = np.linspace(1, 10.0, num=4)
    for j in n_points:
        tet_curr = np.ones(6)
        tet_curr[4] = j
        tet_curr = tf.Variable(tet_curr, dtype=tf.float64)

        iterations = 2000
        learning_rate = 0.0001

        for i in range(iterations):
            with tf.GradientTape() as tape:
                loss = loss_f(tet_curr, x, y)
            dtet = tape.gradient(loss, tet_curr)
            tet_curr.assign_sub(learning_rate * dtet)

            # print ("tet0 {} ,tet1 {} ,tet2 {}".format (tet_curr[0],tet_curr[1],tet_curr[2]))
            # print ("tet3 {} ,tet4 {} ,tet5 {} ,loss {}".format (tet_curr[3],tet_curr[4],tet_curr[5],loss))
        results[loss.numpy()] = [tet_curr]
        print("------------------------------")
        for key, value in results.items():
            print("{0}: {1}".format(key,value))
        #print("!!!",results[min(list(results.keys()))],"!!!")
    return results[min(list(results.keys()))][0]


a = tf.Variable(np.array([4.0, 0, 0.3, 5.0, 5.0, 2.0]), dtype=tf.float64)
x = tf.Variable(np.array([x / 100.0 for x in range(-400, 408, 8)]), dtype=tf.float64)
epsi = tf.Variable(np.random.normal(0, 0.5 ** 2, x.shape[0]), dtype=tf.float64)
y = f(x, a)
samples = y + epsi

'''Метод нулевого порядка'''

results = {}
n_points = np.linspace(1, 10.0, num=4)
for j in n_points:
    tet = np.ones(a.shape[0])
    tet[4] = j
    tet = tf.Variable(tet, dtype=tf.float64)
    tet_pred = sc.optimize.minimize(loss_f, tet, args=(x, samples), method='Nelder-Mead').x
    loss = loss_f(tet_pred, x, samples)
    results[loss.numpy()] = [tet_pred]
tet_pred = results[min(list(results.keys()))][0]

plt.plot(x, f(x, tet_pred), color='g', label='Нулевой порядок')
print(tet_pred, loss_f(tet_pred, x, samples))
print("--------------------")

'''Метод первого порядка без АД'''

tet_pred_gd = gradient_descent(x, samples)
plt.plot(x, f(x, tet_pred_gd), color='black', label='Первый порядок без АД')

print(tet_pred_gd, loss_f(tet_pred_gd, x, samples))
print("--------------------")

'''Метод первого порядка с АД'''

tet_pred_ad = gradient_descent_with_AD(x, samples)
plt.plot(x, f(x, tet_pred_ad), color='pink', label='Первый порядокм АД')

print(tet_pred_ad, loss_f(tet_pred_ad, x, samples))
print("--------------------")

ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.plot(x, y, label='Идеальный график')
plt.scatter(x, samples, marker=".", color="red", label='Идеальный график + шум')

plt.legend()
plt.show()

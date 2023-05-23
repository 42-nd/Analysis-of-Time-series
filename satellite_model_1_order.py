import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from SVD import hsvd
PI = tf.constant(np.pi, dtype=tf.float64)

def f(X, tet):
    ret = tet[0] + tet[1] * X + tet[2] * tf.math.pow(X, 2)
    return ret

def loss_f(tet, X, y):
    return tf.reduce_sum(tf.math.pow(f(X, tet) - y, 2))

def gradient_descent_with_AD(X, y):
    
    iterations = 10000
    learning_rate = 0.000000000000000001

    results = {}
    n_points = np.linspace(1, 10, num=3)
    for j in n_points:
        tet_curr = np.ones(3)*j
        tet_curr = tf.Variable(tet_curr, dtype=tf.float64)
        for i in range(iterations):
            with tf.GradientTape() as tape:
                loss = loss_f(tet_curr, X, y)
            dtet = tape.gradient(loss, tet_curr)
            tet_curr.assign_sub(learning_rate * dtet)
            if (tet_curr.numpy()[0] != tet_curr.numpy()[0]):
                break
            print(tet_curr.numpy(),i)
        print("------------------------------------------------")
        results[loss.numpy()] = [tet_curr]

    return results[min(list(results.keys()))][0]


df = pd.read_csv('D:\Github\Analysis-of-Time-series\Файлы данных РШВ с восстановленными пропусками(365 дней)\R01_365.csv', parse_dates=['ds'])[:500]

X = tf.Variable(np.arange(df['ds'].shape[0]),dtype=tf.float64)
y = np.array(df['y'])*1e6

window = 300
rank = 10
l, h = hsvd(y, window, rank)
y = tf.Variable(l,dtype=tf.float64)

'''First order method with AD'''

tet_pred_ad = gradient_descent_with_AD(X,y)
plt.plot(X, f(X, tet_pred_ad), color='pink')

print(tet_pred_ad, loss_f(tet_pred_ad, X,y))

plt.plot(X, y)

plt.legend()
plt.show()

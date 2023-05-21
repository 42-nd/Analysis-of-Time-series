import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import tensorflow as tf
import math as m
import pandas as pd
from sklearn.model_selection import train_test_split
pi = tf.constant(m.pi, dtype=tf.float64)

def f(x, tet):
    return tet[0] + tet[1] * x + tet[2] * tf.math.pow(x, 2) 

def loss_f(tet, x, y):
    return tf.reduce_sum(tf.math.pow(f(x, tet) - y, 2))

def gradient_descent_with_AD(x, y):
    tet_curr = np.ones(3)
    tet_curr = tf.Variable(tet_curr, dtype=tf.float64)
    iterations = 2000
    learning_rate = 0.000001

    for i in range(iterations):
        with tf.GradientTape() as tape:
            loss = loss_f(tet_curr, x, y)
        dtet = tape.gradient(loss, tet_curr)
        tet_curr.assign_sub(learning_rate * dtet)

        #print ("tet0 {} ,tet1 {} ,tet2 {}".format (tet_curr[0],tet_curr[1],tet_curr[2]))
            # print ("tet3 {} ,tet4 {} ,tet5 {} ,loss {}".format (tet_curr[3],tet_curr[4],tet_curr[5],loss))
    return tet_curr

df = pd.read_csv('R01_365.csv', parse_dates=['ds'])[:100000]

x = tf.Variable((np.arange(df['ds'].shape[0])), dtype=tf.float64)

samples = tf.Variable(df['y'], dtype=tf.float64)

tet = tf.Variable((np.zeros(3)), dtype=tf.float64)

'''Метод нулевого порядка'''
#et_pred = sc.optimize.minimize(loss_f, tet, args=(x, samples), method='Nelder-Mead').x

#plt.plot(x[:10000], f(x, tet_pred)[:10000], color='g', label='Нулевой порядок')
#print(tet_pred, loss_f(tet_pred, x, samples))
#print("--------------------")


'''Метод первого порядка с АД'''

#tet_pred_ad = gradient_descent_with_AD(x, samples)
#plt.plot(x, f(x, tet_pred_ad), color='pink', label='Первый порядокм АД')

#print(tet_pred_ad, loss_f(tet_pred_ad, x, samples))
#print("--------------------")



plt.plot(x[:10000], samples[:10000])

plt.legend()
plt.show()

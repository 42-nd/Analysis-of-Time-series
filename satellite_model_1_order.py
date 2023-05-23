import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import tensorflow as tf
import pandas as pd
PI = tf.constant(np.pi, dtype=tf.float64)

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

df = pd.read_csv('D:\Github\Analysis-of-Time-series\Файлы данных РШВ с восстановленными пропусками(365 дней)\R01_365.csv', parse_dates=['ds'])[:100000]

X = np.arange(df['ds'].shape[0])*30

y = np.array(df['y'])*1e6

'''First order method with AD'''

#tet_pred_ad = gradient_descent_with_AD(x, samples)
#plt.plot(x, f(x, tet_pred_ad), color='pink', label='Первый порядокм АД')

#print(tet_pred_ad, loss_f(tet_pred_ad, x, samples))
#print("--------------------")

plt.plot(X[:10000], y[:10000])

plt.legend()
plt.show()

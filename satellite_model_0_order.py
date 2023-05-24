import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
from SVD import hsvd
PI = np.pi
def f(X, tet):
    ret = tet[0] + tet[1] * X + tet[2] * X**2 
    for i in range(3,14,3):
        ret += tet[i] * np.sin((2 * PI * X)/tet[i+1] + tet[i+2])
    return ret

def loss_f(tet, X, y):
    return np.sum((f(X, tet) - y)**2)

df = pd.read_csv('D:\Github\Analysis-of-Time-series\Файлы данных РШВ с восстановленными пропусками(365 дней)\R01_365.csv', parse_dates=['ds'])[:5000]
X = np.arange(df['ds'].shape[0])
y = np.array(df['y'])*1e6

window = 500    
rank = 4
l, h = hsvd(y, window, rank)
y = l
'''The zero-order method'''
results = {}
n_points = np.linspace(1, 10.0, num=3)
for j in n_points:
    tet = np.ones(15)*j
    tet_pred = sc.optimize.minimize(loss_f, tet, args=(X, y), method='Powell').x
    loss = loss_f(tet_pred, X, y)
    results[loss] = [tet_pred]

tet_pred = results[min(list(results.keys()))][0]
print(tet_pred, loss_f(tet_pred, X, y))
plt.plot(X,f(X,tet_pred),c='g')
plt.plot(X,f(X,tet_pred)+h,c='black')
plt.plot(X, np.array(df['y'])*1e6)

plt.show()

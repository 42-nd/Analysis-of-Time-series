import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
from SVD import hsvd
PI = np.pi
def f(X, tet):
    ret = tet[0] + tet[1] * X + tet[2] * X**2 
    ret += tet[3] * np.sin((2 * PI * X)/tet[4] + tet[5])
    return ret

def loss_f(tet, X, y):
    return np.sum((f(X, tet) - y)**2)

df = pd.read_csv('D:\Github\Analysis-of-Time-series\Файлы данных РШВ с восстановленными пропусками(365 дней)\R01_365.csv', parse_dates=['ds'])[:5000]
X = np.arange(df['ds'].shape[0])
y = np.array(df['y'])*1e6
tet = np.ones(6)
window = 300
rank = 4
l, h = hsvd(y, window, rank)
y = l
'''The zero-order method'''
tet_pred = sc.optimize.minimize(loss_f, tet, args=(X, y), method='CG').x
plt.plot(X, f(X, tet_pred), color='g', label='Zero-order')
print(tet_pred, loss_f(tet_pred, X, y))

plt.plot(X, y)

plt.legend()
plt.show()


'''
def f(X, tet):
    ret = tet[0] + tet[1] * X + tet[2] * X**2 
    for i in range(3,14,3):
        ret += tet[i] * np.sin((2 * PI * X)/tet[i+1] + tet[i+2])
    return ret'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def f(x, tet):
    return tet[0] + tet[1] * x + tet[2] * x**2

def rls(x,y):
    tet = np.zeros(3)
    P = np.eye(3) *1e6
    lam = 1.1
    for i in range (x.shape[0]):
        phi = np.array([x[i]**2, x[i], 1])
        K = np.dot(P, phi) / (lam + np.dot(np.dot(phi.T, P), phi))
        P = (P - np.outer(K, np.dot(phi.T, P))) / lam
        tet = tet + K * (y[i] - np.dot(phi, tet))
    return tet[::-1]

df = pd.read_csv('R01_365.csv', parse_dates=['ds'])[:100000]
x = np.arange(df['ds'].shape[0])
y = df['y']
tet = rls(x,y)
print(tet)
plt.plot(x[:10000], y[:10000],color ='blue')
plt.plot(x[:10000], f(x, tet)[:10000],color ='g')

plt.show()


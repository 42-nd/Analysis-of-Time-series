import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
from SVD import hsvd


def Y0(X, tet):
    ret = tet[0] + tet[1] * X + tet[2] * X**2
    for i in range(3, 14, 3):
        ret += tet[i] * np.sin((2 * np.pi * X) / tet[i + 1] + tet[i + 2])  
        # tet[i] = Bk, tet[i+1] = Pk, tet[i+2] = phik for 4 steps accordingly
    return ret


def Y1(X, tet):
    ret = tet[0] + tet[1] * X + tet[2] * X**2
    for i in range(3, 14, 3):
        ret += tet[i] * np.sin((2 * np.pi * X) / tet[i + 1]) + tet[i + 2] * np.cos((2 * np.pi * X) / tet[i + 1])  
        # tet[i] = Сk, tet[i+1] = Pk, tet[i+2] = Dk for 4 steps accordingly
    return ret


def Y2(X, tet):
    ret = tet[0] + tet[1] * X + tet[2] * X**2
    for i in range(3, 14, 3):
        ret += tet[i] * np.sin(X * tet[i + 1]) + tet[i + 2] * np.cos(X * tet[i + 1])  
        # tet[i] = Сk, tet[i+1] = wk, tet[i+2] = Dk for 4 steps accordingly
    return ret


def loss_f(tet, X, y):
    return np.sum((Y0(X, tet) - y) ** 2)


df = pd.read_csv(
    "D:\Github\Analysis-of-Time-series\Файлы данных РШВ с восстановленными пропусками(365 дней)\R01_365.csv",
    parse_dates=["ds"],
)[:5000]
X = np.arange(df["ds"].shape[0])
y = np.array(df["y"]) * 1e6

window = 500
rank = 4
l, h = hsvd(y, window, rank)
y = l

tet_parameters = np.ones(15) * 7
tet_pred_parameters_Y0 = sc.optimize.minimize(loss_f, tet_parameters, args=(X, y), method="Powell").x
loss_Y0 = loss_f(tet_pred_parameters_Y0, X, y)

print(tet_pred_parameters_Y0, loss_Y0, "LOSS Y0")
plt.plot(X, Y0(X, tet_pred_parameters_Y0), c="g", label="Y0")


tet_parameters = np.ones(15) * 7
tet_pred_parameters_Y1 = sc.optimize.minimize(loss_f, tet_parameters, args=(X, y), method="Powell").x
loss_Y1 = loss_f(tet_pred_parameters_Y1, X, y)

print(tet_pred_parameters_Y1, loss_Y1, "LOSS Y1")
plt.plot(X, Y1(X, tet_pred_parameters_Y1), c="pink", label="Y1")


# tet_parameters = np.ones(15) * 7
# tet_pred_parameters_Y2 = sc.optimize.minimize(loss_f, tet_parameters, args=(X, y), method="Powell").x
# loss_Y2 = loss_f(tet_pred_parameters_Y2, X, y)

# print(tet_pred_parameters_Y2, loss_Y2, "LOSS Y2")
# plt.plot(X, Y2(X, tet_pred_parameters_Y2), c="black", label="Y2")



plt.plot(X, np.array(df["y"]) * 1e6, label="Original data")
plt.legend()
plt.show()

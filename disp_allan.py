import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def allan_variance(data):
    f = open("dispersia.txt", 'a')
    tau = 2880
    sigmas = np.array(())
    for i in range(0,len(data)-tau):
        sigmas = np.hstack((sigmas,(data[i+tau] - data[i])/(tau*30)))
    f.write(str(1/(2*(len(sigmas)-1)) * np.sum(np.diff(sigmas)**2))+" "+str(tau)+'\n')
    print("Дисперсия Аллана: ",1/(2*(len(sigmas)-1)) * np.sum(np.diff(sigmas)**2))
    return sigmas

df = pd.read_csv('R01_365.csv', parse_dates=['ds'])
pd.options.display.float_format = '{:.16f}'.format
df['y'] = df['y'].apply(lambda x: float(x))
df=df[:500000]

sgs=allan_variance(df['y'].to_numpy())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(df['ds'][:500000], df['y'][:500000])
ax1.set_ylabel('Значение')
ax1.set_xlabel('Дата')
ax1.set_title('Данные')

ax2.plot(df['ds'][:len(sgs)], sgs, color='blue')

plt.show()

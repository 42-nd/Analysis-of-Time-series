import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import allantools

def allan_variance(df):
    #diff = np.diff(df.y)
    data = df.to_numpy()
    tau = 2
    sigmas = np.array(())
    for i in range(0,len(data)-tau):
        sigmas = np.hstack((sigmas,(data[i+tau] - data[i])/tau))
    return  sigmas
    #1/(2*(len(sigmas)-1)) * np.sum(np.diff(sigmas)**2)
def allan_variance_CHATGPT(data, tau):
    n = len(data)
    m = int(np.floor(n / tau))
    y = np.reshape(data[:m*tau], (m, tau))
    y_mean = np.mean(y, axis=1)
    z = np.diff(y_mean)
    return z
    #np.sqrt(0.5*np.mean(z**2))


df = pd.read_csv('R01_365.csv', parse_dates=['ds'])
pd.options.display.float_format = '{:.16f}'.format
df['y'] = df['y'].apply(lambda x: float(x))
df=df[:10000]

plt.plot(df['ds'][:len(allan_variance(df['y']))], allan_variance(df['y']),color ='blue')
plt.plot(df['ds'][:len(allan_variance_CHATGPT(df['y'].to_numpy(), 1))], allan_variance_CHATGPT(df['y'].to_numpy(), 1),color ='green')

plt.title('График данных')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.grid(True)
plt.show()


#sigmas = allantools.adev(df['y'].to_numpy(), rate=0.00001, data_type="freq",taus="all")[1]
#plt.plot(df['ds'][:len(sigmas)], sigmas,color ='orange')

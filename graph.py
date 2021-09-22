import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('./data/IRS_10.csv')
dout = pd.read_csv('output.csv')
out = dout.to_numpy()


sample = out[100:, :]


pnl = sample[:, -1]
lisse = []
for i in range(2, len(sample)-2):
    lisse.append(0.125 * pnl[i - 2] + 0.125 * pnl[i - 1] + 0.5 * pnl[i] + 0.125 * pnl[i + 1] +0.125 * pnl[i + 2])


fig, axs = plt.subplots(2)
fig.suptitle('PnL')
axs[0].plot(pnl)
axs[1].plot(lisse)
for ax in axs.flat:
    ax.set(xlabel='steps in days', ylabel= 'PnL')
plt.show()
fig, axs = plt.subplots(5)
fig.suptitle('PnL')
axs[0].plot(sample[:,1]+sample[:,2])
axs[1].plot(sample[:,3]+sample[:,4]+sample[:,5])
axs[2].plot(sample[:,6]+sample[:,7])
axs[3].plot(sample[:,8]+sample[:,9]+sample[:,10])
axs[4].plot(np.sum(sample[:, 1:11], axis=1))
fig.suptitle('Sum of the deltas in every Buckets')

plt.show()
total_limit = 25000
n, m =np.shape(sample)
xx = np.linspace(0, n-1, n)
plt.plot(xx, np.sum(sample[:, 1:11], axis=1), 'b', label = "Sum of all the deltas")
plt.plot(xx , total_limit * np.ones(n), 'r', label = "The highest imposed limit of the sum")
plt.plot(xx , -total_limit * np.ones(n), 'r', label = "The lowest imposed limit of the sum" )
plt.legend(loc="upper left")
plt.show()

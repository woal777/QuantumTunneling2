#%%
from tunneling.main2 import Current, plt, np
import time

t = time.time()
x = np.append(np.linspace(-.5, -.1, 12, endpoint=False), -np.logspace(-1, -4, 8))
x = np.append(x, np.logspace(-4, -1, 8, endpoint=False))
x = np.append(x, np.linspace(.1, .5, 12))
arr = dict()
arr['dn'] = np.genfromtxt('dn', delimiter='\t')
arr['up'] = np.genfromtxt('up', delimiter='\t')
cbEn = [[0.1, 1.], [0.4, 1.5]]
cbEn = np.array(cbEn)
for i, j in arr.items():
    mim = Current(j, cbEn)
    mim.num_estep = 100
    mim.dx = 2e-10
    y = [mim.current(r) for r in x]
    y = np.abs(y)
    output = np.array((x, y))
    output = output.transpose()
    np.savetxt(f'{i}.dat', output, delimiter='\t')
    plt.semilogy(x, y, label=i)

plt.show()

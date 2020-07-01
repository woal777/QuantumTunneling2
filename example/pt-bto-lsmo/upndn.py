from tunneling.main2 import MIM, plt, np
import time

t = time.time()
x = np.append(np.linspace(-.5, -.1, 12, endpoint=False), -np.logspace(-1, -4, 8))
x = np.append(x, np.logspace(-4, -1, 8, endpoint=False))
x = np.append(x, np.linspace(.1, .5, 12))
arr = dict()
arr['dn'] = np.genfromtxt('dn', delimiter='\t')[:, 1]
arr['up'] = np.genfromtxt('up', delimiter='\t')[:, 1]
for i, j in arr.items():
    j += 1
    mim = MIM(j, False, False)
    mim.dx = 18e-10
    y = mim.iv(x)
    y = np.abs(y)
    output = np.array((x, y))
    output = output.transpose()
    np.savetxt(f'{i}.dat', output, delimiter='\t')
    plt.semilogy(x, y, label=i)

plt.show()

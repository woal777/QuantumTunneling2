import shutil
import sys

from matplotlib.axes import Axes
import os

from tunneling.main import Current, plt, np

#barrier = [0, *np.linspace(3.4, .3, 5), *np.linspace(.3, .3, 12), 0]
#barrier = [0, *np.linspace(3.4, .4, 8)[:-1], *np.linspace(.4, .4, 5), 0]
#barrier = [0, *np.linspace(2.5, .0, 12), 0]
#barrier = [0, *np.linspace(1.1, 1.1, 12), 0]
#barrier = [0, *np.linspace(3.4, .0, 7)[:-1], *np.linspace(.0, 3.4, 6), 0]
#barrier = [0, *np.linspace(2.8, -.4, 12), 0]
#barrier = [0, *np.linspace(3.4, .2, 12)[:-1], *np.linspace(.2, .2, 6), 0]
a = 4
b = 2
barrier = [0, *[1] * b, *[0] * a, *[1] * b, 0]

print(len(barrier))
# barrier[5:16] = [0] * 11
c = Current(barrier)
x = np.linspace(-1, 2, 3221)
c.m = 0.03
c.dx = 16e-10
c.temperature = 30
fig: plt.Figure = plt.figure()
ax1: Axes = fig.add_subplot(221)
ax2: Axes = fig.add_subplot(222)
ax3: Axes = fig.add_subplot(223)
ax4: Axes = fig.add_subplot(224)
x2 = []
x2.extend(np.linspace(-1, -.2, 22))
x2.extend(-np.logspace(-1, -3, 6))
x2.extend(np.logspace(-3, -1, 6))
x2.extend(np.linspace(.2, 1, 22))
print(x2)
y2 = []
rep = 5
prob = np.zeros((len(x2) + 1, len(x)))
prob[0, :] = x
for n, i in enumerate(x2):
    y2.append(abs(c.current(i)))
    c.gen_pot(i)
    y = np.array([c.transmission(r + 0j)[0] for r in x])
    prob[n + 1, :] = y
    if n % rep is 0:
        print(i)
        ax1.plot(x, y, label=f'{i:7.3f}')
        ax2.plot(c.v)

if len(y2) == len(x2):
    ax3.semilogy(x2, y2)
# ax1.set_ylim((0, 1))
ax4.text(0.2, 0.5,
         f'c.m={c.m:.4f}\nc.dx={c.dx:.2e}'
         , fontsize=24)
ax1.legend()
ax4.legend()
plt.show()

os.chdir('/home/jinho93')
shutil.copy(sys.argv[0], '/home/jinho93')
if len(y2) == len(x2):
    x2.reverse()
    np.savetxt('current.dat', np.array([x2, y2]).transpose())
np.savetxt('prob.dat', prob.transpose())

from tunneling.main import Current, np, plt
import time

init = time.time()
fig = plt.figure()
x = np.append(np.linspace(-1.5, -.5, 12), -np.logspace(-0.5, -3, 6))
x = np.append(x, np.logspace(-3, -0.5, 6))
x = np.append(x, np.linspace(.5, 1.5, 12))

barrier = [0, *np.linspace(3, 1, 20), 0]
barrier[10] = 0
c = Current(np.array(barrier))
c.m = .05
c.dx *= 3
y = [c.current(r) for r in x]
plt.semilogy(x, np.abs(y), label='right-dead')
plt.legend()
print(time.time() - init)
np.savetxt('down.dat', np.array((x, y)).transpose())
plt.show()

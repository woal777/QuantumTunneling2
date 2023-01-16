from tunneling.main import Current, np, plt

fig = plt.figure()
x = np.append(np.linspace(-1.5, -.2, 8), -np.logspace(-1, -3, 8))
x = np.append(x, np.logspace(-3, -1, 8))
x = np.append(x, np.linspace(.2, 1.5, 8))

c = Current(np.array([0, *np.linspace(1, 1., 20), 0]))
y = [abs(c.current(r)) for r in x]
plt.semilogy(x, y, label='rec')
np.savetxt('rec.dat', np.array([x, y]).transpose())
c = Current(np.array([0, *np.linspace(2, .0, 20), 0]))
y = [abs(c.current(r)) for r in x]
plt.semilogy(x, y, label='tri-left')
c = Current(np.array([0, *np.linspace(.0, 2, 20), 0]))
y = [abs(c.current(r)) for r in x]
plt.semilogy(x, y, label='tri-right')

c = Current(np.array([0, *np.linspace(1.9, 1.9, 10), *np.linspace(.1, .1, 10), 0]))
y = [abs(c.current(r)) for r in x]
plt.semilogy(x, y, label='right-dead')
plt.legend()
plt.show()

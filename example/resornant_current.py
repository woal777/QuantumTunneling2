from tunneling.main import Current, plt, np

pot = np.linspace(1.6, .0, 20)
pot[9:12] = 0
c = Current([0, *pot, 0])
c.m = 0.2
c.dx = 1 * 2e-10
fig = plt.figure()
x = np.append(np.linspace(-1.5, -.2, 12), -np.logspace(-1, -3, 8))
x = np.append(x, np.logspace(-3, -1, 8))
x = np.append(x, np.linspace(.2, 1.5, 12))
y = [abs(c.current(r)) for r in x]
plt.semilogy(x, y)
plt.show()

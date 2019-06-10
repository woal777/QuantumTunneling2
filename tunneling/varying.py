from tunneling.main import Current, np, plt

fig = plt.figure()
x = np.append(np.linspace(-1.5, -.5, 12), -np.logspace(-0.5, -3, 6))
x = np.append(x, np.logspace(-3, -0.5, 6))
x = np.append(x, np.linspace(.5, 1.5, 12))

c = Current(np.array([0, *np.linspace(.7, .1, 10), *np.linspace(.1, .1, 10), 0]))
y = [abs(c.current(r)) for r in x]
plt.semilogy(x, y, label='right-dead')
plt.legend()
plt.show()

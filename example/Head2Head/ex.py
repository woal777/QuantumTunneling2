from tunneling.main import Current, plt, np

x = np.append(np.linspace(-1.5, -.2, 12), -np.logspace(-1, -3, 4))
x = np.append(x, np.logspace(-3, -1, 4))
x = np.append(x, np.linspace(.2, 1.5, 12))
fig = plt.figure(figsize=(8.5 / 2, 10 / 2))
for i, j in enumerate([[0, *np.linspace(2.5, 0, 10), *np.linspace(0, 2.5, 10), 0],
                       [0, *np.linspace(0, .5, 5), *np.linspace(2.5, 0, 10),
                        *np.linspace(0, 2.5, 10), *np.linspace(.5, 0, 5), 0],
                       ]):
    c = Current(j)
    c.dx = 2.4e-10
    c.m = 0.10
    y = [abs(c.current(r)) for r in x]
    plt.semilogy(x, y, label=i)
plt.legend()
# plt.ylim((.1, 2e+9))
plt.show()

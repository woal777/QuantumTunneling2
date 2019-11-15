from tunneling.main import Current, plt, np

x = np.append(np.linspace(-1.5, -.2, 12), -np.logspace(-1, -3, 8))
x = np.append(x, np.logspace(-3, -1, 8))
x = np.append(x, np.linspace(.2, 1.5, 12))
for i, j in enumerate([
    #[0, *np.linspace(0, 2, 20), 0],
                       #                       [0, *np.linspace(2, 0, 20), 0],
                       #                       [0, *np.linspace(1, 1, 20), 0],
                       [0, *np.linspace(3.4, 0, 10), *np.linspace(0, 3.4, 10), 0],
                       #                       [0, *[2] * 5, *[0] * 10, *[2] * 5, 0],
                       #                       [0, *[1] * 9, *[0] * 2, *[1] * 9, 0]
                       ]):
    c = Current(j)
    c.dx = 2e-10
    c.m = 0.2
    y = [abs(c.current(r)) for r in x]
    plt.semilogy(x, y, label=i)
plt.legend()
plt.show()

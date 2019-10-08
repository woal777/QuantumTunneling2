from tunneling.main import Current, plt, np

c = Current([0, 1e+6, 0])
x = np.linspace(1e-19, 3, 5222)
c.m = .1
for i in [1, 3, 7]:
    c.dx = i * 1e-16
    y = np.array([c.transmission(r + 0j) for r in x])
    plt.plot(x, y)
plt.show()
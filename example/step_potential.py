from tunneling.main import Current, plt, np

c = Current([0, 2, 1])
x = np.linspace(-4, 13, 5222)
c.m = .1
for i in [1, 3, 7]:
    c.dx = i * 6e-10
    y = np.array([c.transmission(r + 0j) for r in x])
    plt.plot(x, y, label=i)
plt.legend()
plt.show()

from tunneling.main import Current, plt, np

c = Current([0, 1,0,1, 0])
x = np.linspace(1e-19, 3, 3222)
for i in [1, 3, 7]:
    c.dx = i * 6e-10
    y = np.array([c.transmission(r + 0j) for r in x])
    plt.plot(x, y)
plt.show()
from tunneling.main import Current, plt, np

c = Current([0, *np.linspace(0, 1, 20), 0])
x = np.linspace(1e-19, 3, 1222)
c.dx = 12e-11
for i, j in enumerate([[0, *np.linspace(0, 2, 20), 0],
                       [0, *np.linspace(2, 0, 20), 0],
                       [0, *np.linspace(1, 1, 20), 0],
                       [0, *np.linspace(2, 0, 10), *np.linspace(0, 2, 10), 0],
                       [0, *[2] * 5, *[0] * 10, *[2] * 5, 0]]):
    c.v_tmp = j
    c.v = j
    y = np.array([c.transmission(r + 0j) for r in x])
    plt.plot(x, y[:,0], label=i)
    print(j)
plt.legend()
plt.show()

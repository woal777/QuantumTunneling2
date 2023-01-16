from tunneling.main import Current, plt, np
import pandas as pd
yy = []

x = np.append(np.linspace(-1.5, -.2, 16), -np.logspace(-1, -3, 8))
x = np.append(x, np.logspace(-3, -1, 8))
x = np.append(x, np.linspace(.2, 1.5, 16))
fig = plt.figure(figsize=(8.5 / 2, 10 / 2))
for i, j in enumerate([
                       [0, *np.linspace(1.67, 1.2, 2), *np.linspace(1., .0, 15), 0],
                       [0, *np.linspace(1.67, .4, 2), *np.linspace(.28, .0, 6), 0],
                       ]):
    c = Current(j)
    c.dx = 4.08e-10
    c.m = 0.2
    y = [abs(c.current(r)) for r in x]
    yy.append(y)
    plt.semilogy(x, y, label=i)
plt.legend()
plt.ylim((.1, 2e+9))
yy = np.array(yy)
df = pd.DataFrame(yy.T)
df.to_excel('data_phi.xlsx', index=False)
plt.show()

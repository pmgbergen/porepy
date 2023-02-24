import numpy as np
import matplotlib.pyplot as plt


kappa = 1e-5

def Theta(t):
    return t/ (t + 1)


x = np.linspace(0, 1, 100)


y = Theta(x / (kappa + 1)) * kappa

plt.plot(x, y)

plt.show()

print("done")
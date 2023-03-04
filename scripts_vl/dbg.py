import matplotlib.pyplot as plt
import numpy as np

kappa = 1e-5


def Theta(t):
    return t / (t + 1)


x = np.linspace(0, 1, 100)


y = Theta(x / (kappa + 1)) * kappa

plt.plot(x, y)

plt.show()

try:
    raise RuntimeError("err")
except Exception as err:
    print(str(err))
else:
    print("else")
finally:
    print("finally")


print("Done")

import matplotlib.pylab as plt
import numpy as np

z = [0.99, 0.01]
nc = len(z)

K = [45.48858933984663, 2233.0105094457194]


def _rr_pole(i, y):
    return 1 + y * (K[i] - 1)


def rachford_rice_vle_inversion():
    # numerator
    n = (nc - 1) * sum([(1 - K[i]) * z[i] for i in range(nc)])
    # denominator
    d = sum(
        [
            (K[i] - 1) * sum([(K[j] - 1) for j in range(nc) if j != i]) * z[i]
            for i in range(nc)
        ]
    )
    return n / d


def YPhaseFraction():
    d = (-1 + K[0]) * (-1 + K[1])
    n = z[0] - K[0] * z[0] + z[1] - K[1] * z[1]
    y = n / d
    return y


def rachford_rice_equation(y):

    f = [_rr_pole(i, y) ** (-1) * (K[i] - 1) * z[i] for i in range(nc)]

    return sum(f)


def pot(y):
    F = [-np.log(np.abs(_rr_pole(i, y))) * z[i] for i in range(len(z))]
    return sum(F)


yy = np.linspace(-0.1, 1.1, 5000)

f = pot(yy)
y_sol_ = rachford_rice_vle_inversion()
y_sol = YPhaseFraction()
print("diff", y_sol - y_sol_)

y_max = 1 / (1 - np.min(K))
y_min = 1 / (1 - np.max(K))

YY = rachford_rice_equation(yy)

plt.subplot(1, 2, 1)
plt.plot(yy, f)
plt.axvline(x=y_sol, color="red")
plt.axhline(y=0, color="red", linestyle="--")
plt.subplot(1, 2, 2)
plt.plot(yy, YY)
plt.axvline(x=y_sol, color="red")
plt.axvline(x=y_min, color="green")
plt.axvline(x=y_max, color="green")
plt.axhline(y=0, color="red", linestyle="--")
plt.show()

print("")

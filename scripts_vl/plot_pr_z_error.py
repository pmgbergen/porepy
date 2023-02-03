"""Testing module for the Peng-Robinson EoS class."""
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import porepy as pp


def res_compressibility(Z, A, B):
    """Returns the evaluation of the cubic compressibility polynomial p(A,B)[Z].

    If Z is a root, this should return zero.
    """
    return (
        Z * Z * Z
        + (B - 1) * Z * Z
        + (A - 2 * B - 3 * B * B) * Z
        + (B * B + B * B * B - A * B)
    )

REGION_ENCODING = np.array(
    [
        0,  # 1-real-root region
        1,  # triple-root-region
        2,  # 2-root-region with multiplicity
        3,  # 3-root-region
    ]
)

refinement_a = 100
refinement_b = 100
over_shoot = 1.5

# exclude critical point
a = np.linspace(0, over_shoot * pp.composite.A_CRIT, refinement_a)
b = np.linspace(0, over_shoot * pp.composite.B_CRIT, refinement_b)

a_eps = a.max() / len(a)
b_eps = b.max() / len(b)

GAS = pp.composite.PR_EoS(True)
LIQ = pp.composite.PR_EoS(False)

A, B = np.meshgrid(a, b)
n, m = A.shape
RES_LIQ = np.zeros((n,m))
RES_GAS = np.zeros((n,m))
REG = np.zeros((n,m))
EXTENSION = np.zeros((n, m))

print("Calculating data ...", flush=True)
for i in range(n):
    for j in range(m):

        a_ij = A[i, j]
        b_ij = B[i, j]

        if abs(a_ij - pp.composite.A_CRIT) < a_eps and abs(b_ij - pp.composite.B_CRIT) < b_eps:
            print("Skipping critical point.")
            continue
        ad_a = pp.ad.Ad_array(np.array([a_ij]), sps.lil_matrix((1, 1)))
        ad_b = pp.ad.Ad_array(np.array([b_ij]), sps.lil_matrix((1, 1)))

        Z_G, region = GAS._Z(ad_a, ad_b)
        Z_L, _ = LIQ._Z(ad_a, ad_b)

        res_liq_ij = abs(res_compressibility(Z_L.val[0], a_ij, b_ij))
        res_gas_ij = abs(abs(res_compressibility(Z_G.val[0], a_ij, b_ij)))

        # if res_liq_ij > 1e-12:
        #     res_liq_ij = 0.

        RES_LIQ[i,j] = res_liq_ij
        RES_GAS[i,j] = res_gas_ij
        reg = REGION_ENCODING[region][0]
        REG[i,j] = reg

        liq_is_extended = LIQ.is_extended[0]
        gas_is_extended = GAS.is_extended[0]

        if reg == 0:
            if liq_is_extended and not gas_is_extended:
                EXTENSION[i,j] = 1
                RES_LIQ[i,j] = 0
            if gas_is_extended and not liq_is_extended:
                EXTENSION[i,j] = 2
                RES_GAS[i,j] = 0.
            if gas_is_extended and liq_is_extended:
                EXTENSION[i,j] = 3
                RES_LIQ[i,j] = 0.
                RES_GAS[i,j] = 0.

        else:
            EXTENSION[i,j] = 0

    if i % 5 == 0:
        print(f"... {np.floor(i/n*100)}% of rows calculated...")


# scaling
MAX_ERR_LIQ = RES_LIQ.max()
MAX_ERR_GAS = RES_GAS.max()
RES_LIQ = RES_LIQ / MAX_ERR_LIQ
RES_GAS = RES_GAS / MAX_ERR_GAS
is_zero_l = RES_LIQ == 0
is_zero_g = RES_GAS == 0
RES_LIQ[np.logical_not(is_zero_l)] = np.log(RES_LIQ[np.logical_not(is_zero_l)])
RES_GAS[np.logical_not(is_zero_g)] = np.log(RES_GAS[np.logical_not(is_zero_g)])
RES_LIQ[is_zero_l] = RES_LIQ.min()
RES_GAS[is_zero_g] = RES_GAS.min()


## PLOTTING
print("Plotting...", flush=True)
gs = gridspec.GridSpec(2,2)

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080/1920 * figwidth))

# liquid residual
ax_liq = plt.subplot(gs[0, 0])
img_liq = ax_liq.pcolormesh(A, B, RES_LIQ, cmap='Greys', vmin=RES_LIQ.min(), vmax=RES_LIQ.max())
ax_liq.set_title('Liquid root residuals')
ax_liq.set_xlabel('A')
ax_liq.set_ylabel('B')

divider = make_axes_locatable(ax_liq)
cax = divider.append_axes('right', size='5%', pad=0.1)
cb_liq = fig.colorbar(img_liq, cax=cax, orientation='vertical')
cb_liq.set_label("Log(Rel(abs(residual)))\nMax error: " + '{:.0e}'.format(float(MAX_ERR_LIQ)))

ax_liq.plot([0, pp.composite.A_CRIT], [0, pp.composite.B_CRIT], color='red', linewidth=1)
ax_liq.plot(pp.composite.A_CRIT, pp.composite.B_CRIT, "or", markersize=3)
ax_liq.text(pp.composite.A_CRIT, pp.composite.B_CRIT, "(Ac,Bc)", color='red', fontsize='small',
horizontalalignment='center',
verticalalignment='bottom')

# Gas residual
ax_gas = plt.subplot(gs[0, 1])
img_gas = ax_gas.pcolormesh(A, B, RES_GAS, cmap='Greys', vmin=RES_GAS.min(), vmax=RES_GAS.max())
ax_gas.set_title('Gas root residuals')
ax_gas.set_xlabel('A')
ax_gas.set_ylabel('B')

divider = make_axes_locatable(ax_gas)
cax = divider.append_axes('right', size='5%', pad=0.1)
cb_gas = fig.colorbar(img_gas, cax=cax, orientation='vertical')
cb_gas.set_label("Log(Rel(abs(residual)))\nMax error: " + '{:.0e}'.format(float(MAX_ERR_GAS)))

ax_gas.plot([0, pp.composite.A_CRIT], [0, pp.composite.B_CRIT], color='red', linewidth=1)
ax_gas.plot(pp.composite.A_CRIT, pp.composite.B_CRIT, "or", markersize=3)
ax_gas.text(pp.composite.A_CRIT, pp.composite.B_CRIT, "(Ac,Bc)", color='red', fontsize='small',
horizontalalignment='center',
verticalalignment='bottom')

# root regions
ax_rr = plt.subplot(gs[1, 0])
cmap = mpl.colors.ListedColormap(['yellow', 'green', 'blue', 'indigo'])
img_rr = ax_rr.pcolormesh(A, B, REG, cmap=cmap, vmin=0, vmax=3)
ax_rr.set_title('Polynomial root cases')
ax_rr.set_xlabel('A')
ax_rr.set_ylabel('B')

divider = make_axes_locatable(ax_rr)
cax = divider.append_axes('right', size='5%', pad=0.1)
cb_rr = fig.colorbar(img_rr, cax=cax, orientation='vertical')
cb_rr.set_ticks(REGION_ENCODING)
cb_rr.set_ticklabels(['1-real-root', 'triple-root', '2-real-root', '3-real-root'])

ax_rr.plot([0, pp.composite.A_CRIT], [0, pp.composite.B_CRIT], color='red', linewidth=1)
ax_rr.plot(pp.composite.A_CRIT, pp.composite.B_CRIT, "or", markersize=3)
ax_rr.text(pp.composite.A_CRIT, pp.composite.B_CRIT, "(Ac,Bc)", color='red', fontsize='small',
horizontalalignment='center',
verticalalignment='bottom')

# extensions
ax_ext = plt.subplot(gs[1, 1])
cmap = mpl.colors.ListedColormap(['white', 'blue', 'green', 'red'])
img_ext = ax_ext.pcolormesh(A, B, EXTENSION, cmap=cmap, vmin=0, vmax=3)
ax_ext.set_title('Root extensions')
ax_ext.set_xlabel('A')
ax_ext.set_ylabel('B')

divider = make_axes_locatable(ax_ext)
cax = divider.append_axes('right', size='5%', pad=0.1)
cb_ext = fig.colorbar(img_ext, cax=cax, orientation='vertical')
cb_ext.set_ticks(REGION_ENCODING)
cb_ext.set_ticklabels(['no extension', 'liq extended', 'gas extended', 'both extended'])

ax_ext.plot([0, pp.composite.A_CRIT], [0, pp.composite.B_CRIT], color='red', linewidth=1)
ax_ext.plot(pp.composite.A_CRIT, pp.composite.B_CRIT, "or", markersize=3)
ax_ext.text(pp.composite.A_CRIT, pp.composite.B_CRIT, "(Ac,Bc)", color='red', fontsize='small',
horizontalalignment='center',
verticalalignment='bottom')

fig.tight_layout()
fig.savefig(
    '/mnt/c/Users/vl-work/Desktop/Z_calcs.png',
    format='png',
    dpi=500
)
fig.show()


print("Done.")


# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')
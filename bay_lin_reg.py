import numpy as np
from numpy.random import normal, uniform
from scipy.stats import multivariate_normal as mv_norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",  # Computer Modern (como LaTeX)
    "axes.linewidth": 1,
    "xtick.top": False,
    "ytick.right": False,
    "ytick.left": False,
    "xtick.bottom": False,
})

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parámetros reales ─────────────────────────────────────────────────────────
a_0         = -0.3
a_1         =  0.5
noise_sigma =  0.2
beta        = 1 / noise_sigma**2

# ── Datos ─────────────────────────────────────────────────────────────────────
np.random.seed(20)
x_real = uniform(-1, 1, 1000)
t_real = a_0 + a_1 * x_real + normal(0, noise_sigma, 1000)

# ── Clase LinearBayes ─────────────────────────────────────────────────────────
class LinearBayes:
    def __init__(self, m0, S0, beta):
        self.v_m0 = m0.reshape(-1, 1)
        self.m_S0 = S0
        self.beta = beta
        self.v_mN = self.v_m0.copy()
        self.m_SN = S0.copy()
        self.posterior = mv_norm(mean=m0, cov=S0)

    def get_phi(self, x):
        phi = np.ones((len(x), 2))
        phi[:, 1] = x
        return phi

    def set_posterior(self, x, t):
        phi  = self.get_phi(x)
        v_t  = t.reshape(-1, 1)
        SN   = np.linalg.inv(np.linalg.inv(self.m_S0) + self.beta * phi.T @ phi)
        mN   = SN @ (np.linalg.inv(self.m_S0) @ self.v_m0 + self.beta * phi.T @ v_t)
        self.m_SN    = SN
        self.v_mN    = mN
        self.posterior = mv_norm(mean=mN.flatten(), cov=SN)

    def prediction_limit(self, x, stdevs):
        phi = self.get_phi(x).T.reshape(2, 1, len(x))
        out = []
        for i in range(len(x)):
            xi       = phi[:, :, i]
            sig2     = 1 / self.beta + xi.T @ self.m_SN @ xi
            mean_x   = self.v_mN.T @ xi
            out.append((mean_x + stdevs * np.sqrt(sig2)).flatten())
        return np.concatenate(out)

    def generate_data(self, x):
        phi = self.get_phi(x).T.reshape(2, 1, len(x))
        out = []
        for i in range(len(x)):
            xi     = phi[:, :, i]
            sig2   = 1 / self.beta + xi.T @ self.m_SN @ xi
            mean_x = self.v_mN.T @ xi
            out.append(normal(mean_x.flatten(), np.sqrt(sig2)))
        return np.array(out)

# ── Prior ─────────────────────────────────────────────────────────────────────
alpha   = 2.0
v_m0    = np.array([0., 0.])
m_S0    = np.eye(2) / alpha
linbayes = LinearBayes(v_m0, m_S0, beta)

x_grid  = np.linspace(-1, 1, 100)
ww, vv  = np.mgrid[-1:1:.01, -1:1:.01]
pos     = np.empty(ww.shape + (2,))
pos[:, :, 0] = ww
pos[:, :, 1] = vv

def true_line(x): return a_0 + a_1 * x

def save_scatter(x, t, fname, samples=None, stdevs=None):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if len(x) > 0:
        ax.scatter(x, t, alpha=0.5, s=40, c="#8fb2ff")
    ax.plot([-1, 1], [true_line(-1), true_line(1)], color="#D85A30", lw=2)
    if samples:
        weights = linbayes.posterior.rvs(samples)
        for w in weights:
            ax.plot([-1, 1], [w[0] + w[1]*(-1), w[0] + w[1]*1], c="#aaaaaa", lw=1, alpha=0.7)
    if stdevs:
        y_up = linbayes.prediction_limit(x_grid, stdevs)
        y_lo = linbayes.prediction_limit(x_grid, -stdevs)
        ax.plot(x_grid, y_up, c='#1a5fa8', lw=2)
        ax.plot(x_grid, y_lo, c='#1a5fa8', lw=2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

import pandas as pd

def save_scatter_csv(x, t, fname, samples=None, stdevs=None):
    data = {'x_scatter': x, 't_scatter': t}
    df = pd.DataFrame(data)
    df.to_csv(f"{OUT_DIR}/{fname}_scatter.csv", index=False)

    # Línea verdadera
    df_line = pd.DataFrame({'x': [-1, 1], 'y': [true_line(-1), true_line(1)]})
    df_line.to_csv(f"{OUT_DIR}/{fname}_line.csv", index=False)

    # Muestras de la posterior
    if samples:
        weights = linbayes.posterior.rvs(samples)
        for i, w in enumerate(weights):
            df_samp = pd.DataFrame({'x': [-1, 1], 'y': [w[0] + w[1]*(-1), w[0] + w[1]*1]})
            df_samp.to_csv(f"{OUT_DIR}/{fname}_samp{i}.csv", index=False)

    # Banda de predicción
    if stdevs:
        y_up = linbayes.prediction_limit(x_grid, stdevs)
        y_lo = linbayes.prediction_limit(x_grid, -stdevs)
        df_pred = pd.DataFrame({'x': x_grid, 'y_up': y_up, 'y_lo': y_lo})
        df_pred.to_csv(f"{OUT_DIR}/{fname}_pred.csv", index=False)

    print(f"CSV guardado: {fname}")

def save_contour(fname):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    ax.contourf(ww, vv, linbayes.posterior.pdf(pos), 20, cmap='jet')
    ax.scatter([a_0], [a_1], marker='+', c='black', s=500, linewidths=3)
    ax.set_xlabel('$θ_0$', fontsize=28, labelpad=18)
    ax.set_ylabel('$θ_1$', fontsize=28, labelpad=18)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.tick_params(bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

save_contour("pos_0.png")

# N=1
save_scatter_csv(x_real[:1], t_real[:1], "plot_1")
linbayes.set_posterior(x_real[:1], t_real[:1])
save_contour("pos_1.png")
save_scatter_csv(x_real[:1], t_real[:1], "samp_1", samples=5)
save_scatter_csv(x_real[:1], t_real[:1], "pred_1", stdevs=1)

# N=2
linbayes.set_posterior(x_real[:2], t_real[:2])
save_contour("pos_2.png")
save_scatter_csv(x_real[:2], t_real[:2], "pred_2", samples=5, stdevs=1)

# N=10
linbayes.set_posterior(x_real[:10], t_real[:10])
save_contour("pos_10.png")
save_scatter_csv(x_real[:10], t_real[:10], "pred_10", samples=5, stdevs=1)

# N=1000
linbayes.set_posterior(x_real[:1000], t_real[:1000])
save_contour("pos_1000.png")
save_scatter_csv(x_real[:1000], t_real[:1000], "pred_1000", samples=5, stdevs=1)


print(f"Guardadas figuras en {OUT_DIR}/")
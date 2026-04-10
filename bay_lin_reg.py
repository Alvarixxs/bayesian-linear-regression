import numpy as np
from numpy.random import normal, uniform
from scipy.stats import multivariate_normal as mv_norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams['axes.labelsize']  = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

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
        ax.scatter(x, t, alpha=0.4, s=10)
    ax.plot([-1, 1], [true_line(-1), true_line(1)], color="#D85A30", lw=2)
    if samples:
        weights = linbayes.posterior.rvs(samples)
        for w in weights:
            ax.plot([-1, 1], [w[0] + w[1]*(-1), w[0] + w[1]*1], 'k', c="#aaaaaa", lw=1, alpha=0.7)
    if stdevs:
        y_up = linbayes.prediction_limit(x_grid, stdevs)
        y_lo = linbayes.prediction_limit(x_grid, -stdevs)
        ax.plot(x_grid, y_up, c='#1a5fa8', lw=2)
        ax.plot(x_grid, y_lo, c='#1a5fa8', lw=2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def save_contour(fname):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.contourf(ww, vv, linbayes.posterior.pdf(pos), 20, cmap='jet')
    ax.scatter([a_0], [a_1], marker='+', c='black', s=500, linewidths=3)
    ax.set_xlabel('$θ_0$')
    ax.set_ylabel('$θ_1$')
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Figura 2: prior ───────────────────────────────────────────────────────────
save_contour("pos_0.png")

# ── N=1 ───────────────────────────────────────────────────────────────────────
N = 1
save_scatter(x_real[:N], t_real[:N], "plot_1.png")
linbayes.set_posterior(x_real[:N], t_real[:N])
save_contour("pos_1.png")
save_scatter(x_real[:N], t_real[:N], "samp_1.png", samples=5)
save_scatter(x_real[:N], t_real[:N], "pred_1.png", stdevs=1)

# ── N=2 ───────────────────────────────────────────────────────────────────────
N = 2
save_scatter(x_real[:N], t_real[:N], "plot_2.png")
linbayes.set_posterior(x_real[:N], t_real[:N])
save_contour("pos_2.png")
save_scatter(x_real[:N], t_real[:N], "pred_2.png", samples=5, stdevs=1)

# ── N=10 ──────────────────────────────────────────────────────────────────────
N = 10
save_scatter(x_real[:N], t_real[:N], "plot_10.png")
linbayes.set_posterior(x_real[:N], t_real[:N])
save_contour("pos_10.png")
save_scatter(x_real[:N], t_real[:N], "pred_10.png", samples=5, stdevs=1)

# ── N=1000 ────────────────────────────────────────────────────────────────────
N = 1000
save_scatter(x_real[:N], t_real[:N], "plot_1000.png")
linbayes.set_posterior(x_real[:N], t_real[:N])
save_contour("pos_1000.png")
save_scatter(x_real[:N], t_real[:N], "pred_1000.png", samples=5, stdevs=1)

print(f"Guardadas figuras en {OUT_DIR}/")
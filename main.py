import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from draw_sim import create_gif
import pandas as pd
import pickle as pkl
from tqdm import tqdm

mpl.use('TkAgg')


class IsingSim:
    def __init__(self, grid, j, h, beta, k=1):
        self.grid = grid  # number of spins
        self.j = j  # interaction strength
        self.h = h  # magnetic field
        self.beta = beta  # inverse temperature
        self.k = k  # Boltzmann constant

    def gen_random_grid(self):
        return np.random.choice([-1, 1], size=self.grid)

    def hamiltonian(self, x):
        m, n = self.grid
        interaction = 0
        for i in range(m):
            for j in range(n):
                interaction += x[i, j] * (x[i, (j + 1) % n] + x[i, j - 1] + x[(i + 1) % m, j] + x[i - 1, j])
        field = np.sum(x)
        return -self.j * interaction / 2 - self.h * field

    def magnetization(self, x):
        return np.sum(x)/(x.shape[0] * x.shape[1])

    def metropolis_step(self, x, trials=1):
        for i in range(trials):
            m = np.random.randint(0, self.grid[0])
            n = np.random.randint(0, self.grid[1])
            # x_new = x.copy()
            # x_new[m, n] *= -1
            # delta_h = self.hamiltonian(x_new) - self.hamiltonian(x)
            delta_h = 2 * self.j * (
                        x[m, (n + 1) % x.shape[1]] + x[m, n - 1] + x[(m + 1) % x.shape[0], n] + x[m - 1, n]) * x[
                          m, n] + 2 * self.h * x[m, n]
            if delta_h < 0:
                x[m, n] *= -1
            else:
                if np.random.rand() < np.exp(-beta * delta_h):
                    x[m, n] *= -1
        return x, self.hamiltonian(x)


if __name__ == "__main__":
    j0, h0, beta0 = 1, 0, .1  # 15, 0, 0.06
    grid = (20, 20)
    runs = 10_000
    # beta_range = np.arange(0.25, .55, .005)
    beta_range = np.linspace(0.3, .6, 100)

    specific_heat = np.empty_like(beta_range)
    avg_magnetization = np.zeros_like(beta_range)
    avg_energy = np.zeros_like(beta_range)

    magnetization_err = np.zeros_like(beta_range)
    energy_err = np.zeros_like(beta_range)

    sim = IsingSim(grid, j0, h0, beta0)
    x0 = sim.gen_random_grid()

    i = 0
    for beta in tqdm(beta_range):
        sim = IsingSim(grid, j0, h0, beta)
        x = sim.gen_random_grid()

        energy = np.empty(runs)
        magnetization = np.empty(runs)

        # burn-in
        x, energy[0] = sim.metropolis_step(x, trials=20000)

        for j in range(runs):
            x, energy[j] = sim.metropolis_step(x, trials=1000)
            magnetization[j] = np.abs(sim.magnetization(x))

        specific_heat[i] = np.var(energy) * (sim.beta ** 2)
        avg_magnetization[i] = np.mean(magnetization)
        avg_energy[i] = np.mean(energy)

        magnetization_err[i] = np.std(magnetization)/np.sqrt(runs)
        energy_err[i] = np.std(energy)/np.sqrt(runs)
        i += 1

    plt.style.use('seaborn-dark-palette')
    fig, axs = plt.subplots(3, 1)
    axs[0].errorbar(beta_range, avg_energy, yerr=energy_err, capsize=3, ecolor="k")
    axs[0].title.set_text('Energy')

    axs[1].errorbar(beta_range, avg_magnetization, yerr=magnetization_err, capsize=3, ecolor="k")
    axs[1].title.set_text('Magnetization')

    axs[2].plot(beta_range, specific_heat)
    axs[2].title.set_text('Specific Heat')
    plt.show()

import pandas as pd
import numpy as np
import torch
from scipy.integrate import odeint
from pathlib import Path

dataset_folder = Path("Data")


def get_smoking_data(selected_state=2):
    """
    Returns
    X is a tensor of observations, of shape (batch=1, sequence, paths)
    y is a tensor of labels, of shape (batch=1, sequence=31)
    """

    data = pd.read_csv(dataset_folder / "smoking.csv", header=0)
    data = data[["state", "cigsale"]]
    state_ids = np.unique(data["state"])

    y = list()
    for state in state_ids:
        y.append(np.array(data.loc[data["state"] == state]))

    y = np.array(y)[:, :, 1].astype(float) / 100.0

    t = torch.linspace(0.0, 1.0, y.shape[1])
    xs = torch.tensor(np.delete(y, (selected_state), axis=0)).float().t().unsqueeze(0)
    X = torch.cat([t.unsqueeze(0).unsqueeze(2), xs], dim=2)
    Y = torch.tensor(y[selected_state]).float().unsqueeze(0).unsqueeze(2)  # spain

    return X, Y


def get_smoking_data_numpy(selected_state=2):
    """
    Returns
    X is a tensor of observations, of shape (batch=1, sequence, paths)
    y is a tensor of labels, of shape (batch=1, sequence=31)
    """

    data = pd.read_csv(dataset_folder / "smoking.csv", header=0)
    data = data[["state", "cigsale"]]
    state_ids = np.unique(data["state"])

    y = list()
    for state in state_ids:
        y.append(np.array(data.loc[data["state"] == state]))

    y = np.array(y)[:, :, 1].astype(float) / 100.0

    X = np.delete(y, (selected_state), axis=0)
    Y = y[selected_state]

    return np.transpose(X), Y


def get_emu_data(selected_country=4):
    """
    Returns
    X is a tensor of observations, of shape (batch=1, sequence, paths)
    y is a tensor of labels, of shape (batch=1, sequence=31)
    """

    data = pd.read_csv(dataset_folder / "hope_emu.txt", header=0)
    data = data[["country_ID", "CAB"]]

    y = list()
    for country in np.unique(data["country_ID"]):
        y.append(np.array(data.loc[data["country_ID"] == country]))

    y = np.array(y)[:, :, 1].astype(float) / 10.0

    t = torch.linspace(0.0, 1.0, y.shape[1])
    xs = torch.tensor(np.delete(y, (selected_country), axis=0)).float().t().unsqueeze(0)

    X = torch.cat([t.unsqueeze(0).unsqueeze(2), xs], dim=2)
    Y = torch.tensor(y[selected_country]).float().unsqueeze(0).unsqueeze(2)  # spain

    return X, Y


def get_emu_data_numpy(selected_country=4):
    """
    Returns
    X is a tensor of observations, of shape (batch=1, sequence, paths)
    y is a tensor of labels, of shape (batch=1, sequence=31)
    """

    data = pd.read_csv(dataset_folder / "hope_emu.txt", header=0)
    data = data[["country_ID", "CAB"]]

    y = list()
    for country in np.unique(data["country_ID"]):
        y.append(np.array(data.loc[data["country_ID"] == country]))

    y = np.array(y)[:, :, 1].astype(float) / 10.0

    X = np.delete(y, (selected_country), axis=0)
    Y = y[selected_country]

    return np.transpose(X), Y


def statins(x, t, a, kappa=0.002, kin=0.11, d50=0.5, h=1):
    """Partial derivatives for Lotka-Volterra ODE.
    Args:
    - x (np.array): x[0] = p, x[1] = d, x[2] = y
    - alpha (pxp np.array): matrix of interactions"""
    dxdt = np.zeros(3)
    dxdt[0] = kin - kappa * x[0]
    dxdt[1] = a - h * x[1]
    dxdt[2] = kappa * x[0] - x[1] * kappa * x[2] / (x[1] - d50)

    return dxdt


def simulate_statins(T, a, delta_t=0.1, sd=0.01, burn_in=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.zeros(3)
    x0[0] = 2.1
    x0[1] = 1
    x0[2] = 2.1
    # = np.random.normal(scale=0.01, size=p) + 0.25]
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(statins, x0, t, args=(a,))
    X += np.random.normal(scale=sd, size=(T + burn_in, 3))

    return X[burn_in:]


def lorenz(x, t, F):
    """Partial derivatives for Lorenz-96 ODE."""
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC

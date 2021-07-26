import pandas as pd
import numpy as np
import math
import torch
import torchcde
from torch.nn import functional as F
from IPython.display import clear_output

# We acknowledge the use of tutorials at https://github.com/patrick-kidger/torchcde that served as a skeleton


def plot_trajectories(X, Y, model, title=[1, 2.1]):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.3))
    fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)

    predictions = predict(model, X)
    axs[0].plot(torch.cat([Y, predictions], dim=-1).squeeze())
    axs[0].set_title("Iteration = %i" % title[0] + ",  " + "Loss = %1.3f" % title[1])
    axs[1].plot(Y.squeeze() - predictions.squeeze())
    axs[1].set_title("Treatment Effect")
    axs[2].plot(X[0, :, :])
    axs[2].set_title("Control trajectories")
    plt.show()


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(
            hidden_channels, input_channels * hidden_channels
        )
        self.elu = torch.nn.ELU(inplace=True)
        self.W = torch.nn.Parameter(torch.Tensor(input_channels))
        self.W.data.fill_(1)

    def l2_reg(self):
        """L2 regularization on all parameters"""
        reg = 0.0
        reg += torch.sum(self.linear1.weight ** 2)
        reg += torch.sum(self.linear2.weight ** 2)
        return reg

    def l1_reg(self):
        """L1 regularization on input layer parameters"""
        return torch.sum(torch.abs(self.W))

    # The t argument can be ignored or added specifically if you want your CDE to behave differently at
    # different times.
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = self.elu(z)
        z = self.linear2(z)
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        z = torch.matmul(z, torch.diag(self.W))
        return z


# Next, we need to package CDEFunc up into a model that computes the integral.
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, 1)

    def forward(self, coeffs):
        X = torchcde.CubicSpline(coeffs)

        z0 = self.initial(X.evaluate(0.0))

        # Actually solve the CDE.
        z_hat = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.grid_points)

        pred_y = self.readout(z_hat.squeeze(0)).unsqueeze(0)

        return pred_y

    def predict(self, data, z0):

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data)
        X = torchcde.CubicSpline(coeffs)

        # z0 = X.evaluate(0.)

        # Actually solve the CDE.
        z_hat = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=torch.linspace(X.grid_points[0], X.grid_points[-1], 100),
        )

        return z_hat.detach()


def predict(model, test_X):
    full_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
    return model(full_coeffs).detach()


def train(model, train_X, train_y, test_X, test_y, iterations=1000, l1_reg=0.0001):
    optimizer = torch.optim.Adam(model.parameters())
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
        train_X
    )

    l2_reg = 0.001
    # l1_reg = 0.0001

    for i in range(iterations):
        # horizon = 5
        # index = torch.from_numpy(np.random.choice(np.arange(coeffs.shape[1] - batch_time, dtype=np.int64),
        #                                      1, replace=False))
        pred_y = model(train_coeffs)
        loss = F.mse_loss(pred_y, train_y)
        loss = loss + l1_reg * model.func.l1_reg() + l2_reg * model.func.l2_reg()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            # print('Iteration: {}   Training loss: {}'.format(i, loss.item()))
            plot_trajectories(test_X, test_y, model=model, title=[i, loss])
            clear_output(wait=True)

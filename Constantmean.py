import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D
import torch
import gpytorch

import matplotlib.pyplot as plt
from util import *

plt.style.use("bmh")
plt.rcParams["image.cmap"] = "Blues"

from tqdm.notebook import tqdm

train_x = torch.tensor([[60,500],[60,1000],[60,1500],[60,2000],[60,2500],
                        [100,500],[100,1000],[100,1500],[100,2000],[100,2500],
                        [140,500],[140,1000],[140,1500],[140,2000],[140,2500],
                        [180,500],[180,1000],[180,1500],[180,2000],[180,2500]], dtype=torch.double)

train_y = torch.tensor([0,2,2,1,2,2,5,5,4,4,6,5,5,8,3,3,5,4,3,6], dtype=torch.double)


x1_range = torch.linspace(0, 200, 1000)
x2_range = torch.linspace(0, 3000, 3000)
X1, X2 = torch.meshgrid(x1_range, x2_range, indexing="ij")
xs = torch.vstack((X1.flatten(), X2.flatten())).transpose(-1, -2)

class ConstantMeanGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# declare the GP
noise = 1e-4

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ConstantMeanGPModel(train_x, train_y, likelihood)

# fix the hyperparameters
model.likelihood.noise = noise

# train the hyperparameter (the constant)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

model.train()
likelihood.train()

losses = []
constants = []
for i in tqdm(range(1000)):
    optimizer.zero_grad()

    output = model(train_x)
    loss = -mll(output, train_y)

    loss.backward()

    losses.append(loss.item())
    constants.append(model.mean_module.constant.item())

    optimizer.step()

model.eval()
likelihood.eval()

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ax[0].plot(losses)
ax[0].set_ylabel("negative marginal log likelihood")

ax[1].plot(constants)
ax[1].set_ylabel("constant");

print(model.covar_module.base_kernel.lengthscale)

visualize_2d_contour(model, likelihood, train_x, train_y, X1, X2, xs)
visualize_gp_belief(model, likelihood, train_x, train_y, X1, X2, xs)
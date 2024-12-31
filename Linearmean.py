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

torch.set_default_tensor_type(torch.DoubleTensor)

train_x = torch.tensor([[1/13,1/6],[1/13,3/8],[1/13,7/12],[1/13,19/24],[1/13,1],
                        [5/13,1/6],[5/13,3/8],[5/13,7/12],[5/13,19/24],[5/13,1],
                        [9/13,1/6],[9/13,3/8],[9/13,7/12],[9/13,19/24],[9/13,1],
                        [1,1/6],[1,3/8],[1,7/12],[1,19/24],[1,1]], dtype=torch.double)

train_y = torch.tensor([0,2,2,1,2,2,5,5,4,4,8,6,7,10,6,6,9,9,8,8], dtype=torch.double)


x1_range = torch.linspace(0, 200, 1000)
x2_range = torch.linspace(0, 3000, 3000)
X1, X2 = torch.meshgrid(x1_range, x2_range, indexing="ij")
xs = torch.vstack((X1.flatten(), X2.flatten())).transpose(-1, -2)

class LinearMeanGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(2)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

lenthscale = torch.tensor([6.3, 63.0]) ##############给定长度尺度的先验数值
noise = 1e-4

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = LinearMeanGPModel(train_x, train_y, likelihood)


model.covar_module.base_kernel.lengthscale = lenthscale ###############
model.likelihood.noise = noise

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

model.train()
likelihood.train()

losses = []
weights = []
biases = []
for i in tqdm(range(1000)):
    optimizer.zero_grad()

    output = model(train_x)
    loss = -mll(output, train_y)

    loss.backward()

    losses.append(loss.item())
    weights.append(model.mean_module.weights.detach().cpu().clone().numpy())  
    biases.append(model.mean_module.bias.item())

    optimizer.step()

model.eval()
likelihood.eval()

fig, ax = plt.subplots(1, 3, figsize=(8, 2))

ax[0].plot(losses)
ax[0].set_ylabel("negative marginal log likelihood")


weights = np.array([w for w in weights])  
ax[1].plot(weights[:, 0], label="Weight 1")  
ax[1].plot(weights[:, 1], label="Weight 2") 
ax[1].set_ylabel("weights")
ax[1].legend()

ax[2].plot(biases)
ax[2].set_ylabel("bias")

print(model.covar_module.base_kernel.lengthscale)

plt.tight_layout()

visualize_2d_contour(model, likelihood, train_x, train_y, X1, X2, xs)
visualize_gp_belief(model, likelihood, train_x, train_y, X1, X2, xs)
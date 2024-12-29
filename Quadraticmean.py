import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D
import torch
import gpytorch

import matplotlib.pyplot as plt

plt.style.use("bmh")
plt.rcParams["image.cmap"] = "Blues"

from tqdm.notebook import tqdm

# 将接下来创建的变量类型均为Double
torch.set_default_tensor_type(torch.DoubleTensor)

train_x = torch.tensor([[60,500],[60,1000],[60,1500],[60,2000],[60,2500],
                        [100,500],[100,1000],[100,1500],[100,2000],[100,2500],
                        [140,500],[140,1000],[140,1500],[140,2000],[140,2500],
                        [180,500],[180,1000],[180,1500],[180,2000],[180,2500]], dtype=torch.double)

train_y = torch.tensor([0,2,2,1,2,2,5,5,4,4,6,5,5,8,3,3,5,4,3,6], dtype=torch.double)


x1_range = np.linspace(0, 200, 1000)
x2_range = np.linspace(0, 3000, 3000)
X1, X2 = np.meshgrid(x1_range, x2_range)
X_test = np.vstack([X1.ravel(), X2.ravel()]).T  # 将网格转换为样本点

xs = torch.tensor(X_test)
X1 = torch.tensor(X1)
X2 = torch.tensor(X2)

def visualize_gp_belief(model, likelihood):
    with torch.no_grad():
        predictive_distribution = likelihood(model(xs))
        predictive_mean = predictive_distribution.mean
        predictive_upper, predictive_lower = predictive_distribution.confidence_region()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X1, X2, predictive_mean.reshape(X1.shape), 
        cmap=plt.cm.coolwarm, alpha=0.7
    )
    ax.scatter(
        train_x[:, 0], train_x[:, 1], train_y, 
        color='r', s=50, label='Data Points', marker='o'
    )
    for i in range(len(train_x)):
        ax.text(
            train_x[i, 0], train_x[i, 1], train_y[i], 
            f"({train_x[i, 0]:.2f}, {train_x[i, 1]:.2f})", 
            color='black', fontsize=10
        )

    ax.set_title("GP Regression (Quadratic rmean)", fontsize=16)
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_zlabel('Predicted y', fontsize=12)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=6)

    plt.show()
    plt.close(fig)  # 清理绘图状态


def visualize_2d_contour(model, likelihood):
    with torch.no_grad():
        predictive_distribution = likelihood(model(xs))
        predictive_mean = predictive_distribution.mean

    predictive_mean_reshaped = predictive_mean.reshape(X1.shape).detach().numpy()
    min_value = predictive_mean_reshaped.min()
    min_idx = np.unravel_index(predictive_mean_reshaped.argmin(), predictive_mean_reshaped.shape)
    min_x1 = X1[min_idx]
    min_x2 = X2[min_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X1, X2, predictive_mean_reshaped, cmap="coolwarm", levels=50)
    plt.colorbar(contour, ax=ax, label="Predicted y")

    ax.scatter(train_x[:, 0], train_x[:, 1], color="black", marker="o", label="Training Points")
    for i in range(len(train_x)):
        ax.text(
            train_x[i, 0], train_x[i, 1], 
            f"({train_x[i, 0]:.0f}, {train_x[i, 1]:.0f})", fontsize=8
        )

    ax.scatter(min_x1, min_x2, color="red", marker="*", s=200, label="Min Value")
    ax.text(
        min_x1, min_x2, 
        f"Min: ({min_x1:.2f}, {min_x2:.2f})\nValue: {min_value:.2f}", 
        color="red", fontsize=10, ha="left", va="bottom"
    )

    ax.set_title("2D Contour Plot of GP Regression with Min Value", fontsize=16)
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("X2", fontsize=12)
    ax.legend()
    plt.tight_layout()

    plt.show()
    plt.close(fig)  # 清理绘图状态

class QuadraticMean(gpytorch.means.Mean):
    def __init__(self, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(
            name="second0", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
        )
        self.register_parameter(
            name="first0", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
        )

        self.register_parameter(
            name="second1", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
        )
        self.register_parameter(
            name="first1", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
        )
        self.register_parameter(
            name="mutual", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
        )

        if bias:
            self.register_parameter(
                name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
            )
        else:
            self.bias = None

    def forward(self, x):
        res = (
            x[:, 0].pow(2) * self.second0.squeeze(-1)
            + x[:, 0] * self.first0.squeeze(-1)
            + x[:, 1].pow(2) * self.second1.squeeze(-1)
            + x[:, 1] * self.first1.squeeze(-1)
            + x[:, 0] * x[:, 1] * self.mutual.squeeze(-1)
        )
        if self.bias is not None:
            res = res + self.bias
        return res


class QuadraticMeanGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nu):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = QuadraticMean()
        self.covar_module = gpytorch.kernels.MaternKernel(nu)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# declare the GP
lengthscale = 100
noise = 1e-4

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = QuadraticMeanGPModel(train_x, train_y, likelihood, 2.5)

# fix the hyperparameters
model.covar_module.lengthscale = lengthscale
model.likelihood.noise = noise

# train the hyperparameter (the constant)
optimizer = torch.optim.Adam(model.mean_module.parameters(), lr=0.01)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

model.train()
likelihood.train()

losses = []
first0s = []
second0s = []
first1s = []
second1s = []
mutuals = []
biases = []
for i in tqdm(range(1000)):
    optimizer.zero_grad()

    output = model(train_x)
    loss = -mll(output, train_y)

    loss.backward()

    losses.append(loss.item())
    first0s.append(model.mean_module.first0.item())
    second0s.append(model.mean_module.second0.item())
    first1s.append(model.mean_module.first1.item())
    second1s.append(model.mean_module.second1.item())
    mutuals.append(model.mean_module.mutual.item())
    biases.append(model.mean_module.bias.item())

    optimizer.step()

model.eval()
likelihood.eval()

fig, ax = plt.subplots(1, 5, figsize=(16, 4))

ax[0].plot(losses)
ax[0].set_ylabel("negative marginal log likelihood")

ax[1].plot(second0s, label="second 0")  # 绘制第一个维度的权重
ax[1].plot(second1s, label="second 1")  # 绘制第二个维度的权重
ax[1].set_ylabel("seconds")
ax[1].legend()

ax[2].plot(first0s, label="first 0")  # 绘制第一个维度的权重
ax[2].plot(first1s, label="first 1")  # 绘制第二个维度的权重
ax[2].set_ylabel("firsts")
ax[2].legend()

ax[3].plot(mutuals)
ax[3].set_ylabel("mutuals")

ax[4].plot(biases)
ax[4].set_ylabel("bias")

plt.tight_layout()

visualize_2d_contour(model, likelihood)
visualize_gp_belief(model, likelihood)
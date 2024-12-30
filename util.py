import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import gpytorch

import matplotlib.pyplot as plt

def visualize_gp_belief(model, likelihood, train_x, train_y, X1, X2, xs):
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

    ax.set_title("GP Regression (Constant mean)", fontsize=16)
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_zlabel('Predicted y', fontsize=12)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=6)

    plt.show()
    plt.close(fig)  


def visualize_2d_contour(model, likelihood, train_x, train_y, X1, X2, xs):
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
    plt.close(fig)  
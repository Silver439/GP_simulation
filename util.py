import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import gpytorch
import scipy.spatial as scpspatial
import scipy.stats as stats
import math


import matplotlib.pyplot as plt

def visualize_gp_belief(name, model, likelihood, train_x, train_y, X1, X2, xs):

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

    ax.set_title(name, fontsize=16)
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_zlabel('Predicted y', fontsize=12)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=6)

    plt.show()
    plt.close(fig)  


def visualize_2d_contour(name, predictive_distribution, train_x, X1, X2, xs, suggest_x):

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

    ax.scatter(suggest_x[:, 0], suggest_x[:, 1], color="orange", marker="o", label="Suggest Points")
    ax.scatter(min_x1, min_x2, color="red", marker="*", s=200, label="Min Value")
    ax.text(
        min_x1, min_x2, 
        f"Min: ({min_x1:.2f}, {min_x2:.2f})\nValue: {min_value:.2f}", 
        color="red", fontsize=10, ha="left", va="bottom"
    )

    ax.set_title(name, fontsize=16)
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("X2", fontsize=12)
    ax.legend()
    plt.tight_layout()

    plt.show()
    plt.close(fig)  

def visualize_rbf_belief(fX_pred, train_x, train_y, X1, X2):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X1, X2, fX_pred.reshape(X1.shape), 
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

    ax.set_title("RBF model", fontsize=16)
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_zlabel('Predicted y', fontsize=12)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=6)

    plt.show()
    plt.close(fig)  


def visualize_rbf_contour(fX_pred, train_x, X1, X2, suggest_x):

    predictive_mean_reshaped = fX_pred.reshape(X1.shape)
    min_value = predictive_mean_reshaped.min()
    min_idx = np.unravel_index(predictive_mean_reshaped.argmin(), predictive_mean_reshaped.shape)
    min_x1 = X1[min_idx]
    min_x2 = X2[min_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X1, X2, predictive_mean_reshaped, cmap="coolwarm", levels=50)
    plt.colorbar(contour, ax=ax, label="Predicted y")

    ax.scatter(train_x[:, 0], train_x[:, 1], color="black", marker="o", label="Training Points")

    ax.scatter(suggest_x[:, 0], suggest_x[:, 1], color="orange", marker="o", label="Suggest Points")

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

    ax.set_title('RBF model', fontsize=16)
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("X2", fontsize=12)
    ax.legend()
    plt.tight_layout()

    plt.show()
    plt.close(fig)  


def unit_rescale(x):
    """Shift and rescale elements of a vector to the unit interval

    :param x: array that should be rescaled to the unit interval
    :type x: numpy.ndarray
    :return: array scaled to the unit interval
    :rtype: numpy.ndarray
    """

    x_max = x.max()
    x_min = x.min()
    if x_max == x_min:
        return np.ones(x.shape)
    else:
        return (x - x_min) / (x_max - x_min)


def weighted_distance_merit(num_pts, surrogate, X, cand, weights, dtol=1e-3):

    # Distance
    dim = X.shape[1]
    dists = scpspatial.distance.cdist(cand, np.vstack(X))
    dmerit = np.amin(dists, axis=1, keepdims=True)

    # Values
    fvals = surrogate.predict(cand)
    fvals = unit_rescale(fvals)

    # Pick candidate points
    new_points = np.ones((num_pts, dim))
    for i in range(num_pts):
        w = weights[i]
        merit = w * fvals + (1.0 - w) * (1.0 - unit_rescale(np.copy(dmerit)))

        merit[dmerit < dtol] = np.inf
        jj = np.argmin(merit)
        fvals[jj] = np.inf
        new_points[i, :] = cand[jj, :].copy()

        # Update distances and weights
        ds = scpspatial.distance.cdist(cand, np.atleast_2d(new_points[i, :]))
        dmerit = np.minimum(dmerit, ds)

    return new_points
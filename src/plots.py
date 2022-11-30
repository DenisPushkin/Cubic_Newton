import plotly.graph_objects as go
import matplotlib.pyplot as plt
from trainers import get_metrics, get_hessian_metrics
import numpy as np


def visualize_spectrum_evolution(trainer, plot_every_iter=1, bin_size=0.2, y_range=None):

    fig = go.Figure()
    
    #check that have something to plot
    assert len(trainer.hessian_metrics["spectrum_iter"]) > 0
    
    iterations = []

    min_val = min([min(spectrum) for spectrum in trainer.hessian_metrics["spectrum"]])
    max_val = max([max(spectrum) for spectrum in trainer.hessian_metrics["spectrum"]])

    for i, spectrum_iter in enumerate(trainer.hessian_metrics["spectrum_iter"]):
        if spectrum_iter % plot_every_iter == 0:
            iterations.append(spectrum_iter)
            fig.add_trace(
                go.Histogram(
                    visible=False,
                    x = trainer.hessian_metrics["spectrum"][i],
                    xbins={"start": min_val-0.1, "end": max_val+0.1, "size": bin_size},
                )
            )
    
    fig.data[0].visible = True
    
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Iter: " + str(iterations[i])}]
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Iter: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        xaxis_range=[min_val-1., max_val+1.],
        yaxis_range=y_range,
    )

    fig.show()


def visualize_hessians_evolution(trainer, plot_every_iter=1):
    fig = go.Figure()
    
    #check that have something to plot
    assert len(trainer.hessian_metrics["hessian_iter"]) > 0
    
    iterations = []
    
    for i, hessian_iter in enumerate(trainer.hessian_metrics["hessian_iter"]):
        if hessian_iter % plot_every_iter == 0:
            iterations.append(hessian_iter)
            fig.add_trace(
                go.Heatmap(
                    visible=False,
                    z = trainer.hessian_metrics["hessian"][i],
                    colorscale='RdBu',
                    zmid=0,
                    reversescale = True,
                )
            )
    
    fig.data[0].visible = True
    
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Iter: " + str(iterations[i])}]
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Iter: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders,
        width=600,
        height=600
    )

    fig.show()


def plot_M(trainer_):
    plt.scatter(trainer_.metrics["iter"], trainer_.metrics["M"], s=1)
    plt.xlabel("Iteration")
    plt.ylabel("M")
    plt.yscale('log')
    plt.show()


def get_psd_ness(trainer_):
    # we used lambda_n computed after previous step, but corresponding M and r are computed after current step
    CN_metrics = get_metrics(trainer_)
    CN_hessian_metrics = get_hessian_metrics(trainer_)
    iters = []
    psd_ness = []
    for i, iter in enumerate(CN_hessian_metrics["iter"]):
        if (iter+1) in CN_metrics["iter"]:
            iters.append(iter+1)
            j = list(CN_metrics["iter"]).index(iter+1)
            psd_ness.append(CN_hessian_metrics["lambda_n"][i] + CN_metrics["M"][j]*CN_metrics["step_size"][j])

    return np.array(iters), np.array(psd_ness)


def plot_psd_ness(trainer_):
    iters, psd_ness = get_psd_ness(trainer_)

    print(f'min eigenvalue of (H + Mr/2*I) over all observed iterations: {np.min(psd_ness): .8f}')

    plt.scatter(iters[psd_ness>0], psd_ness[psd_ness>0], s=2)
    plt.xlabel("Iteration")
    plt.ylabel("lambda_n (H + Mr/2*I)")
    plt.yscale('log')
    plt.show()


def plot_losses(metrics_list, labels_list, min_train_loss=0., min_test_loss=0.):
    fig, axes = plt.subplots(1,2, figsize=(12,6))

    ax = axes[0]
    for metrics, label in zip(metrics_list, labels_list):
        ax.plot(metrics["iter"], metrics["train_loss"] - min_train_loss, label=label)
    ax.set_xlabel("Iteration")
    y_label = "Train loss" if min_train_loss == 0 else "Train loss overhead"
    ax.set_ylabel(y_label)
    ax.set_yscale('log')
    ax.legend()

    ax = axes[1]
    for metrics, label in zip(metrics_list, labels_list):
        ax.plot(metrics["iter"], metrics["test_loss"] - min_test_loss, label=label)
    ax.set_xlabel("Iteration")
    y_label = "Test loss" if min_test_loss == 0 else "Test loss overhead"
    ax.set_ylabel(y_label)
    ax.set_yscale('log')
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_errors(metrics_list, labels_list):
    fig, axes = plt.subplots(1,2, figsize=(12,6))

    ax = axes[0]
    for metrics, label in zip(metrics_list, labels_list):
        ax.plot(metrics["iter"], 1 - metrics["train_acc"], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Train error rate")
    ax.set_yscale('log')
    ax.legend()

    ax = axes[1]
    for metrics, label in zip(metrics_list, labels_list):
        ax.plot(metrics["iter"], 1 - metrics["test_acc"], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test error rate")
    ax.set_yscale('log')
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_grad_norms(metrics_list, labels_list):
    for metrics, label in zip(metrics_list, labels_list):
        plt.plot(metrics["iter"], metrics["grad_norm"], label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient norm")
    plt.yscale('log')
    plt.legend()
    plt.show()


def plot_max_min_eigvals(metrics_list, labels_list, plot_min_eigval=True, yscale=None):
    fig, axes = plt.subplots(1,2, figsize=(12,6))

    ax = axes[0]
    for metrics, label in zip(metrics_list, labels_list):
        ax.plot(metrics["iter"], metrics["lambda_1"], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max Eigenvalue")
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend()

    if plot_min_eigval:
        ax = axes[1]
        for metrics, label in zip(metrics_list, labels_list):
            ax.plot(metrics["iter"], -metrics["lambda_n"], label=label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Minus Min Eigenvalue")
        ax.set_yscale('log')
        ax.legend()
        fig.tight_layout()
    
    plt.show()


def plot_distances(metrics_list, labels_list, plot_min_eigval=True, yscale=None):
    fig, axes = plt.subplots(1,2, figsize=(12,6))

    ax = axes[0]
    for metrics, label in zip(metrics_list, labels_list):
        ax.plot(metrics["iter"], metrics["dist_from_start"], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance from start")
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend()

    ax = axes[1]
    for metrics, label in zip(metrics_list, labels_list):
        ax.plot(metrics["iter"], metrics["step_size"], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step size")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    plt.show()
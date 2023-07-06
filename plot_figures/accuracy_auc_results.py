import pandas as pd
import seaborn as sns

sns.set(style="ticks", context="poster", font_scale=1)
import matplotlib.pyplot as plt
import matplotlib.colors as matcolors
from matplotlib.gridspec import GridSpec
from scipy import stats
import numpy as np


def accuracy_auc_all_cnn():
    df = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/plot_figures/accuracy_auc_for_plotting.xlsx",
    )
    fig, ax = plt.subplots(figsize=(20, 12))
    df = df[df["Classification"] == "donor_lung_outcome"]
    df = df[df["Trend"] == "No"]
    df = df[df["Metric"] == "AUROC"]
    df["Value"] = df["Value"].astype(float)

    plot(
        x="Model",
        y="Value",
        hue="Pretrain",
        data=df,
        ylabel="Accuracy",
        legend="Baseline/Pretrain",
        filepath="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/plot_figures/multiclass_notrend_AUROC.png",
    )


def accuracy_auc_cnn_xgboost():
    df_binary = pd.DataFrame()
    df_multiclass = pd.DataFrame()

    df_binary["Metric"] = ["Accuracy", "AUROC", "Accuracy", "AUROC"]
    df_binary["Method"] = [
        "XGBoost\n(Manual Labels)​",
        "XGBoost\n(Manual Labels)​",
        "CNN\n(Automatically Extracted Features)​",
        "CNN\n(Automatically Extracted Features)​",
    ]
    df_binary["Value"] = [80.0, 90.3, 86.9, 89.8]

    df_multiclass["Metric"] = ["Accuracy", "AUROC", "Accuracy", "AUROC"]
    df_multiclass["Method"] = [
        "XGBoost\n(Manual Labels)​",
        "XGBoost\n(Manual Labels)​",
        "CNN\n(Automatically Extracted Features)​",
        "CNN\n(Automatically Extracted Features)​",
    ]
    df_multiclass["Value"] = [52.0, 77.3, 66.9, 78.4]

    plot(
        x="Method",
        y="Value",
        hue="Metric",
        data=df_binary,
        ylabel="Transplant Decision Classification",
        legend=None,
        filepath="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/plot_figures/binary_cnn_xgboost.png",
    )
    plot(
        x="Method",
        y="Value",
        hue="Metric",
        data=df_multiclass,
        ylabel="Outcome Classification",
        legend=None,
        filepath="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/plot_figures/multiclass_cnn_xgboost.png",
    )


def plot(x, y, hue, data, ylabel, legend, filepath):
    fig, ax = plt.subplots(figsize=(15, 9))
    sns.barplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        ax=ax,
        saturation=0.6,
        palette="pastel",
    )
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 10
    ax.set_ylim(50, 100)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.set_xlabel(None, fontsize=30)
    ax.legend(
        title=legend,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(
        filepath,
        dpi=200,
    )
    plt.show()
    plt.close()


accuracy_auc_cnn_xgboost()

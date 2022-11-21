"""Visualization methods"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_curve, average_precision_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score


from src import model
from src import config


def plot_correlation(corr):
    """Plot correlation heatmap with diverging palette and masked triangle"""
    ax = sns.heatmap(corr, cmap=sns.diverging_palette(230, 20, as_cmap=True), annot=False, mask=np.triu(np.ones_like(corr, dtype=bool)))
    return ax


def plot_distributions(df, column_names=None, sort_index=True, dropna=False, normalize=False):
    """Plot distribution of all columns or user-specified list.
    Checks number of unique values in column to determine optimal plotting method.
    Options to normalize and include missing values.
    """
    # if only one column is given as string, create an iterable list
    if isinstance(column_names, str):
        column_names = [column_names]

    # plot all columns
    if not column_names:
        column_names = df.columns

    # fig, ax = plt.subplots(len(column_names), 1, figsize=(8, len(column_names) * 4))
    sns.set_theme(style='ticks')
    sns.despine()

    for idx, col in enumerate(column_names):
        fig, ax = plt.subplots(figsize=(8, 4))

        column_value_counts = df[col].value_counts(dropna=dropna, normalize=normalize)
        if sort_index:
            column_value_counts = column_value_counts.sort_index()

        bar_position = np.arange(len(column_value_counts.values))
        bar_width = 0.35

        # with small column cardinality plot original distribution as bars, else plot as line
        if len(column_value_counts.values) <= 25:
            ax.bar(x=bar_position, height=column_value_counts.values, width=bar_width)
            # include values above bars
            if not normalize:
                ax.bar_label(ax.containers[0])
        else:
            ax.plot(bar_position + bar_width, column_value_counts.values, marker='o',
                      markersize=3, linewidth=2)

        # display x-ticks and labels depending on the cardinality of the column
        if df[col].nunique() <= 50:
            ax.set_xticks(bar_position + bar_width / 2)
            if df[col].nunique() <= 25:
                ax.set_xticklabels(column_value_counts.keys(), rotation=25)
            else:
                ax.set_xticklabels('')
        else:
            ax.set_xticks([], [])

        ax.set_title(col)
        if normalize:
            ax.set_ylabel('Probability')
        else:
            ax.set_ylabel('Count')

        plt.show()
        # fig.tight_layout()


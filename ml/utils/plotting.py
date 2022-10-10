import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc


# from tensorboard.backend.event_processing import event_accumulator


# def plot_raw_signal(input_data, input_data_norm_1, input_data_norm_2):
#     num_subs = input_data.shape[0]
#     fig, axs = plt.subplots(3,num_subs)
#     # axs.ravel()
#     data = [input_data, input_data_norm_1, input_data_norm_2]
#     for i in range(3):
#         for ax, s in zip(axs[i], data[i]):
#             axs.plot(s)


def plot_confusion_matrix(confusion_matrix, class_names, errors_only=False, figsize=(15, 6), fontsize=16):
    """
    # `Multiclass confusion matrix function`
    # By Justin Mackie, jmackie at gmail dot com

    Plots confusion matrix as a color-encoded Seaborn heatmap.  Zeroes are
    colored white.  Normalized values that are zero when rounded to three
    decimals, Ex. 0.000, will be colored white.  Get more decicmals by
    updating fmt, for example to '0.4f', and updating get_text() value.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object sklearn.metrics.confusion_matrix.
    class_names: list
        List of class names in the order they index the confusion matrix.
    figsize: tuple
        A pair tuple.  The first value is figure width.  The second
        value is figure height. Defaults to (15,6).
    fontsize: int
        Font size for axes labels. Defaults to 16.
    """
    # Instantiate Figure
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plt.subplots_adjust(wspace=0.5)

    # Show errors only by filling diagonal with zeroes.
    if errors_only:
        np.fill_diagonal(confusion_matrix, 0)

        # ax1 - Normalized Confusion Matrix
    # Normalize by dividing (M X M) matrix by (M X 1) matrix.  (M X 1) is row totals.
    conf_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # fix any nans caused by zero row total
    df_cm_norm = pd.DataFrame(conf_matrix_norm, index=class_names, columns=class_names)
    heatmap = sns.heatmap(df_cm_norm, ax=ax1, cmap='Blues', fmt='.3f', annot=True, annot_kws={"size": fontsize},
                          linewidths=2, linecolor='black', cbar=False)

    ax1.tick_params(axis='x', labelrotation=0, labelsize=fontsize, labelcolor='black')
    ax1.tick_params(axis='y', labelrotation=0, labelsize=fontsize, labelcolor='black')
    ax1.set_ylim(ax1.get_xlim()[0], ax1.get_xlim()[1])  # Fix messed up ylim
    ax1.set_xlabel('PREDICTED CLASS', fontsize=fontsize, color='black')
    ax1.set_ylabel('TRUE CLASS', fontsize=fontsize, color='black')
    ax1.set_title('Confusion Matrix - Normalized', pad=15, fontsize=fontsize, color='black')

    # ax2 - Confusion Matrix - Class Counts
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    heatmap = sns.heatmap(df_cm, ax=ax2, cmap='Blues', fmt='d', annot=True, annot_kws={"size": fontsize + 4},
                          linewidths=2, linecolor='black', cbar=False)

    ax2.tick_params(axis='x', labelrotation=0, labelsize=fontsize, labelcolor='black')
    ax2.tick_params(axis='y', labelrotation=0, labelsize=fontsize, labelcolor='black')
    ax2.set_ylim(ax1.get_xlim()[0], ax1.get_xlim()[1])  # Fix bug in matplotlib 3.1.1.  Or, use earlier matplotlib.
    ax2.set_xlabel('PREDICTED CLASS', fontsize=fontsize, color='black')
    ax2.set_ylabel('TRUE CLASS', fontsize=fontsize, color='black')
    ax2.set_title('Confusion Matrix - Class Counts', pad=15, fontsize=fontsize, color='black')

    return plt


def plot_cm(cm, class_type, figsize=(15, 6), fontsize=16, fig_path=None):
    if class_type == 1:
        class_names = ["healthy", "IR_damage_real", "OR_damage_real"]
        num_classes = len(class_names)
    elif class_type == 2:
        class_names = ["healthy", "IR_damage_real", "OR_damage_real"]
        num_classes = len(class_names)
    elif class_type == 5:
        class_names = ["healthy", "IR_damage_1", "IR_damage_2", "OR_damage_1", "OR_damage_2", "IO_OR_damage"]
        num_classes = len(class_names)
        font = {'family': 'monospace',
                'weight': 'bold',
                'size': 10}
        rc('font', **font)
    else:
        raise NotImplemented

    group_counts = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            group_counts.append("{}/{}".format(cm[i, j], cm.sum(axis=1)[j]))
    group_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    group_percentages = np.nan_to_num(group_percentages)
    labels = [f"{v1} \n {v2 * 100: .4}%" for v1, v2 in zip(group_counts, group_percentages.flatten())]
    labels = np.asarray(labels).reshape(num_classes, num_classes)
    plt.ioff()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=labels, ax=ax, fmt="", cmap="Blues", cbar=True)
    # sns.heatmap(cm, annot=True, ax=ax, fmt="", cmap="Blues", cbar=True)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels', fontsize=fontsize, color='black')
    ax.set_ylabel('True labels', fontsize=fontsize, color='black')
    # ax.set_title('Confusion Matrix of real damage')
    True_class_name = [str(i) for i in range(len(class_names))]
    ax.xaxis.set_ticklabels(class_names, fontsize=10, color='black')
    ax.yaxis.set_ticklabels(class_names, rotation=30, fontsize=10, color='black')
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight")
    else:
        plt.show()


# def cm_from_best_model(test_data, best_model):
#     model = torch.load(PATH)
#     model.eval()
#     pass

# def plot_from_tensorboard(events_dir_name, tag, figsize=(15, 6), fontsize=16, fig_path=None):
#     event_dir_path = event_dir_name
#     event_file_name = os.listdir(event_dir_path)[0]
#     data = {}
#     event_file_path = os.path.join(event_dir_path, event_file_name)
#     ea = event_accumulator.EventAccumulator(event_file_path)
#     ea.Reload()
#     metrics = ea.Tags()["scalars"]
#     for metric in metrics:
#         data[metric] = ea.Scalars(metric)
#     pass

def plot_from_csv(to_plot_folder, labels,title, epoch, figsize=(15, 6), fontsize=16, fig_path=None):
    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 10}
    rc('font', **font)
    # RUN_PATH = Path(__file__).parent.parent.parent / "runs"
    # to_plot_folder = os.path.join(RUN_PATH, folder_name)
    files_path = os.listdir(to_plot_folder)
    data = {}
    for file, label in zip(files_path, labels):
        data[label] = pd.read_csv(os.path.join(to_plot_folder, file))["Value"].tolist()


    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('epochs', fontsize=fontsize, color='black')
    ax.set_ylabel('Accuracy', fontsize=fontsize, color='black')
    ax.set_title(title)
    for k, v in data.items():
        ax.plot(v, label=k)

    ax.set_title(title)
    # sns.lineplot(data=data)
    plt.axvline(x=epoch, color="r", label=f'epoch = {epoch}')
    ax.legend()
    fig.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight")
    else:
        plt.show()

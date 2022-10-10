"""
Created on 25.09.20
@author :ali
"""
import os
from pathlib import Path

import torch

from ml.model.nn_models import create_ffn_net
from ml.utils.data import prepare_dataset, load_bearing_data_cnn
from ml.utils.file_managment import yaml_argparse
from ml.utils.plotting import plot_from_csv

BEARING_DATA_PATH = Path(__file__).parent / "params"
RUN_PATH = Path(__file__).parent.parent / "runs"


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    return model


def main(raw_args=None):
    baseline_yaml_path = os.path.join(BEARING_DATA_PATH, "testing_dnn.yaml")
    args = yaml_argparse(baseline_yaml_path, raw_args=raw_args)
    margs = yaml_argparse(args.hparam_path)

    if args.plot_acc:
        plot_path = os.path.join(RUN_PATH, args.plot_path)
        plot_from_csv(folder_name=plot_path, labels=args.labels, fig_path=args.fig_path)

    if args.enable_testing:

        test_set = load_bearing_data_cnn(margs.class_type, margs.signal_type, margs.filter, margs.bearing_name_test,
                                         margs.num_measurement_test, margs.segement_size, margs.norm_type_input)
        test_data, test_loader, _, _, num_samples_test = \
            prepare_dataset(test_set[0], test_set[1], [1., .0, 0.], batch_size=0)

        # run_path = os.path.join(RUN_PATH, args.experiment_name, args.run_name)

        net_size = [test_data.tensors[0].shape[1]] + margs.net_size  # input dimension
        model = create_ffn_net(net_size, margs.activations, margs.layer_types,
                               margs.skip_connections, weight_init=margs.weight_init, batch_norm=margs.batch_norm,
                               eps=float(margs.eps), momentum=margs.momentum, dropout=margs.dropout)
        model.load_state_dict(torch.load(args.best_model_path))
        model.eval()

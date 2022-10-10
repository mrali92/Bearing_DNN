from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events
import numpy as np


def _collect_metric_on_epoch(engine, metrics_engine, metric, ep, metrics_dict):
    if engine.state.epoch == ep:
        key = "{:}{:}".format(metric, ep)
        if isinstance(metrics_engine, list):
            num_metrics = len(metrics_engine)
            value = 0.
            for name, engine, _ in metrics_engine:
                value += engine.state.metrics[metric]
            metrics_dict[key] = value / num_metrics
        else:
            metrics_dict[key] = metrics_engine.state.metrics[metric]

def log_metrics_on_epochs(engine, metrics_engine, metrics, args):
    epochs = tuple(np.floor(np.linspace(0, args.num_epochs, 4)[1::]).astype(np.int))
    metrics_dict = {}
    for metric in metrics:
        for epoch in epochs:
            engine.add_event_handler(Events.EPOCH_COMPLETED, _collect_metric_on_epoch, metrics_engine, metric,
                                     epoch, metrics_dict)
    return metrics_dict


def log_hparams(tensorboard_logger: TensorboardLogger, train_engine, metrics_dict,
                param_dict):
    from tensorboardX.summary import hparams

    param_dict = dict(param_dict)
    rem_keys = []

    for k, v in param_dict.items():

        if type(v) == bool:
            v = int(v)
        if type(v) == list or type(v) == tuple:
            if all([isinstance(e, str) for e in v]):
                v = "-".join(v)
            elif any([isinstance(e, str) for e in v]):
                v = "-".join([str(e) for e in v])
            elif all([isinstance(e, bool) for e in v]):
                v = "-".join(str(int(e)) for e in v)
            elif any([isinstance(e, int) for e in v]):
                v = "-".join(str(i) for i in v)
        elif v is None:
            rem_keys.append(k)
        param_dict[k] = v

    for k in rem_keys:
        del param_dict[k]

    writer = tensorboard_logger.writer

    def _add_hparams(engine, *args, **kwargs):
        exp, ssi, sei = hparams(param_dict, metrics_dict)
        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)
        for k, v in metrics_dict.items():
            writer.add_scalar(k, v)

    train_engine.add_event_handler(Events.COMPLETED, _add_hparams)

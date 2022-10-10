"""
Created on 15.08.20
@author :ali
"""

import datetime
import gc
import signal
import subprocess

from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.handlers.early_stopping import EarlyStopping
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, TopKCategoricalAccuracy, Fbeta
from ignite.utils import setup_logger
from torch.optim.lr_scheduler import ExponentialLR
from torchsummary import summary

from ml.model.nn_models import *
from ml.utils.data import *
from ml.utils.data import _prepare_batch, load_bearing_data_cnn
from ml.utils.file_managment import yaml_argparse, args_to_dict, args_to_yaml
from ml.utils.plotting import plot_cm

BEARING_DATA_PATH = Path(__file__).parent / "params"
RUN_PATH = Path(__file__).parent / "runs"


def main(raw_args=None):
    pid = os.getpid()
    subprocess.Popen("renice -n 10 -p {}".format(pid), shell=True)

    baseline_yaml_path = os.path.join(BEARING_DATA_PATH, "conv_network.yaml")
    args = yaml_argparse(baseline_yaml_path, raw_args=raw_args)

    if args.log_dir == "":
        log_dir = Path(__file__).parent.parent / 'runs/{}/{}/{}'.format(
            args.experiment_name, args.run_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    device = args.device
    if device == "cuda":
        torch.device("cuda:" + str(args.gpu_id))
    else:
        torch.device('cpu')




    # Loading and prepare Training data
    train_set = load_bearing_data_cnn(args.class_type, args.signal_type, args.filter, args.bearing_name_train,
                                      args.num_measurement_train, args.segement_size, args.norm_type_input)

    data, train_loader, eval_loader, _, num_samples_train = \
        prepare_dataset(train_set[0], train_set[1], args.train_dev_test_split, args.batch_size)

    # Start testing
    if args.enable_testing:
        test_set = load_bearing_data_cnn(args.class_type, args.signal_type, args.filter, args.bearing_name_test,
                                         args.num_measurement_test, args.segement_size, args.norm_type_input)
        data, test_loader, _, _, num_samples_test = \
            prepare_dataset(test_set[0], test_set[1], [1., .0, 0.], batch_size=0)
        plt_name = "test_set"
    else:
        test_loader = eval_loader
        plt_name = "eval_set"

    # Create cnn model
    if args.convNet_type == "AutoEncoder":
        model = ConvAutoencoder(
            conv_channels_en=args.conv_channels_en,
            conv_kernel_size_en=args.conv_kernel_size_en,
            conv_stride_en=args.conv_stride_en,
            conv_padding_en=args.conv_padding_en,
            conv_out_padding_en=args.conv_out_padding_en,
            conv_channels_de=args.conv_channels_de,
            conv_kernel_size_de=args.conv_kernel_size_de,
            conv_stride_de=args.conv_stride_de,
            conv_padding_de=args.conv_padding_de,
            conv_out_padding_de=args.conv_out_padding_de,
            bottleneck_layer=args.bottleneck_layer,
            output_layer=args.output_layer,
            en_functionals=args.en_functionals,
            en_batch_norm=args.en_batch_norm,
            en_dropout=args.en_dropout,
            de_functionals=args.de_functionals,
            de_batch_norm=args.de_batch_norm,
            de_dropout=args.de_dropout,
            eps=float(args.eps),
            momentum=args.momentum,
            input_dim=args.segement_size)

    else:
        model = ConvNet(convNet_type=args.convNet_type,
                        conv_channels=args.conv_channels_en,
                        conv_kernel_size=args.conv_kernel_size_en,
                        conv_stride=args.conv_stride_en,
                        conv_padding=args.conv_padding_en,
                        conv_out_padding=args.conv_out_padding_en,
                        fc_net_size=args.output_layer,
                        functionals=args.en_functionals,
                        batch_norm=args.en_batch_norm,
                        eps=float(args.eps),
                        momentum=args.momentum,
                        dropout=args.en_dropout,
                        input_dim=args.segement_size)
    model.to(device)

    optimizer = getattr(torch.optim, args.optim)(model.parameters(), lr=float(args.lr))
    loss_fn = getattr(torch.nn, args.loss)()

    def thresholded_output_transform(output):
        y_pred, y = output
        y_pred = torch.round(y_pred)
        return y_pred, y

    if args.loss == "CrossEntropyLoss":
        if args.class_type == 1 or args.class_type == 2:
            metrics = {
                'accuracy': Accuracy(output_transform=thresholded_output_transform),
                'loss': Loss(loss_fn),
                'cm': ConfusionMatrix(num_classes=3),
                "F1": Fbeta(beta=1)}
        elif args.class_type == 5:
            metrics = {
                'accuracy': Accuracy(output_transform=thresholded_output_transform),
                'loss': Loss(loss_fn),
                'cm': ConfusionMatrix(num_classes=6),
                "F1": Fbeta(beta=1),
                'Top-5 Accuracy': TopKCategoricalAccuracy(k=6)}

        transform_batch = _prepare_batch

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device, prepare_batch=transform_batch)
    trainer.logger = setup_logger("Trainer")

    train_evaluator = create_supervised_evaluator(model,
                                                  metrics=metrics,
                                                  device=device, prepare_batch=transform_batch)
    train_evaluator.logger = setup_logger("Train Evaluator")

    val_evaluator = create_supervised_evaluator(model,
                                                metrics=metrics,
                                                device=device, prepare_batch=transform_batch)
    val_evaluator.logger = setup_logger("Val Evaluator")

    test_evaluator = create_supervised_evaluator(model,
                                                 metrics=metrics,
                                                 device=device, prepare_batch=transform_batch)
    test_evaluator.logger = setup_logger("test Evaluator")

    lr_scheduler = ExponentialLR(optimizer, gamma=0.975)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())

    # Tabular Logging
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    # Tensorboard Logging
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # log model graph
    tb_logger.writer.add_graph(model, torch.zeros((1, 1, args.segement_size)).to(device))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        avg_f1 = metrics["F1"]
        pbar.log_message(
            "Training Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f} Avg F1_score:{:.3f}".format(
                engine.state.epoch, avg_accuracy, avg_loss, avg_f1
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_evaluator.run(eval_loader)
        metrics = val_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        avg_f1 = metrics["F1"]
        pbar.log_message(
            "Validation Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f} Avg F1_score:{:.3f}".format(
                engine.state.epoch, avg_accuracy, avg_loss, avg_f1
            )
        )

        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.EPOCH_COMPLETED(every=20))
    def log_validation_results(engine):
        test_evaluator.run(test_loader)
        metrics = test_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        avg_f1 = metrics["F1"]
        pbar.log_message(
            "Testing Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f} Avg F1_score:{:.3f}".format(
                engine.state.epoch, avg_accuracy, avg_loss, avg_f1))
        metrics = test_evaluator.state.metrics
        cm = metrics['cm']
        cm = cm.numpy()
        cm = cm.astype(int)
        fig_path = os.path.join(log_dir, "cm_{}_ep_{}.png".format(plt_name, trainer.state.epoch))
        plot_cm(cm, args.class_type, fig_path=fig_path)

        tb_logger.attach_output_handler(
            test_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="testing_eval",
            metric_names=["accuracy", "loss", "F1"],
            global_step_transform=global_step_from_engine(trainer),
        )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"loss": loss},
        metric_names="all",
    )

    for tag, evaluator in [("training_eval", train_evaluator), ("validation_eval", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=["accuracy", "loss", "F1"],
            global_step_transform=global_step_from_engine(trainer),
        )

    # tb_logger.attach(trainer, log_handler=WeightsScalarHandler(model), event_name=Events.ITERATION_COMPLETED(every=100))

    tb_logger.attach(trainer, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED(every=100))

    # tb_logger.attach(trainer, log_handler=GradsScalarHandler(model), event_name=Events.ITERATION_COMPLETED(every=100))

    tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED(every=100))

    # Log optimizer parameters
    tb_logger.attach(trainer,
                     log_handler=OptimizerParamsHandler(optimizer, "lr"),
                     event_name=Events.EPOCH_STARTED)

    # Early stopping with patience of 5 when score_fn gives a result lower than the best result
    def score_function(engine):
        # return engine.state.metrics["F1"]
        return engine.state.metrics["accuracy"]

    es_handler = EarlyStopping(patience=40, score_function=score_function, trainer=trainer)
    train_evaluator.add_event_handler(Events.COMPLETED, es_handler)
    setup_logger("es_handler")

    # log Hparams in tensorboard TODO: fix logging scalar bug
    hparam_dict = args_to_dict(args)
    hparam_dict["num_parameters"] = model.parameters()
    # hparam_metrics = log_metrics_on_epochs(trainer,
    #                                        train_evaluator,
    #                                        ["loss", "accuracy", "F1"],
    #                                        args)
    # log_hparams(tb_logger, trainer, hparam_metrics, hparam_dict)

    # save hparams in yaml file
    # dict_to_yaml(list(hparam_dict), os.path.join(log_dir, "Hparams.yaml")) # TODO fix pickle error
    args_to_yaml(args, os.path.join(log_dir, "Hparams.yaml"))
    with Tee(os.path.join(log_dir, "model_summary.txt")):
        summary(model, (1, args.segement_size), batch_size=args.batch_size, device=device)
    # Model checkpoint

    model_checkpoint = ModelCheckpoint(
        log_dir,
        n_saved=2,
        filename_prefix="best",
        score_function=score_function,
        score_name="validation_accuracy",
        global_step_transform=global_step_from_engine(val_evaluator),
    )
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    # Start training
    trainer.run(train_loader, max_epochs=args.num_epochs)
    # close tensorboard session
    tb_logger.close()

    # free gpu memeory after finish
    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        gc.collect()
        os.kill(pid, signal.SIGSTOP)

    trainer.add_event_handler(Events.TERMINATE, empty_cuda_cache)
    evaluator.add_event_handler(Events.TERMINATE, empty_cuda_cache)
    train_evaluator.add_event_handler(Events.TERMINATE, empty_cuda_cache)


if __name__ == "__main__":
    main(raw_args=None)

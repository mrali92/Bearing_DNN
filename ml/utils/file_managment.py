"""
Created on 04.08.20
@author :ali
"""
import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import yaml


def prepare_log_dir(experiment_name, run_name, root_log_dir):
    """Creates a experiment directory to save the console logging file of each run and its corresponding script.

    The experiment_name determines the directory name. It contains a sub-folder named after the runtime
    and the given run_name argument. An experiment .py script and a console logging .txt file are stored
    after each run. The root_log_dir sets the root path of the experiment folder.

    :param experiment_name: Name of the experiment folder
    :type experiment_name: str
    :param run_name: Name of the experiment run folder.
    :type run_name: str
    :param root_log_dir: Path where the experiment folder is stored.
    :returns: Path where the log directory is saved and the Tee object of the log file
    :rtype: (str,Tee)

    """
    log_dir = create_working_folder(root_log_dir, experiment_name=experiment_name, run_name=run_name)
    save_main_script_to_path(log_dir)
    # log_file = Tee(sys.stdout, open(os.path.join(log_dir, "console_log.txt"), 'a'))

    return log_dir


def save_main_script_to_path(path):
    """Saves the running script as text file.

    :param path: Path where to save the script

    """
    if not os.path.isdir(path):
        raise NotADirectoryError("Path {} is not a directory.".format(path))
    script_path = Path(sys.argv[0])
    if path[-1] != os.sep:
        path = path + os.sep
    copyfile(script_path, Path(path) / script_path.name)


def create_working_folder(root_path, experiment_name, run_name):
    """ Creates a folder structure located in the root path.

    This function create an experiment directory in the given root_path.
    the experiment directory contains a run folder, which is named after the run time and the run_name parameter

    :param root_path: Path of the created folder structure
    :type root_path : str
    :param experiment_name: Name of the experiment folder
    :type experiment_name: str
    :param run_name: Name of the run folder
    :type run_name: str
    """
    if not Path(root_path).is_dir():
        raise NotADirectoryError("Argument root_path must be an existing directory.")
    date_str = datetime.now().strftime("%y%m%d_%H%M%S")
    wdir = os.path.join(root_path, experiment_name, date_str + "_" + run_name)

    if os.path.isdir(wdir):
        raise RuntimeWarning("Directory {:} already exists.".format(wdir))
    os.makedirs(wdir, exist_ok=True)

    return wdir


def yaml_argparse(yaml_path, raw_args=None):
    """
    Returns hyperparameters dict as a ArgumentParser object .

    This function first needs to have a template yaml file, where all needed hyperparameters set is defined.
    Then according to the given argument, the function sets a new value of an existing hyperparameter key
    or add a new one.
    These hyperparameters are stored as Argumentparser object and can be accessed directly with the dot notation.

    Also, you can add yaml file, that contained customized configuration of hyperparameters, as an argument.
    The new values will be overridden, otherwise the default values wil remain.

    :param yaml_path: The path of the default hyperparameter yaml file.
    :param raw_args:  Sets value of an exiting hyperparameter argument e.g raw_args=['--key', 'value' ]
                     (Default value = None)

    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="yaml_file",
                        help="yaml file containing experiment parameters. Stored as attribute 'yaml_file' in parsed "
                             "args.",
                        metavar="FILE",
                        required=False)

    class ListAction(argparse.Action):
        """Creates list from string containing a list expression"""

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, eval(values))

    hparam_dict = load_yaml(yaml_path)

    for key, value in hparam_dict.items():
        # kwargs unpacking
        if isinstance(value, dict):
            kwargs = dict(value)
            value = kwargs.pop("default")
        else:
            kwargs = {}

        if isinstance(value, list):
            parser.add_argument("--" + key, action=ListAction, default=value, **kwargs)
        else:
            parser.add_argument("--" + key, type=type(value), default=value, **kwargs)

    def _get_cmd_args(args):
        cmd_args = []
        [cmd_args.append(x[2::]) for x in args if x[0:2] == "--"]
        return cmd_args

    # get command line arguments to prevent overriding them when applying parameters from another yaml file
    if raw_args:
        raw_args_cmd = _get_cmd_args(raw_args)
    elif len(sys.argv) > 1:
        raw_args_cmd = _get_cmd_args(sys.argv[1::])
    else:
        raw_args_cmd = []

    args = parser.parse_args(raw_args)
    if args.yaml_file:
        hparam_dict_yaml = load_yaml(args.yaml_file)
        # hparam analysis
        [print("Obsolete parameter '{:}' in {:}".format(key, args.yaml_file)) for key in hparam_dict_yaml.keys()
         if (key not in hparam_dict and not key == "yaml_file")]
        [print("Missing parameter {:} in {:}. Using default value.".format(key, args.yaml_file)) for key in
         hparam_dict.keys() if key not in hparam_dict_yaml]

        for key, value in hparam_dict.items():
            value = value["default"] if isinstance(value, dict) else value
            if (key in hparam_dict_yaml) and (key not in raw_args_cmd):
                if isinstance(hparam_dict_yaml[key], dict):
                    if "default" in hparam_dict_yaml[key]:
                        setattr(args, key, hparam_dict_yaml[key]["default"])
                    else:
                        setattr(args, key, hparam_dict_yaml[key])
                else:
                    setattr(args, key, hparam_dict_yaml[key])
    else:
        args.yaml_file = str(yaml_path)

    return args


def load_yaml(yaml_path):
    """
    Returns a dict from a yaml file

    :param yaml_path: Path of the yaml file.
    :type yaml_path: str

    """
    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.Loader)
    return yaml_dict

def args_to_yaml(args, yaml_path):
    """
    Returns a yaml file.

    Save the set of arguments (--key = "value") in a yaml file.

    :param args: Arguments saved in the Namespace list
    :type args: argparse.Namespace.
    :param yaml_path: Path of the yaml file.
    :type yaml_path: str
    """
    kwargs_dict = {kwarg[0]: kwarg[1] for kwarg in args._get_kwargs()}
    with open(yaml_path, 'w') as stream:
        yaml.dump(kwargs_dict, stream)

def args_to_dict(args):
    """Creates a dict from (nested) argparse.Namespace objects

    :param args: parsed args
    :type args: argparse.Namespace
    :returns: args as dict
    :rtype: dict
    """

    d = vars(args).copy()
    for k, v in d.items():
        if isinstance(v, argparse.Namespace):
            d[k] = args_to_dict(v)

    return d

def dict_to_yaml(data, yaml_path):
    """
    Returns a yaml file.

    Save dictionary in a yaml file.

    :param data: Dictionary
    :type data: dict().
    :param yaml_path: Path of the yaml file.
    :type yaml_path: str
    """
    kwargs_dict = {kwarg[0]: kwarg[1] for kwarg in data.items()}
    with open(yaml_path, 'w') as stream:
        yaml.dump(kwargs_dict, stream, default_flow_style=False)

# Context manager that copies stdout and any exceptions to a log file

class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
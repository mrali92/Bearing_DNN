"""
Created on 15.07.20
@author :ali
"""
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, BatchSampler, SubsetRandomSampler, SequentialSampler, DataLoader, TensorDataset

from ml.Bearing import Bearing_Dataset


def normalize(data, norm_type="std", axis=None, keepdims=True, eps=10 ** -5, norm_pars=None, clip=True):
    if norm_pars is not None:
        if norm_pars["norm_type"] == "std":
            data_norm = (data - norm_pars["mean"]) / norm_pars["std"]
        elif norm_pars["norm_type"] == "minmax":
            data_norm = (data - norm_pars["min"]) / (norm_pars["max"] - norm_pars["min"])
            if clip:
                data_norm[data_norm < 0.] = 0.
                data_norm[data_norm > 1.] = 1.
        elif norm_pars["norm_type"] == "symmetric":
            data_norm = data / norm_pars["c"]
            if clip:
                data_norm[data_norm < -1.] = -1.
                data_norm[data_norm > 1.] = 1.
        else:
            raise RuntimeError("Unknown norm type in norm params, got {:}".format(norm_pars["norm_type"]))
    else:
        if norm_type == "std":
            data_std = data.std(axis=axis, keepdims=keepdims)
            data_std[data_std < eps] = eps
            data_mean = data.mean(axis=axis, keepdims=keepdims)
            data_norm = (data - data_mean) / data_std
            norm_pars = dict(mean=data_mean, std=data_std, norm_type=norm_type, axis=axis, keepdims=keepdims)

        elif norm_type == "minmax":
            data_min = data.min(axis=axis, keepdims=keepdims)
            data_max = data.max(axis=axis, keepdims=keepdims)
            data_max[data_max == data_min] = data_max[data_max == data_min] + eps
            data_norm = (data - data_min) / (data_max - data_min)
            norm_pars = dict(min=data_min, max=data_max, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type == "symmetric":
            data_min = data.min(axis=axis, keepdims=keepdims)
            data_max = data.max(axis=axis, keepdims=keepdims)

            c = abs(data_max) if abs(data_max) >= abs(data_min) else abs(data_min)
            data_norm = data / c
            norm_pars = dict(c=c, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type == "asymmetric":
            data_min = data.min(axis=axis, keepdims=keepdims)
            data_max = data.max(axis=axis, keepdims=keepdims)

            c = abs(data_max) if abs(data_max) >= abs(data_min) else abs(data_min)
            data_norm = data.copy()
            data_norm[data_norm > 0] = data_norm[data_norm > 0] / abs(data_max)
            data_norm[data_norm < 0] = data_norm[data_norm < 0] / abs(data_min)
            norm_pars = dict(c_max=abs(data_max), c_min=abs(data_min), norm_type=norm_type, axis=axis,
                             keepdims=keepdims)
        else:
            data_norm, norm_pars = None, None
            ValueError("Unknown norm_type. Only 'std' or 'minmax' are supported.")
    return data_norm, norm_pars


def unormalize(data_norm, norm_pars):
    if norm_pars["norm_type"] == "std":
        data = data_norm * norm_pars["std"] + norm_pars["mean"]
    elif norm_pars["norm_type"] == "minmax":
        data = data_norm * (norm_pars["max"] - norm_pars["min"]) + norm_pars["min"]
    elif norm_pars["norm_type"] == "symmetric":
        data = data_norm * norm_pars["c"]
    return data


class Data(Dataset):
    def __init__(self, inputs, targets, transform=None):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs)
        if not isinstance(targets, torch.Tensor):
            targets = torch.Tensor(targets)

        assert inputs.shape[0] == targets.shape[0]

        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        sample = [self.inputs[idx, :], self.targets[idx, :]]
        if self.transform:
            sample = self.transform(sample)
        return sample[0], sample[1]


class GpuDataLoader:
    def __init__(self, data, sampler, batch_size=1024):
        self._data = data
        self._sampler = sampler
        self._batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        self.sample_iter = iter(self._batch_sampler)

    def __next__(self):
        indices = next(self.sample_iter)  # may raise StopIteration
        batch = self._data[indices]
        return batch

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        self.sample_iter = iter(self._batch_sampler)
        return self


def create_data_loaders(data, train_dev_test_split=[0.8, 0.2, 0.], batch_size=1024, pin_memory=False, num_workers=0):
    """

    Parameters
    ----------
    data
    train_dev_test_split
    batch_size
    pin_memory
    num_workers

    Returns
    -------

    """
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    if len(train_dev_test_split) == 2:
        train_dev_test_split = train_dev_test_split + [0.0]

    split_train_dev = int(np.round(train_dev_test_split[0] * len(data)))
    split_dev_test = split_train_dev + int(np.round(train_dev_test_split[1] * len(data)))

    train_indices = indices[:split_train_dev]
    dev_indices = indices[split_train_dev:split_dev_test]
    test_indices = indices[split_dev_test:]

    if batch_size == 0:
        batch_size = len(train_indices)
    else:
        batch_size = min(batch_size, len(train_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    dev_sampler = SequentialSampler(dev_indices)
    test_sampler = SequentialSampler(test_indices)
    if data.tensors[0].is_cuda:
        train_loader = GpuDataLoader(data, train_sampler, batch_size=batch_size)
        dev_loader = GpuDataLoader(data, dev_sampler, batch_size=len(dev_indices))
    else:
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                                  pin_memory=pin_memory)

    if len(dev_indices) > 0:
        dev_loader = DataLoader(data, batch_size=len(dev_indices), sampler=dev_sampler, num_workers=num_workers,
                                pin_memory=pin_memory)
    else :
        dev_loader = []

    if len(test_indices) > 0:
        test_loader = DataLoader(data, batch_size=len(test_indices), sampler=test_sampler, num_workers=num_workers,
                                 pin_memory=pin_memory)
    else:
        test_loader = []

    return train_loader, dev_loader, test_loader, [len(train_indices), len(dev_indices), len(test_indices)]


def prepare_dataset(inputs, targets, train_dev_test_split, batch_size):
    data = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets))
    train_loader, dev_loader, test_loader, num_samples = \
        create_data_loaders(data, train_dev_test_split=train_dev_test_split, batch_size=batch_size)

    return data, train_loader, dev_loader, test_loader, num_samples


def _prepare_batch(batch, device=None, non_blocking=False):
    """Factory function to prepare batch for training: pass to a device with options
    """
    x, y = batch

    return (x.to(device=device, non_blocking=non_blocking),
            y.to(device=device, non_blocking=non_blocking))


def _prepare_batch_onehot(batch, class_type=1, device=None, non_blocking=False):
    """Factory function to prepare batch for training. convert the labels to one hot coding.
    """
    x, y = batch
    if class_type == 1 or class_type == 2:
        y_onehot = torch.FloatTensor(y.shape[-1], 3)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)

    return (x.to(device=device, non_blocking=non_blocking),
            y_onehot.to(device=device, non_blocking=non_blocking, dtype=torch.long))


def load_bearing_data_cnn(class_type, signal_type, filter, bearings_name, num_measurement, segment_size, norm_type):
    le = LabelEncoder()

    if signal_type == "raw_vibration":
        bearing_data = Bearing_Dataset(bearing_names=bearings_name,
                                       mesurement_indices=list(range(num_measurement)),
                                       slice_index=250000,
                                       vibration_data=True,
                                       spectrum_data=False,
                                       filter=filter,
                                       type=class_type)

    if signal_type == "spectrum":
        bearing_data = Bearing_Dataset(bearing_names=bearings_name,
                                       mesurement_indices=list(range(num_measurement)),
                                       slice_index=125000,
                                       vibration_data=False,
                                       spectrum_data=True,
                                       filter=filter,
                                       type=class_type)

    inputs, targets = bearing_data.data

    # reshape data (num_bearing, num_channel, -1)
    inputs = inputs.reshape(len(bearings_name), -1)

    # normalize all data

    inputs_norm, x_norm_params = normalize(inputs, norm_type=norm_type, axis=1)  # normalization over axis 1

    # split data into small segments
    num_seg = inputs_norm.shape[-1] // segment_size
    if isinstance(num_seg, int):
        x = np.array(np.array_split(inputs_norm, num_seg, axis=-1))

    x = np.swapaxes(x, 0, 1)

    # add labels
    targets = le.fit_transform(targets)
    y = np.array([np.repeat(targets[i], num_seg) for i in range(len(bearings_name))]).flatten()

    return (x.reshape(-1, 1, segment_size), y, x_norm_params)


def load_bearing_data(class_type, signal_type, filter, bearings_name, num_measurement, segment_size, norm_type):
    le = LabelEncoder()

    if signal_type == "raw_vibration":
        bearing_data = Bearing_Dataset(bearing_names=bearings_name,
                                       mesurement_indices=list(range(num_measurement)),
                                       slice_index=250000,
                                       vibration_data=True,
                                       spectrum_data=False,
                                       filter=filter,
                                       type=class_type)

    if signal_type == "spectrum":
        bearing_data = Bearing_Dataset(bearing_names=bearings_name,
                                       mesurement_indices=list(range(num_measurement)),
                                       slice_index=125000,
                                       vibration_data=False,
                                       spectrum_data=True,
                                       filter=filter,
                                       type=class_type)

    inputs, targets = bearing_data.data

    # reshape data (num_bearing, num_channel, -1)
    inputs = inputs.reshape(len(bearings_name), -1)

    # normalize all data

    inputs_norm, x_norm_params = normalize(inputs, norm_type=norm_type, axis=1)  # normalization over axis 1

    # split data into small segments
    num_seg = inputs_norm.shape[-1] // segment_size
    if isinstance(num_seg, int):
        x = np.array(np.array_split(inputs_norm, num_seg, axis=-1))

    x = np.swapaxes(x, 0, 1)

    # add labels
    targets = le.fit_transform(targets)
    y = np.array([np.repeat(targets[i], num_seg) for i in range(len(bearings_name))]).flatten()

    return (x.reshape(-1, segment_size), y, x_norm_params)
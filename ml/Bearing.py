"""
Created on 24.06.20
@author :ali
"""

from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np

from ml.utils.feature_extraction import mean_feature, peaks_features
from ml.utils.fft_transformation import spektrum, butter_bandpass_filter, freq_IR, freq_OR

BEARING_DATA_PATH = Path(__file__).parent.parent / "struct_data"

z = 8  # Anzahl der rollenden Elemente im Lager
d = 6.75  # mm  Rollkörper Durchmesser
D = 33.1  # mm  Käfig Durchmesser
alpha = 0  # ° Druckwinkel


class Bearing(object):
    def __init__(self, bearing_name, mesurement_indices, sampling_frequency=64e3, filter="Bandpass",
                 fft_transformation=False, feature_space_params=None):

        self.bearing_name = bearing_name
        self.mesurement_indices = mesurement_indices
        self.vibration_signal = {}
        self.fs = sampling_frequency
        self.fft_transformation = fft_transformation
        self.feature_space_param = feature_space_params
        self.labels = {}
        self.group_measurements_by_operation()

        if self.fft_transformation:
            self.pre_signal = {}
            self.filtering(operation_name="N15_M07_F04", filter=filter)
            self.filtering(operation_name="N09_M07_F10", filter=filter)
            self.filtering(operation_name="N15_M01_F10", filter=filter)
            self.filtering(operation_name="N15_M07_F10", filter=filter)

            self.fft_transform(operation_name="N15_M07_F04")
            self.fft_transform(operation_name="N09_M07_F10")
            self.fft_transform(operation_name="N15_M01_F10")
            self.fft_transform(operation_name="N15_M07_F10")

        if self.feature_space_param is not None:
            self.features_spaces = {}
            self.create_features_spaces(operation_name="N15_M07_F04")
            self.create_features_spaces(operation_name="N09_M07_F10")
            self.create_features_spaces(operation_name="N15_M01_F10")
            self.create_features_spaces(operation_name="N15_M07_F10")

    def check_is_damage(self):
        pass

    def reshape_struct_data(self, data):
        res = []
        for exp in data:
            arr = []
            for vals in exp:
                if vals.shape[-1] > 1:
                    arr.append(vals.squeeze())
                else:
                    arr.append(vals.squeeze())
            res.append(arr)
        return res

    def remove_field_name(self, a, *fieldnames_to_remove):
        return a[[name for name in a.dtype.names if name not in fieldnames_to_remove]]

    def load_mat_exp_raw(self):
        pattern = self.bearing_name + "*.mat"
        for file in BEARING_DATA_PATH.glob(pattern):
            data = sio.loadmat(str(file), mat_dtype=True)["new_data_struct"].squeeze()
            self.labels["name"] = self.bearing_name
            self.labels["is_damage"] = data["damage_type"][0][0]
            self.labels["ir_damage"] = data["ir_damage"][0][0][0]
            self.labels["or_damage"] = data["or_damage"][0][0][0]
            self.labels["damage_extent"] = data["damage_extent"][0][0][0]
            data = self.remove_field_name(data,
                                          "bearing_num", "damage_type", "ir_damage", "or_damage", "damage_extent")

        raw_data = self.reshape_struct_data(data)
        return raw_data

    def group_measurements_by_operation(self):
        raw_data = self.load_mat_exp_raw()
        keys = ['rpm', 'torque', 'radial_force', 'mesurement', 'vibration_data']
        df = pd.DataFrame(raw_data, columns=keys)
        self.set_operations_values("N15_M07_F04")
        self.vibration_signal["N15_M07_F04"]["freq_IR_theoretic"] = freq_IR(self.vibration_signal["N15_M07_F04"]["rpm"],
                                                                            d, D, alpha, z)
        self.vibration_signal["N15_M07_F04"]["freq_OR_theoretic"] = freq_OR(self.vibration_signal["N15_M07_F04"]["rpm"],
                                                                            d, D, alpha, z)
        grp_1 = df[df["radial_force"] == 400].sort_values(["mesurement"])
        self.vibration_signal["N15_M07_F04"]["vibration_data"] = [grp_1["vibration_data"].to_numpy()[id] for id in
                                                                  self.mesurement_indices]
        # self.vibration_signal["N15_M07_F04"]["vibration_data"] = grp_1["vibration_data"].to_numpy()

        self.set_operations_values("N15_M01_F10")
        self.vibration_signal["N15_M01_F10"]["freq_IR_theoretic"] = freq_IR(self.vibration_signal["N15_M01_F10"]["rpm"],
                                                                            d, D, alpha, z)
        self.vibration_signal["N15_M01_F10"]["freq_OR_theoretic"] = freq_OR(self.vibration_signal["N15_M01_F10"]["rpm"],
                                                                            d, D, alpha, z)
        grp_2 = df[df["torque"] == 0.1].sort_values(["mesurement"])
        self.vibration_signal["N15_M01_F10"]["vibration_data"] = [grp_2["vibration_data"].to_numpy()[id] for id in
                                                                  self.mesurement_indices]

        self.set_operations_values("N09_M07_F10")
        self.vibration_signal["N09_M07_F10"]["freq_IR_theoretic"] = freq_IR(self.vibration_signal["N09_M07_F10"]["rpm"],
                                                                            d, D, alpha, z)
        self.vibration_signal["N09_M07_F10"]["freq_OR_theoretic"] = freq_OR(self.vibration_signal["N09_M07_F10"]["rpm"],
                                                                            d, D, alpha, z)
        grp_3 = df[df["rpm"] == 900].sort_values(["mesurement"])
        self.vibration_signal["N09_M07_F10"]["vibration_data"] = [grp_3["vibration_data"].to_numpy()[id] for id in
                                                                  self.mesurement_indices]

        self.set_operations_values("N15_M07_F10")
        self.vibration_signal["N15_M07_F10"]["freq_IR_theoretic"] = freq_IR(self.vibration_signal["N15_M07_F10"]["rpm"],
                                                                            d, D, alpha, z)
        self.vibration_signal["N15_M07_F10"]["freq_OR_theoretic"] = freq_OR(self.vibration_signal["N15_M07_F10"]["rpm"],
                                                                            d, D, alpha, z)
        tmp = df[df["radial_force"] == 1000]
        tmp_1 = tmp[tmp["torque"] == 0.7]
        grp_4 = tmp_1[tmp_1["rpm"] == 1500].sort_values(["mesurement"])
        self.vibration_signal["N15_M07_F10"]["vibration_data"] = [grp_4["vibration_data"].to_numpy()[id] for id in
                                                                  self.mesurement_indices]

    def set_operations_values(self, name):
        x = self.vibration_signal[name] = {}
        params = name.split("_")
        for p in params:
            if p == "N15":
                x["rpm"] = 1500
            elif p == "N09":
                x["rpm"] = 900
            if p == "M01":
                x["torque"] = 0.1
            elif p == "M07":
                x["torque"] = 0.7
            if p == "F10":
                x["radial_force"] = 1000
            elif p == "F04":
                x["radial_force"] = 400

    def calculate_cut_intervall(self, data):
        f_low = (data["freq_OR_theoretic"] - (data["rpm"] * z * d / D))
        f_high = (data["freq_IR_theoretic"] + (data["rpm"] * z * d / D))
        return f_low, f_high

    def filtering(self, operation_name, filter):
        if filter == "Bandpass":
            x = self.vibration_signal[operation_name]
            f_low, f_high = self.calculate_cut_intervall(x)
            self.pre_signal[operation_name] = {}
            self.pre_signal[operation_name]["fmin"] = f_low
            self.pre_signal[operation_name]["fmax"] = f_high
            # TODO fix nan values in filter signal
            selected_signales = [x["vibration_data"][id] for id in self.mesurement_indices]
            self.pre_signal[operation_name]["filt_signal"] = \
                [butter_bandpass_filter(s, f_low, f_high, self.fs, order=5) for s in selected_signales]
        if filter == "none":
            x = self.vibration_signal[operation_name]
            self.pre_signal[operation_name] = {}
            self.pre_signal[operation_name]["filt_signal"] = \
                [x["vibration_data"][id] for id in self.mesurement_indices]
        # else:
        #     raise ValueError('Bandpass filter is only available')

    def fft_transform(self, operation_name):  # TODO fft on unfilterd signal
        self.pre_signal[operation_name]["frequencies"] = []
        self.pre_signal[operation_name]["power_spectrum"] = []
        for sig in self.pre_signal[operation_name]["filt_signal"]:
            freq, power_spec = spektrum(sig, self.fs)
            power_spec = power_spec / sum(power_spec)
            self.pre_signal[operation_name]["frequencies"].append(freq)
            self.pre_signal[operation_name]["power_spectrum"].append(power_spec)

    def create_features_spaces(self, operation_name):
        if isinstance(self.feature_space_param, dict):
            self.features_spaces[operation_name] = {}

            for feature_name, vals in self.feature_space_param.items():
                if feature_name == "mean_lib":
                    self.features_spaces[operation_name][feature_name] = []
                    fmin = self.pre_signal[operation_name]["fmin"]
                    fmax = self.pre_signal[operation_name]["fmax"]
                    teiler = vals["teiler"]
                    for spectrum in self.pre_signal[operation_name]["power_spectrum"]:
                        pts = mean_feature(spectrum, self.fs, fmin, fmax, teiler)
                        self.features_spaces[operation_name][feature_name].append(pts)

                if feature_name == "peak_lib":
                    threshold = self.feature_space_param[feature_name]["threshold"]
                    height = self.feature_space_param[feature_name]["height"]
                    distance = self.feature_space_param[feature_name]["distance"]
                    rel_height = self.feature_space_param[feature_name]["rel_height"]
                    num_peaks = self.feature_space_param[feature_name]["num_peaks"]

                    self.features_spaces[operation_name] = {}
                    self.features_spaces[operation_name]["peaks_freq"] = []
                    self.features_spaces[operation_name]["peaks_idx"] = []
                    self.features_spaces[operation_name]["peaks_widths"] = []
                    for spectrum in self.pre_signal[operation_name]["power_spectrum"]:
                        peaks_freq, peaks_idx, peaks_widths = peaks_features(spectrum, threshold, height,
                                                                             distance, rel_height, num_peaks)

                        self.features_spaces[operation_name]["peaks_freq"].append(peaks_freq)
                        self.features_spaces[operation_name]["peaks_idx"].append(peaks_idx)
                        self.features_spaces[operation_name]["peaks_widths"].append(peaks_widths)

    def save_feature_lib(self):
        pass

def create_class_label(dict_label, num_classes, label_type, label_shape):

    if "accerlerated" in dict_label["is_damage"] or\
       "healthy" in dict_label["is_damage"] or \
       "artificial" in dict_label["is_damage"] :
        label = ""
        if dict_label["or_damage"] and dict_label["ir_damage"]:
            label = label + "2"
        if dict_label["or_damage"]:
            label = label + "1"
        if dict_label["ir_damage"]:
            label = label + "0"
        label = label + str(dict_label["damage_extent"])
    #

    # if label_type == "one_hot_vector":
    #     res = np.zeros(num_classes)
    #     if label_type == "single_label":
    #         pass

    return label
def map_labels(dict_label, class_type):
    # label[0] kind of damage and label[1] location of damage and label[2] extend of damage

    if class_type == 1:
        if "healthy" in dict_label["is_damage"]:
            label = 0
        elif "artificial" in dict_label["is_damage"] or "accerlerated" in dict_label["is_damage"]:
            if dict_label["ir_damage"] == 1.0:
                label = 1
            elif dict_label["or_damage"] == 1.0:
                label = 2
            elif dict_label["or_damage"] == 1.0 and dict_label["ir_damage"] == 1.0 :
                label = 3
            else:
                raise ValueError ("bearing name does not belong to class type {}".format(class_type))


    elif class_type == 2:
        if "healthy" in dict_label["is_damage"]:
            label = 0
        elif "accerlerated" in dict_label["is_damage"]:
            if dict_label["ir_damage"] == 1.0 and dict_label["or_damage"] == 0.0:
                label = 1
            elif dict_label["or_damage"] == 1.0 and dict_label["ir_damage"] == 0.0:
                label = 2
            elif dict_label["or_damage"] == 1.0 and dict_label["ir_damage"] == 1.0 :
                label = 3
            else:
                raise ValueError ("bearing name does not belong to class type {}".format(class_type))

    elif class_type == 5:
        if "healthy" in dict_label["is_damage"]:
            label = 0
        elif "artificial" in dict_label["is_damage"] or "accerlerated" in dict_label["is_damage"]:
            if dict_label["ir_damage"] == 1.0 and dict_label["or_damage"] == 0.0:
                if dict_label["damage_extent"] == 1.0:
                    label = 1
                elif dict_label["damage_extent"] == 2.0:
                    label = 2
            elif dict_label["or_damage"] == 1.0 and dict_label["ir_damage"] == 0.0:
                if dict_label["damage_extent"] == 1.0:
                    label = 3
                elif dict_label["damage_extent"] == 2.0:
                    label = 4
            elif dict_label["or_damage"] == 1.0 and dict_label["ir_damage"] == 1.0 :
                label = 5
            else:
                raise ValueError ("bearing name does not belong to class type {}".format(class_type))


    return label


class Bearing_Dataset():
    def __init__(self, bearing_names,
                 mesurement_indices,
                 type,
                 slice_index,
                 vibration_data=False,
                 spectrum_data=False,
                 feature_space_params=None,
                 feature_spaces=None,
                 filter="none",
                 normalization="std"):

        self.inputs = []
        self.targets = []
        self.bearing_names = bearing_names
        self.type = type
        self.bearing_objs = []
        self.mesurement_indices = mesurement_indices
        self.vibration_data = vibration_data
        self.spectrum_data = spectrum_data
        self.feature_space_params = feature_space_params
        self.feature_spaces = feature_spaces
        self.filter = filter
        self.normalization = normalization

        if self.vibration_data:
            for bearing in self.bearing_names:
                bearing = Bearing(bearing, self.mesurement_indices, filter=self.filter,
                                  fft_transformation=self.spectrum_data,
                                  feature_space_params=self.feature_space_params)
                self.bearing_objs.append(bearing)

                tmp = []
                for k, v in bearing.vibration_signal.items():
                    for sig in v["vibration_data"]:
                        tmp.append(sig[:slice_index])  # to fix because signals have not the same length
                self.inputs.append(tmp)

                self.targets.append(map_labels(dict_label=bearing.labels, class_type=self.type))

            # sig_shape = np.array(
            #     [[[len(z) for z in v["vibration_data"]] for _, v in bearing.vibration_signal.items()] for bearing in
            #      self.bearing_objs]).flatten()
        if self.spectrum_data:
            for bearing in self.bearing_names:
                bearing = Bearing(bearing, self.mesurement_indices, filter=self.filter,
                                  fft_transformation=self.spectrum_data,
                                  feature_space_params=self.feature_space_params)
                self.bearing_objs.append(bearing)

                tmp = []
                for k, v in bearing.pre_signal.items():
                    for sig in v["power_spectrum"]:
                        tmp.append(sig[:slice_index])  # to fix because signals have not the same length
                self.inputs.append(tmp)

                self.targets.append(map_labels(dict_label=bearing.labels, class_type=self.type))

                # sig_shape = np.array(
                #     [[[len(z) for z in v["power_spectrum"]] for _, v in bearing.pre_signal.items()] for bearing in
                #      self.bearing_objs]).flatten()

        if self.feature_spaces is not None:
            self.features_libs = []
            self.features_libs.append(self.get_feature_space(bearing))

        # self.inputs = [[tt[:np.amin(sig_shape)] for tt in t] for t in self.inputs]
        self.data = (np.array(self.inputs), np.array(self.targets))

    def get_feature_space(self, bearing):

        feature_lib = {}
        feature_lib["data"] = {}
        for selected_feature in self.feature_spaces:
            feature_lib["data"][selected_feature] = {}
            for operation_name, features in bearing.features_spaces.items():
                feature_lib["data"][selected_feature][operation_name] = features[selected_feature]

        feature_lib["label"] = bearing.labels
        return feature_lib





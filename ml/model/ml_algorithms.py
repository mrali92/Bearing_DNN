#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm

from ml.Bearing import Bearing_Dataset


def meanFeature(power,fs,fmin,fmax,teiler):
    freqTeil = int((fmax - fmin)/teiler)
    pTeil = []
    for i in range(0,teiler):
        anfang=int(len(power)/(fs/2)*(fmin+i*freqTeil))
        avg = 0.0
        for j in range(0,freqTeil):
            avg = avg + power[anfang+j]       
        pTeil.append(avg)
    return pTeil


def kNN_learn(features, labels, k):
    '''
    

    Parameters
    ----------
    data : features as array
    classes : data containing the labels of the features as string
    k : count of the nearest neighbors for classification

    Returns
    -------
    classkNN : classificator

    '''
    
    
    # Erstellung des Klassifikator
    class_kNN = KNeighborsClassifier(n_neighbors= k) #alogorithm kd_tree oder ball_tree?
    class_kNN.fit(features, labels)
    return class_kNN


def SVM_learn(features, labels):
    
    class_SVM = svm.SVC()
    class_SVM.fit(features, labels)
    return class_SVM

class Klassifikatoren():
    def __init__(self, class_type=1,
                 algorithm=None,
                 training_set=None,
                 test_set=None,
                 feature_spaces=["peaks_freq", "peaks_idx", "peaks_widths"],  # "mean_lib"],
                 feature_space_params=None,
                 mesurement_indices=None,
                 normalization=False,
                 plotting=False, 
                 k=10):

        self.class_type = class_type
        self.algorithm = algorithm
        self.feature_spaces = feature_spaces
        self.feature_space_params = feature_space_params
        self.mesurment_indices = mesurement_indices

        self.normalization = normalization
        self.kNN = None
        self.SVM = None
        
        if isinstance(training_set, list):
            training_set=Bearing_Dataset(bearing_names = trainig_set, feature_space_params = feature_space_params, mesurement_indices = mesurement_indices, feature_spaces =feature_spaces)
        if isinstance(test_set, list):
            test_set = Bearing_Dataset(bearing_names = test_set, feature_space_params = feature_space_params, mesurement_indices = mesurement_indices, feature_spaces =feature_spaces)
               
        if "kNN" in algorithm or "SVM" in self.alogorithm:
            assert training_set is not None
            self.train_feature_list, self.train_label_list = self.convert_to_list(training_set, self.class_type)
            train_label=self.convert_to_int(self.train_label_list,class_type)
            if self.test_set:
                self.test_feature_list, self.test_label_list = self.convert_to_list(test_set, self.class_type)
                
            if "kNN" in algorithm:
                self.kNN = kNN_learn(self.train_feature_list,train_label, k)
                if self.test_feature_list and self.test_label_list:
                    self.kNN_accuracy = self.classifier_accuracy(self.kNN)
                
            if "SVM" in algorithm():
                self.SVM = None
                if self.test_feature_list and self.test_label_list:
                    self.SVM_accuracy = self.classifier_accuracy(self.SVM)
                
        
            
#--------------------------functions---------------------------------------------------------
#%%
    def convert_to_list(self, dataset, class_type):
        
        """
        Converts the feature_lib of a Bearing_Dataset into a feature_list and label_list,
        usable for kNN and SVM

        Parameters
        ----------
        dataset : Bearing_Dataset
        
        class_type : 1 == artificial or 2 == real

        Returns
        -------
        feature_list : List with a feature per entry. Features can be a list too. 
        label_list : List with labels in string format

        """
        label_list=[]
        feature_list=[]
        
        #insert operating conditions
        operating_conditions = ["N15_M07_F04", "N09_M07_F10", "N15_M01_F10","N15_M07_F10"]
        if not isinstance(dataset, Bearing_Dataset):
            print("Wrong input. Need BearingDataset.")
            return 
        
        for feature_lib in dataset.features_libs:
            current_label=feature_lib["label"]
            label=None
            
            # label[0] kind of damage and label[1] location of damage and label[2] extend of damage       
                        
            if "accerlerated" in current_label.is_damage or "healthy" in current_label.is_damage and class_type == 2 :
                label=""
                if current_label.or_damage and current_label.ir_damage:
                    label=label+"2"
                if current_label.or_damage:
                    label=label+"1"
                if current_label.ir_damage:
                    label=label +"0"
                label = label + str(current_label.damage_extent)
                    
                features, labels = self.create_feature_list(feature_lib, operating_conditions, label)
                feature_list.extend(features)
                label_list.extend(labels)
                    
            if "artificial" in current_label.is_damage or "healthy" in current_label.is_damage and class_type == 1:
                label=""
                if current_label.or_damage and current_label.ir_damage:
                    label=label+"2"
                if current_label.or_damage:
                    label=label+"1"
                if current_label.ir_damage:
                    label=label +"0"
                label = label + str(current_label.damage_extent)
                
                features, labels = self.create_feature_list(feature_lib, operating_conditions, label)
                feature_list.extend(features)
                label_list.extend(labels)
                
            assert isinstance(label, str)

                    
        return feature_list, label_list
    
    def create_feature_list(self, feature_lib, operating_conditions, label):
        feature_list=[]
        label_list=[]
        for operating_condition in operating_conditions:
            label_list.append(label)
            tmp_list=[]
            current_dir=feature_lib["data"]
            for feature_space in self.feature_space:
                for idx in current_dir[feature_space][operating_condition]:
                    tmp_list.append(idx)
            feature_list.append(tmp_list)
        return feature_list, label_list
    
    def convert_to_int(self, label_list, class_type):
        
        assert isinstance(label_list, list)
        if class_type == 1 or class_type == 2:
            start=0
            end=1
        if class_type == 3:
            start=0
            end=2
            
            
        int_label_list=[]
        for label in label_list:
            assert isinstance(label, str)
            int_label_list.append(int(label[start:end]))
        return int_label_list
    
    def classifier_accuracy(self, classifier, testdata = None):
        """
        Generates the accuracy of a classifier made with sclearn

        Parameters
        ----------
        classifier : --
        testdata : if given, returns accuracy of new test data and overwrites attributes test_feature_list test_label_list
        Returns
        -------
        accuracy : --

        """
        assert isinstance(testdata, list)
        if testdata is not None:
            test_set = Bearing_Dataset(testdata, self.feature_space_params, self.mesurement_indices, self.feature_spaces)
            self.test_feature_list, self.test_label_list = self.convert_to_list(test_set, self.class_type)
        test_label=self.convert_to_int(self.test_label_list)
        
        pred_label_list = classifier.predict(self.test_feature_list)
        
        accuracy = metrics.accuracy_score(test_label, pred_label_list)
        print(accuracy)
        return accuracy
    
    
        
    
            
#%%
    



if __name__ == "__main__":
    feature_space_params = {"mean_lib": {"teiler": 1},
                            "peak_lib": {"num_peaks": 1,
                                         "threshold": -100,  # threshold of peaks
                                         "height": 0,  # height of peaks
                                         "distance": 5000,
                                         "rel_height": 0.999}}

    trainig_set = ["K002","KA01", "KI01"]
    test_set = ["K001", "KA22"]
    
    idx = list(range(0,10))
    klassifikator_1 = Klassifikatoren(algorithm="kNN SVM",class_type=1, training_set=trainig_set, test_set=test_set, mesurement_indices=idx, feature_space_params=feature_space_params)

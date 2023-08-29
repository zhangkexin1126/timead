"""
Thank to 
https://github.com/thuml/Anomaly-Transformer/blob/main/data_factory/data_loader.py

"""

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split, Dataset

import numpy as np
import random
import os
import pickle
import pandas as pd

# from ncad.ts import TimeSeries, TimeSeriesDataset
from tqdm import tqdm
from typing import Union, Iterable
from pathlib import PosixPath

from sklearn import preprocessing
import sktime
from sktime.datasets import load_from_tsfile


datarootpath = "/home/kexin/proj/data"


class SequencePreprocessing():
    def __init__(self, expected_length=600):
        '''

        :param expected_length: expected length of the sequence
        '''
        self.expected_length = expected_length

    def lengthwrapping(self, sequence):
        sequence_len = sequence.shape[0]
        if sequence_len > self.expected_length:
            return sequence[0:self.expected_length, :]
        elif sequence_len < self.expected_length:
            temp = sequence
            e_total = self.expected_length - sequence_len
            e_ratio = int(e_total/sequence_len)
            for i in range(e_ratio):
                sequence = np.concatenate((sequence, temp), axis=0)
            e_rest = self.expected_length - sequence.shape[0]
            rest = sequence[-e_rest-1:-1,:]
            seqeunce = np.concatenate((sequence, rest), axis=0)
            return seqeunce
        elif sequence_len == self.expected_length:
            return sequence

    def noisefilter(self, sequence):
        b, a = signal.butter(3, 0.3, 'lowpass')
        return signal.filtfilt(b, a, sequence, axis=0)

class DataPreprocessing():
    def __init__(self, mode='standard'):
        self.mode = mode

    def preprocess(self, X_train):
        if self.mode == 'maxmin':
            scaler = preprocessing.MinMaxScaler().fit(X_train)
            X = scaler.transform(X_train)
        elif self.mode == 'standard':
            scaler = preprocessing.StandardScaler().fit(X_train)
            X = scaler.transform(X_train)
        elif self.mode == "norm":
            scaler = preprocessing.Normalizer().fit(X_train)
            X = scaler.transform(X_train)
        else:
            X = X_train
        return X

def preprocess(df):
    """returns normalized and standardized data.
    """
    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()
    # normalize data
    df = preprocessing.StandardScaler().fit_transform(df)
    # MinMaxScaler, StandardScaler
    return df

def load_ad(dataname, validrate):
    if dataname == "SMD":
        prefix = os.path.join(datarootpath, "SMD")
        x_dim = 38

        # print(' @load dataset:', dataname)
        train_data = np.load(prefix + "/SMD_train.npy")
        train_label = np.zeros(len(train_data), dtype=int)
        test_data = np.load(prefix + "/SMD_test.npy")
        test_label = np.load(prefix + "/SMD_test_label.npy")
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

        ## split data into training and validation
        n = int(len(train_data) * validrate)
        train_x, valid_x = train_data[:-n], train_data[-n:]
        train_y, valid_y = train_label[:-n], train_label[-n:]
        test_x = test_data
        test_y = test_label
        # print("   @train_x and train_y shape:", train_x.shape, train_y.shape)
        # print("   @valid_x and valid_y shape:", valid_x.shape, valid_y.shape)
        # print("   @test_x and test_y shape:", test_x.shape, test_y.shape)

        return train_x, valid_x, test_x, test_y


    elif dataname == "MSL":
        prefix = os.path.join(datarootpath, "MSL")
        x_dim = 55

        # print(' @load dataset:', dataname)
        train_data = np.load(prefix + "/MSL_train.npy")
        train_label = np.zeros(len(train_data), dtype=int)
        test_data = np.load(prefix + "/MSL_test.npy")
        test_label = np.load(prefix + "/MSL_test_label.npy")
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

        ## split data into training and validation
        n = int(len(train_data) * validrate)
        train_x, valid_x = train_data[:-n], train_data[-n:]
        train_y, valid_y = train_label[:-n], train_label[-n:]
        test_x = test_data
        test_y = test_label
        # print("   @train_x and train_y shape:", train_x.shape, train_y.shape)
        # print("   @valid_x and valid_y shape:", valid_x.shape, valid_y.shape)
        # print("   @test_x and test_y shape:", test_x.shape, test_y.shape)

        return train_x, valid_x, test_x, test_y


    elif dataname == "SMAP":
        prefix = os.path.join(datarootpath, "SMAP")
        x_dim = 25

        # print(' @load dataset:', dataname)
        train_data = np.load(prefix + "/SMAP_train.npy")
        train_label = np.zeros(len(train_data), dtype=int)
        test_data = np.load(prefix + "/SMAP_test.npy")
        test_label = np.load(prefix + "/SMAP_test_label.npy")
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

        ## split data into training and validation
        n = int(len(train_data) * validrate)
        train_x, valid_x = train_data[:-n], train_data[-n:]
        train_y, valid_y = train_label[:-n], train_label[-n:]
        test_x = test_data
        test_y = test_label
        # print("   @train_x and train_y shape:", train_x.shape, train_y.shape)
        # print("   @valid_x and valid_y shape:", valid_x.shape, valid_y.shape)
        # print("   @test_x and test_y shape:", test_x.shape, test_y.shape)

        return train_x, valid_x, test_x, test_y

    elif dataname == "SWAT":
        pass

    elif dataname == "PSM":
        prefix = os.path.join(datarootpath, "PSM")
        x_dim = 25

        # print('   @load dataset:', dataname)
        train_data = pd.read_csv(prefix + "/train.csv")
        train_data = train_data.values[:, 1:]
        train_data = np.nan_to_num(train_data)
        train_label = np.zeros(len(train_data), dtype=int)
        test_data = pd.read_csv(prefix + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_label = pd.read_csv(prefix + '/test_label.csv').values[:, 1:]
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

        ## split data into training and validation
        n = int(len(train_data) * validrate)
        train_x, valid_x = train_data[:-n], train_data[-n:]
        train_y, valid_y = train_label[:-n], train_label[-n:]
        test_x = test_data
        test_y = test_label
        # print("   @train_x and train_y shape:", train_x.shape, train_y.shape)
        # print("   @valid_x and valid_y shape:", valid_x.shape, valid_y.shape)
        # print("   @test_x and test_y shape:", test_x.shape, test_y.shape)
        
        return train_x, valid_x, test_x, test_y

    elif dataname == "YAHOO":
        pass

    elif dataname == "KPI":
        pass

    else:
        pass

class Dataset_AnomalyDetection(Dataset):
    def __init__(self, dataname, validrate, win_size, step, mode="train", config=None):
        
        self.dataname = dataname
        self.win_size = win_size
        self.validrate = validrate
        self.step = step
        self.mode = mode
        self.config = config
        self.num_classes = 2
        self.train, self.val, self.test, self.test_labels = load_ad(self.dataname, self.validrate)
        
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]) 


"""
    Load UCR
"""
"""
    ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
     'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction',
     'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Earthquakes',
     'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
     'Fish', 'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
     'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
     'Mallat', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
     'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
     'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
     'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim',
     'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
     'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
     'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
     'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga', 'ACSF1',
     'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'BME', 'Chinatown', 'Crop', 'DodgerLoopDay',
     'DodgerLoopGame', 'DodgerLoopWeekend', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'EthanolLevel',
     'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
     'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
     'HouseTwenty', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'MelbournePedestrian', 'MixedShapesRegularTrain',
     'MixedShapesSmallTrain', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID',
     'PowerCons', 'Rock', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
     'SmoothSubspace', 'UMD']
"""
def loaducr(dataname):
    DATA_PATH = os.path.join(datarootpath, "UCR/UCRArchive_2018")
    ucrlist = os.listdir(DATA_PATH)
    if dataname not in ucrlist:
        raise ValueError('dataset not found')
    
    train_file = os.path.join(DATA_PATH, dataname, dataname + "_TRAIN.tsv")
    test_file = os.path.join(DATA_PATH, dataname, dataname + "_TEST.tsv")
    train_array = np.array(pd.read_csv(train_file, sep='\t', header=None))
    test_array = np.array(pd.read_csv(test_file, sep='\t', header=None))
    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = np.expand_dims(train_array[:, 1:], 2).astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = np.expand_dims(test_array[:, 1:], 2).astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    if dataname not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2'
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train, train_labels, test, test_labels
    
    # Post-publication note:
    # Using the testing set to normalize might bias the learned network,
    # but with a limited impact on the reported results on few datasets.
    # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
    mean = np.nanmean(np.concatenate([train, test]))
    var = np.nanvar(np.concatenate([train, test]))
    train = (train - mean) / math.sqrt(var)
    test = (test - mean) / math.sqrt(var)

    return train, train_labels, test, test_labels

class Dataset_UCR(Dataset):
    def __init__(self, dataname = "Beef", flag='train', config=None):
        # init
        self.dataname = dataname
        self.flag = flag
        self.config = config

        self.dp = DataPreprocessing(mode="standard")
        # 'standard'  # or 'standard' 'maxmin' 'norm'

        self.x, self.y = self.__read_data()
        self.ts_num, self.ts_len, self.ts_dim = self.x.shape
        self.num_classes = self.nc

    def __read_data(self):
        if self.flag == 'train':
            x, y, *_ = loaducr(self.dataname)
        elif self.flag == 'test':
            *_, x, y = loaducr(self.dataname)
        else:
            raise ValueError('Unknown flag, TRAIN or TEST is required')
        self.nc = len(np.unique(y))
        return x, y
    
    def __getitem__(self, idx):
        seq_x, seq_y = self.x[idx], self.y[idx]
        seq_x = self.dp.preprocess(seq_x) 
        return seq_x, np.array(seq_y)

    def __len__(self):
        return self.ts_num 


"""
    Load UEA
"""
"""
    ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
    'Cricket', 'DuckDuckGeese', 'ERing', 'EigenWorms', 'Epilepsy', 'EthanolConcentration',
    'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
    'InsectWingbeat', 'JapaneseVowels', 'LSST', 'Libras', 'MotorImagery', 'NATOPS', 'PEMS-SF',
    'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
    'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

    CharacterTrajectories
    SpokenArabicDigits

    DuckDuckGeese?

"""

def loaduea(dataname):

    if dataname in ["EigenWorms", "InsectWingbeat", "DuckDuckGeese", "FaceDetection", "Heartbeat", "LSST", "MotorImagery",
                    "PEMS-SF", "PenDigits", "PhonemeSpectra", "SpokenArabicDigits"]:
        
        train = np.load("/home/kexin/proj/data/UEA/Multivariate_ts/{}NPY/train.npy".format(dataname), allow_pickle=True)
        train_labels = np.load("/home/kexin/proj/data/UEA/Multivariate_ts/{}NPY/train_labels.npy".format(dataname), allow_pickle=True)
        test = np.load("/home/kexin/proj/data/UEA/Multivariate_ts/{}NPY/test.npy".format(dataname), allow_pickle=True)
        test_labels = np.load("/home/kexin/proj/data/UEA/Multivariate_ts/{}NPY/test_labels.npy".format(dataname), allow_pickle=True)
    
    else:
        DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "/home/kexin/proj/data/UEA/Multivariate_ts")

        trainpath = f"{dataname}/{dataname}_TRAIN.ts"
        train_x, train_y = load_from_tsfile(os.path.join(DATA_PATH, trainpath)) 
        train_x = train_x.applymap(lambda x: x.astype(np.float32))
        train_x = train_x.to_numpy()

        testpath = f"{dataname}/{dataname}_TEST.ts"
        test_x, test_y = load_from_tsfile(os.path.join(DATA_PATH, testpath))
        test_x = test_x.applymap(lambda x: x.astype(np.float32))
        test_x = test_x.to_numpy()

        train_size = len(train_x)
        test_size = len(test_x)
        ts_dim = train_x.shape[1]
        if dataname in ["CharacterTrajectories"]:
            ts_length = 150
        else:
            ts_length = len(train_x[0][0])

        train = np.empty((train_size, ts_length, ts_dim))
        test = np.empty((test_size, ts_length, ts_dim))
        train_labels = []
        test_labels = []

        # print(train_x.shape, train.shape)
        # print(test_x.shape, test.shape)

        dp = SequencePreprocessing(expected_length=ts_length)

        for i in range(train_size):
            train_labels.append(train_y[i])
            for j in range(ts_dim):
                seq = np.expand_dims(train_x[i][j], 1)
                seq = dp.lengthwrapping(seq).flatten()
                train[i, :, j] = seq

        for i in range(test_size):
            test_labels.append(test_y[i])
            for j in range(ts_dim):
                seq = np.expand_dims(test_x[i][j], 1)
                seq = dp.lengthwrapping(seq).flatten()
                test[i, :, j] = seq
        
        # Move the labels to {0, ..., L-1}
        labels = np.unique(train_labels)
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i
        train_labels = np.vectorize(transform.get)(train_labels)
        test_labels = np.vectorize(transform.get)(test_labels)

    return train, train_labels, test, test_labels

class Dataset_UEA(Dataset):
    def __init__(self, dataname = "Cricket", flag='train', config=None):
        # init
        self.dataname = dataname
        self.flag = flag
        self.config = config

        self.dp = DataPreprocessing(mode="standard")
        # 'standard'  # or 'standard' 'maxmin' 'norm'

        self.x, self.y = self.__read_data()
        self.ts_num, self.ts_len, self.ts_dim = self.x.shape
        self.num_classes = self.nc

    def __read_data(self):
        if self.flag == 'train':
            x, y, *_ = loaduea(self.dataname)
        elif self.flag == 'test':
            *_, x, y = loaduea(self.dataname)
        else:
            raise ValueError('Unknown flag, TRAIN or TEST is required')
        self.nc = len(np.unique(y))
        return x, y
    
    def __getitem__(self, idx):
        seq_x, seq_y = self.x[idx], self.y[idx]
        seq_x = self.dp.preprocess(seq_x)
        return seq_x, np.array(seq_y)

    def __len__(self):
        return self.ts_num 

ad_list = ["SMD", "MSL", "PSM", "SMAP", "SWAT"]

ucr_list = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
     'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction',
     'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Earthquakes',
     'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
     'Fish', 'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
     'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
     'Mallat', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
     'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
     'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
     'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim',
     'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
     'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
     'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
     'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga', 'ACSF1',
     'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'BME', 'Chinatown', 'Crop', 'DodgerLoopDay',
     'DodgerLoopGame', 'DodgerLoopWeekend', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'EthanolLevel',
     'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
     'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
     'HouseTwenty', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'MelbournePedestrian', 'MixedShapesRegularTrain',
     'MixedShapesSmallTrain', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID',
     'PowerCons', 'Rock', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
     'SmoothSubspace', 'UMD']

uea_list = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
    'Cricket', 'DuckDuckGeese', 'ERing', 'EigenWorms', 'Epilepsy', 'EthanolConcentration',
    'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
    'InsectWingbeat', 'JapaneseVowels', 'LSST', 'Libras', 'MotorImagery', 'NATOPS', 'PEMS-SF',
    'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
    'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']



from sklearn.preprocessing import StandardScaler
class NIPS_TS_WaterSegLoader(object):
    def __init__(self, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load("/home/kexin/proj/data/NIPS_TS_GECCO/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load("/home/kexin/proj/data/NIPS_TS_GECCO/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load("/home/kexin/proj/data/NIPS_TS_GECCO/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

        self.num_classes = 2

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]) 


class NIPS_TS_SwanSegLoader(object):
    def __init__(self, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load("/home/kexin/proj/data/NIPS_TS_Swan/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load("/home/kexin/proj/data/NIPS_TS_Swan/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load("/home/kexin/proj/data/NIPS_TS_Swan/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

        self.num_classes = 2

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]) 


def get_loader(dataname, validrate, win_size=100, step=100, mode='train', batch_size=256):
    if dataname in ad_list:
        ds = Dataset_AnomalyDetection(dataname, validrate,  win_size, step, mode=mode)
    elif dataname in ucr_list:
        ds = Dataset_UCR(dataname = dataname, flag=mode)
    elif dataname in uea_list:
        ds = Dataset_UEA(dataname = dataname, flag=mode)
    if dataname == 'NIPS_TS_Water':
        ds = NIPS_TS_WaterSegLoader(win_size, step, mode)
    elif dataname == 'NIPS_TS_Swan':
        ds = NIPS_TS_SwanSegLoader(win_size, step, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return ds, dl



if __name__ == "__main__":

    load_ad("SMAP", 0.1, do_preprocess=True)

    # loaducr("Beef")
    # dataname = "SpokenArabicDigits"
    # train, train_labels, test, test_labels = loaduea(dataname)

    ## Save Wrong Data
    
    # np.save("/home/kexin/proj/data/uea/Multivariate_ts/SpokenArabicDigitsNPY/train.npy", train)
    # np.save("/home/kexin/proj/data/uea/Multivariate_ts/SpokenArabicDigitsNPY/train_labels.npy", train_labels)
    # np.save("/home/kexin/proj/data/uea/Multivariate_ts/SpokenArabicDigitsNPY/test.npy", test)
    # np.save("/home/kexin/proj/data/uea/Multivariate_ts/SpokenArabicDigitsNPY/test_labels.npy", test_labels)

    # # np.load("file.npy", allow_pickle=True)

    # print(train.shape, train_labels.shape)
    # print(test.shape, test_labels.shape)
    

    # ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
    # 'Cricket', 'DuckDuckGeese', 'ERing', 'EigenWorms', 'Epilepsy', 'EthanolConcentration',
    # 'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
    # 'InsectWingbeat', 'JapaneseVowels', 'LSST', 'Libras', 'MotorImagery', 'NATOPS', 'PEMS-SF',
    # 'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
    # 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

    # DuckDuckGeese
    # InsectWingbeat
    
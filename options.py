import os
import argparse
import torch

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

class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Anomaly Detection / Classification Experiments')

        ## data prepare
        self.parser.add_argument('--dataname', type=str, default='SMD', help='dataset name [SMD, MSL, PSM, SMAP, SWAT] ["UCR*"] [UEA*]')
        self.parser.add_argument('--validrate', type=float, default=0.01, help='how many data are used to valid')
        

        # dataloader args
        self.parser.add_argument('--tsdim', type=int, default=25, help='dim of raw time series data')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch_size for training')
        self.parser.add_argument('--win_size', type=int, default=100, help='window size')
        self.parser.add_argument('--win_step', type=int, default=100, help='segment step')

        
        ## Model args
        self.parser.add_argument('--encoder_select', type=str, default='tcn', help='trans or tcn')


        ## Embedding args
        
        
        ### Encoder args
        self.parser.add_argument('--tsembeddim', type=int, default=256, help='Dim of hidden representations in embedding space for each time-step')
        self.parser.add_argument('--representation_dim', type=int, default=256, help='dim of final encoder save')

        ## TCN Net args
        self.parser.add_argument('--tcnindim', type=int, default=256, help='dim of each tcn block output, except final block')
        self.parser.add_argument('--tcnblockdim', type=int, default=256, help='dim of each tcn block output, except final block')
        self.parser.add_argument('--tcndepth', type=int, default=3, help='depth of TCN encoder')
        self.parser.add_argument('--tcnkernelsize', type=int, default=3, help='convolution kernel size')
        self.parser.add_argument('--tcndropout', type=float, default=0.1, help='dropout')
        self.parser.add_argument('--tcninputmode', type=str, default="blc", help='length first or dim first, blc / bcl')
        
        ## Transformer args
        self.parser.add_argument('--trans_in_dim', type=int, default=256, help='dim of each tcn block save, except final block')
        self.parser.add_argument('--trans_in_length', type=int, default=100, help='dim of each tcn block save, except final block')
        self.parser.add_argument('--trans_n_heads', type=int, default=8, help='number of heads')
        self.parser.add_argument('--trans_n_layers', type=int, default=3, help='number of layers')
        self.parser.add_argument('--trans_block_dim', type=int, default=128, help='dim of each trans block inside')

        ## DDPM args



        # loss args
        self.parser.add_argument('--temperature', type=float, default=0.07, help='Logits are divided by temperature before calculating the cross entropy')

        # Train&Value
        self.parser.add_argument("-epo", '--epoch', type=int, default=5, help='max iteration for training')
        self.parser.add_argument('--evalonly', action='store_true', help='whether to evaluate, if True, training process is not available')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument('--outputpath', type=str, default='save', help='[ ./save ] prefix path to save algorithm save')
        self.parser.add_argument('-evme', '--evalmethod', type=str, default='rec', help='rec/sim/recsim')

        # GPU settings
        self.parser.add_argument('--use_gpu', type=bool, default=True, help='whether use gpu')

        # Threshlod
        self.parser.add_argument('--global_thre', type=float, default=0.5, help='Global Threshold for AE')

        

        ####

    def parse(self):
        args = self.parser.parse_args()
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        if args.use_gpu:
            args.device = torch.device("cuda:0")
        else:
            args.device = torch.device("cpu")

        if not os.path.exists(args.outputpath):
            os.makedirs(args.outputpath)

        if args.dataname == "YAHOO":
            args.tsdim = 1
        elif args.dataname == "SMD":
            args.tsdim = 38
        elif args.dataname == "MSL":
            args.tsdim = 55
        elif args.dataname == "PSM":
            args.tsdim = 25
        elif args.dataname == "SMAP":
            args.tsdim = 25
        elif args.dataname == "SWAT":
            args.tsdim = 33333333
        else:
            args.tsdim = 0

        return args
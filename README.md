## Run
python -m main --epoch 15 --dataname SMD --evalmethod sim


## Dataname 
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

uea_list = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories','Cricket', "DuckDuckGeese", 'ERing', 'EigenWorms', 'Epilepsy', 'EthanolConcentration','FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
    'InsectWingbeat', 'JapaneseVowels', 'LSST', 'Libras', 'MotorImagery', 'NATOPS', 'PEMS-SF',
    'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
    'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']


## Apply GPU in CSC101 [Drop]
salloc -N 1 --cpus-per-task=4 -t 5:00:00 -p gpu-2 --gres=gpu:1
salloc -N 1 --cpus-per-task=4 -t 10:00:00 -p gpu-2 --gres=gpu:1
salloc -N 1 --cpus-per-task=8 -t 15:00:00 -p gpu-3 --gres=gpu:1
## ID & Password in CSC101
- ssh zhangkexin@10.12.218.211 -p 23422
- oxjwrGQx4bqLak*z
- scp -r -P 23422 zhangkexin@10.12.218.211:/home/zhangkexin/Workspace/data/uea/ /home/kexin/proj/data/UEA/
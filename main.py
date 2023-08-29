import warnings
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import time
import random
import torch
from options import Options
from src import tools
from src import dataloader
from src import model_TSAD, loss, optimizer, runner

import logging
logging.basicConfig(format='> %(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

tools.set_seed(960214)
args = Options().parse()

"""Load Data"""
print("------------------------------")
starttime = time.time()
trainds, traindl = dataloader.get_loader(dataname=args.dataname, validrate=args.validrate, win_size=args.win_size, step=args.win_step, batch_size=args.batch_size, mode='train')
testds, testdl = dataloader.get_loader(dataname=args.dataname, validrate=args.validrate, win_size=args.win_size, step=args.win_step, batch_size=args.batch_size, mode='test')
validds, validdl = None, None
endtime = time.time()
print('Loading Data Time: ', round(endtime - starttime, 2))

args.ts_length = trainds[0][0].shape[0]
args.tsdim = trainds[0][0].shape[1]
args.nclass = trainds.num_classes
trainsize = len(trainds)
testsize = len(testds)
args.trans_in_length = args.ts_length

print("------------------------------")
print(" @Data-Info: {}".format(args.dataname))
print(" @Device: {}".format(args.device))
print(" @Batchsize: {}".format(args.batch_size))
print(" @TS_LENGTH: {}".format(args.ts_length))
print(" @TS_DIM: {}".format(args.tsdim))
print(" @N_Class: {}".format(args.nclass))
print(" @Train_Size: {}".format(trainsize))
print(" @Test_Size: {}".format(testsize))
print(" @Train_Sample_Shape: {}".format(trainds[0][0].shape))


# print("--->", args.evalonly)

"""Pretrain"""
if not args.evalonly:
    """Load Model"""
    tsadmodel = model_TSAD.TSAD(args)
    tsadmodel = tsadmodel.to(args.device)
    print(" @Number of Parameteres:", tools.count_parameters(tsadmodel))
    print("------------------------------")

    """Bulid Loss"""
    lossf = loss.Loss(args=args)

    """Load optimizer"""
    optim = optimizer.build_optimizer(model=tsadmodel, args=args)

    """Load runner"""
    exp = runner.Runner(traindl=traindl, validdl=validdl, testdl=testdl,model=tsadmodel, optimizer=optim, lossf=lossf, args=args)
    trained_model = exp.train()
    checkpoint = {
        'epoch': args.epoch,
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optim.state_dict()}
    cpsavepath = "save/model/trained_model_{}.pth".format(args.dataname)
    torch.save(checkpoint, cpsavepath)
    print(" @model is saved at: {}".format(cpsavepath))

"""Evaluating"""
pretrain_model = model_TSAD.TSAD(args)
pretrain_model = pretrain_model.to(args.device)
cppath = "save/model/trained_model_{}.pth".format(args.dataname)
cp = torch.load(cppath)
pretrain_model.load_state_dict(cp['model_state_dict'])
pretrain_exp = runner.Runner(traindl=traindl, validdl=validdl, testdl=testdl, model=pretrain_model, optimizer=None, lossf=None, args=args)
pretrain_exp.evaluate(pretrain_model, testdl, args)



"""Finetune"""

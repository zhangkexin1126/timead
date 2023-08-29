import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from einops import repeat, rearrange
from scipy.signal import butter, lfilter, freqz
import os
import numpy as np
import random
import math
from tqdm import tqdm, trange
from collections import OrderedDict
from scipy import signal
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.metrics import PrecisionRecallDisplay, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt


import logging
logging.basicConfig(format='> %(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRunner(object):

    def __init__(self, traindl, validdl, testdl, model, optimizer, lossf, args):

        self.traindl = traindl
        self.validdl = validdl
        self.testdl = testdl
        self.model = model
        self.optimizer = optimizer
        self.lossf = lossf
        self.args = args
        self.epo_metrics = OrderedDict()

    def train(self):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, trained_model, testdl, args):
        raise NotImplementedError('Please override in child class')

    def keepmoving(self):
        raise NotImplementedError('Do not override in child class')
    
class Runner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(Runner, self).__init__(*args, **kwargs)

    def threshold_and_predict(self, scores, y_ture, thretype, point_adjust=False, composite_best_f1=False):
        """
        https://github.com/astha-chem/mvts-ano-eval/blob/main/src/evaluation/evaluation_utils.py
        :param scores:
        :param y_ture:
        :param thretype:
        :param point_adjust:
        :param composite_best_f1:
        :return:
        """
        score_t_test = scores.cpu().numpy()
        y_test = y_ture.cpu().numpy()
        true_events = get_events(y_test)
        if thretype == "fixed_thre":
            opt_thres = 0.0017
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
        elif thretype == "best_f1_test" and point_adjust:
            prec, rec, thresholds = precision_recall_curve(y_test, score_t_test, pos_label=1)
            fscore_best_time = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
            opt_num = np.squeeze(np.argmax(fscore_best_time))
            opt_thres = thresholds[opt_num]
            thresholds = np.random.choice(thresholds, size=5000) + [opt_thres]
            fscores = []
            for thres in thresholds:
                _, _, _, _, _, fscore = get_point_adjust_scores(y_test, score_t_test > thres, true_events)
                fscores.append(fscore)
            opt_thres = thresholds[np.argmax(fscores)]
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
        elif thretype == "best_f1_test" and composite_best_f1:
            prec, rec, thresholds = precision_recall_curve(y_test, score_t_test, pos_label=1)
            precs_t = prec
            fscores_c = [get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t) for thres, prec_t in
                         zip(thresholds, precs_t)]
            try:
                opt_thres = thresholds[np.nanargmax(fscores_c)]
            except:
                opt_thres = 0.0
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
        elif thretype == "best_f1_test":
            prec, rec, thres = precision_recall_curve(y_test, score_t_test, pos_label=1)
            fscore = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
            opt_num = np.squeeze(np.argmax(fscore))
            opt_thres = thres[opt_num]
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
        elif thretype == "top_k_time":
            test_anom_frac = 0.1
            opt_thres = np.nanpercentile(score_t_test, 100 * (1 - test_anom_frac), interpolation='higher')
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
        return pred_labels, opt_thres

    def train(self):
        print(">>>>>>> Start Train <<<<<<<")
        self.model.train()
        for epo in range(self.args.epoch):
            epo_sumloss = 0
            tb_loss = 0
            for i, batch in enumerate(tqdm(self.traindl)):
                self.optimizer.zero_grad()
                x, y = batch[0].to(self.args.device), batch[1].to(self.args.device)

                out = self.model(x)
                x_raw, x_rec, encoder_out = out[0], out[1], out[2]
                point_real, predicted_rep, context_rep = out[3], out[4], out[5]
                point_out_short, point_out_long = predicted_rep[0], predicted_rep[1] # point_out_1 -> point_out_3，采样点间隔逐渐变大
                context_short, context_long = context_rep[0], context_rep[1]

                # compute simsiamloss
                w = 0.5
                contrast_loss = w*self.lossf.contrastiveloss(point_real, point_out_short) + (1-w)*self.lossf.contrastiveloss(point_real, point_out_long)

                # long and short
                # long_short_error = self.lossf.reconstructionloss(point_out_short, point_out_long)

                # compute recloss
                rec_loss = self.lossf.reconstructionloss(x_raw, x_rec)


                # backward
                batch_loss = rec_loss + contrast_loss
                batch_loss.backward()
                self.optimizer.step()

                epo_sumloss += batch_loss.item()
                # epo_sumloss += batch_loss.item()

            if i == 0:
                i = i+1
            epo_meanloss = epo_sumloss / i
            logger.info('Training Epoch - {} Summary: EpoMeanLoss={}'.format(epo, epo_meanloss))
            # writer.add_scalar("epo_meanloss", epo_meanloss, epo)

        return self.model

    def evaluate(self, trained_model, testdl, args):
        print("=== Start Evaluating ===")

        gt, ad_scores = self.evaluate_ad(trained_model, testdl, args)
        gtnp = gt.cpu().numpy()
        anomalyscore = ad_scores
        raw_pred, opt_thre = self.threshold_and_predict(anomalyscore, gt, thretype="best_f1_test", point_adjust=True, composite_best_f1=False)
        # fixed_thre best_f1_test
        adjusted_pred = adjust_prediction(gtnp, raw_pred)
        print("Dataname:", args.dataname)
        self.show_results(gtnp, adjusted_pred)
        print("Best Threshold:", opt_thre)

    def evaluate_ad(self, trained_model, testdl, args):
        trained_model.eval()
        with torch.no_grad():
            gt = []
            ad_scores = []
            for k, batch in enumerate(tqdm(testdl)):
                x, y = batch[0].to(args.device), batch[1].to(args.device)
                gt.append(torch.flatten(y))
                out = trained_model(x)
                x_raw, x_rec, encoder_out = out[0], out[1], out[2]
                point_real, predicted_rep, context_rep = out[3], out[4], out[5]
                point_out_short, point_out_long = predicted_rep[0], predicted_rep[1] # point_out_1 -> point_out_3，采样点间隔逐渐变大
                context_short, context_long = context_rep[0], context_rep[1]
                score = self.compute_score(x_raw, x_rec, point_real, point_out_short, point_out_long, args)
                ad_scores.append(torch.flatten(score))

            gt = torch.cat(gt, dim=0)
            ad_scores = torch.cat(ad_scores, dim=0)

        return gt, ad_scores

    def compute_score(self, x_raw, x_rec, point_real, point_out1, point_out2, args):
        B = x_raw.shape[0]
        L = x_raw.shape[1]
        if args.evalmethod == "rec":
            rec_error_dim = (x_raw - x_rec) ** 2
            score = torch.mean(rec_error_dim, dim=-1)
            # 
        elif args.evalmethod == "softrec":
            rec_error_dim = (x_raw - x_rec) ** 2
            score = torch.softmax(torch.mean(rec_error_dim, dim=-1), dim=-1)

        elif args.evalmethod == "sim":
            crit = nn.CosineSimilarity(dim=-1).to(args.device)
            score1 = 1 - torch.abs(crit(point_out1, point_real))
            score2 = 1 - torch.abs(crit(point_out2, point_real))
            # score1 = torch.exp((1 - torch.abs(crit(point_out1, point_real)))/0.07)
            # score2 = torch.exp((1 - torch.abs(crit(point_out2, point_real)))/0.07)
            score = (score1+score2)/2
            score = score.reshape(B, -1)
            # score = torch.softmax(score, dim=-1)
 
        elif args.evalmethod == "best":
            rec_error_dim = (x_raw - x_rec) ** 2
            # scorerec = torch.mean(rec_error_dim, dim=-1)
            scorerec = torch.softmax(torch.mean(rec_error_dim, dim=-1), dim=-1)
            crit = nn.CosineSimilarity(dim=-1).to(args.device)
            score1 = 1 - torch.abs(crit(point_out1, point_real))
            score2 = 1 - torch.abs(crit(point_out2, point_real))
            # score1 = torch.exp((1 - torch.abs(crit(point_out1, point_real)))/0.07)
            # score2 = torch.exp((1 - torch.abs(crit(point_out2, point_real)))/0.07)
            
            scoresim = (score1+score2)/2
            scoresim = scoresim.reshape(B, -1)
            # scoresim = torch.softmax(scoresim, dim=-1)
            score = scorerec*scoresim
        
        elif args.evalmethod == "best2":
            pass


        return score
        

    def show_results(self, gt, pred):
        # print("Precision/Recall/F1")
        p = precision_score(gt, pred)
        r = recall_score(gt, pred)
        f1 = f1_score(gt, pred)
        print("\n>>>>>>>> Precision={}, Recall={}, F1={} <<<<<<<<\n".format(p, r, f1))

def get_events(y_test, outlier=1, normal=0, breaks=[]):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
                # A break point was hit, end current event and start new one
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim

        else:
            # event_by_time_true[tim] = 0
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events

def get_f_score(prec, rec):
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec)
    return f_score


def get_point_adjust_scores(y_test, pred_labels, true_events):
    tp = 0
    fn = 0
    for true_event in true_events.keys():
        true_start, true_end = true_events[true_event]
        if pred_labels[true_start:true_end].sum() > 0:
            tp += (true_end - true_start)
        else:
            fn += (true_end - true_start)
    fp = np.sum(pred_labels) - np.sum(pred_labels * y_test)

    prec, rec, fscore = get_prec_rec_fscore(tp, fp, fn)
    return fp, fn, tp, prec, rec, fscore

def get_prec_rec_fscore(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
    fscore = get_f_score(precision, recall)
    return precision, recall, fscore

def get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t, return_prec_rec=False):
    pred_labels = score_t_test > thres
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

def adjust_prediction(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    return pred
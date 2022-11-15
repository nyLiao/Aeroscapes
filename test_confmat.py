# -*- coding:utf-8 -*-
""" Get confusion matrix on validation set
File: logger.py
File Created: 2022-11-15
Author: nyLiao
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import torch
import torchvision.transforms as transforms

from model import *
from dataloader import DatasetVal, DatasetClean
from utils import *
from logger import Logger, ModelLogger, prepare_opt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='./data/', help='path to your dataset')
    parser.add_argument('-c', '--config', type=str, default='./config/unet.json', help='path to config JSON')
    parser.add_argument('-f', '--flag', type=str, help='id of the run')
    return prepare_opt(parser)

def restore_model(mname, logger):
    assert logger.path_existed, f"Path {logger.dir_save} not found"
    model_logger = ModelLogger(logger, state_only=True)
    model_logger.metric_name = 'iou'

    # ===== Model =====
    model = get_model(mname)
    model = model_logger.load_model('best', model=model).to(device)
    model.eval()
    return model


if __name__ == '__main__':
    args = get_args()
    BACH_SIZE = args.batch

    cln_dataset = DatasetClean(args.data)
    val_dataset = DatasetVal(args.data)
    cln_dataloader = torch.utils.data.DataLoader(cln_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)

    flag_run = "{}_{}".format(args.loss, args.flag)
    logger = Logger(prj_name=args.model, flag_run=flag_run)
    # logger.load_opt(args)
    model = restore_model(args.model, logger)

    cfms = np.zeros((12, 12), dtype=np.float64)

    for batch_i, (xy_val, xy_cln) in enumerate(zip(val_dataloader, cln_dataloader)):
        # get prediction
        with torch.no_grad():
            xs, _ = xy_val
            pred_mask = model(xs.to(device))
        pred_mask = torch.softmax(pred_mask, dim=1)
        y_pred = torch.argmax(pred_mask, dim=1)
        y_pred = y_pred.cpu()
        # transfer to label map
        h, w = y_pred.shape[1:]
        y_pred = transforms.functional.crop(y_pred, 8, 0, h-16, w)
        # get true label
        _, y_true = xy_cln

        y_pred = y_pred.reshape(-1).numpy()
        y_true = y_true.reshape(-1).numpy()
        cfm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=range(12), normalize=None)
        cfms += cfm

    np.save(logger.path_join('conf_mat.npy'), cfms)

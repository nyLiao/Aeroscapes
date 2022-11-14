import os
import sys
import argparse
import numpy as np

import torch

from model import *
from dataloader import DatasetVal
from utils import *
from logger import Logger, ModelLogger, prepare_opt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='./data/', help='path to your dataset')
    parser.add_argument('-c', '--config', type=str, default='./config/unet.json', help='path to config JSON')
    parser.add_argument('-f', '--flag', type=str, help='id of the run')
    # parser.add_argument('--batch', type=int, default=2, help='batch size')
    # parser.add_argument('--loss', type=str, default='crossentropy', help='focalloss | iouloss | crossentropy')
    return prepare_opt(parser)


def restore_model(mname, flag):
    logger = Logger(prj_name=mname, flag_run=flag)
    # logger.load_opt(args)
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

    val_dataset = DatasetVal(args.data)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)

    flag_run = "{}_{}".format(args.loss, args.flag)
    model = restore_model(args.model, flag_run)

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    for batch_i, (x, y) in enumerate(val_dataloader):
        with torch.no_grad():
            pred_mask = model(x.to(device))
            stuff_pred_mask,thing_pred_mask = model(x.to(device))
        # pred_mask = stuff_pred_mask+thing_pred_mask
        back_pred_mask = (stuff_pred_mask[:,0,:,:]+thing_pred_mask[:,0,:,:])/2
        back_pred_mask = back_pred_mask.unsqueeze(dim=1)
        stuff_pred_mask = stuff_pred_mask[:,8:,:,:]
        thing_pred_mask = thing_pred_mask[:,1:8,:,:]
        print(back_pred_mask.shape,thing_pred_mask.shape,stuff_pred_mask.shape)

        pred_mask = torch.cat((back_pred_mask,thing_pred_mask,stuff_pred_mask),dim=1)
        pred_mask = torch.softmax(pred_mask, dim=1)
        pred = torch.argmax(pred_mask, dim=1)
        intersection, union, target = intersectionAndUnionGPU(pred, y.to(device), 12)
        intersection_meter.update(intersection)
        union_meter.update(union)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = torch.mean(iou_class)
    print(mIoU)
    for i,iou in enumerate(iou_class):
        print('{}:{}'.format(i,iou))

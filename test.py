import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms as transforms
from tqdm import tqdm
from losses import FocalLoss, mIoULoss
from model import *
from dataloader import DatasetTrain,DatasetVal
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu" )
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Semantic segmentation dataset', help='path to your dataset')
    parser.add_argument('--batch', type=int, default=2, help='batch size')
    parser.add_argument('--loss', type=str, default='crossentropy', help='focalloss | iouloss | crossentropy')
    return parser.parse_args()

def acc(y, pred_mask):
    # print(y[0].sum(),torch.argmax(pred_mask, axis=1)[0].sum())
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
    return seg_acc

if __name__ == '__main__':
    args = get_args()
    BACH_SIZE = args.batch

    train_dataset = DatasetTrain('/home/ubuntu/data/aeroscapes/')
    val_dataset = DatasetVal('/home/ubuntu/data/aeroscapes/')

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)

    model = DoubleHeadUnet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=12,                      # model output channels (number of classes in your dataset)
        ).to(device) 
    
    # testing
    # checkpoint = torch.load('saved_models/unet_epoch_4_0.70781.pt')
    # model.load_state_dict(checkpoint)
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
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
      


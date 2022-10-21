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
from model import UNet
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

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)

    if args.loss == 'focalloss':
        criterion = FocalLoss(gamma=3/4).to(device)
    elif args.loss == 'iouloss':
        criterion = mIoULoss(n_classes=6).to(device)
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        print('Loss function not found!')


    model = UNet(n_channels=3, n_classes=12, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    min_loss = torch.tensor(float('inf'))

    os.makedirs('./saved_models', exist_ok=True)

    plot_losses = []
    scheduler_counter = 0

    
    # testing
    checkpoint = torch.load('saved_models/unet_epoch_4_0.70781.pt')
    model.load_state_dict(checkpoint)
    model.eval()
    val_loss_list = []
    val_acc_list = []   
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    for x, y in tqdm(val_dataloader):
        with torch.no_grad():    
            pred_mask = model(x.to(device))  
        pred = torch.argmax(torch.softmax(pred_mask, dim=1), dim=1)
        intersection, union, target = intersectionAndUnionGPU(pred, y.to(device), 12)
        intersection_meter.update(intersection)
        union_meter.update(union)
        val_loss = criterion(pred_mask, y.to(device))
        val_loss_list.append(val_loss.cpu().detach())
        # val_acc_list.append(acc(y,pred_mask).numpy())
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = torch.mean(iou_class)
        
    print('val loss : {:.5f} - val mIoU : {:.2f}'.format(np.mean(val_loss_list),mIoU))
      


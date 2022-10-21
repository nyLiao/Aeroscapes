import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

from losses import FocalLoss, mIoULoss
from model import UNet
from dataloader import DatasetTrain,DatasetVal
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/ubuntu/data/aeroscapes/', help='path to your dataset')
    parser.add_argument('--num_epochs', type=int, default=30, help='dnumber of epochs')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--loss', type=str, default='focalloss', help='focalloss | iouloss | crossentropy')
    return parser.parse_args()

def acc(y, pred_mask):
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
    return seg_acc

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

if __name__ == '__main__':
    args = get_args()
    N_EPOCHS = args.num_epochs
    BACH_SIZE = args.batch

    train_dataset = DatasetTrain(args.data)
    val_dataset = DatasetVal(args.data)

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


    # model = UNet(n_channels=3, n_classes=12, bilinear=True).to(device)
    model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=12,                      # model output channels (number of classes in your dataset)
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.num_epochs * len(train_dataloader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args.lr,
            ),
        )
    max_mIoU = -1

    os.makedirs('./saved_models', exist_ok=True)

    plot_losses = []
    scheduler_counter = 0

    for epoch in range(N_EPOCHS):
        # training
        model.train()
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(train_dataloader):

            pred_mask = model(x.to(device))  
            # print(pred_mask.shape,y.shape)
            loss = criterion(pred_mask, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(acc(y,pred_mask).numpy())

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    N_EPOCHS,
                    batch_i,
                    len(train_dataloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )
        scheduler_counter += 1
        # testing
        model.eval()
        val_loss_list = []
        val_iou_list = []
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        for batch_i, (x, y) in enumerate(val_dataloader):
            with torch.no_grad():    
                pred_mask = model(x.to(device))  
            pred = torch.argmax(torch.softmax(pred_mask, dim=1), dim=1)
            intersection, union, target = intersectionAndUnionGPU(pred, y.to(device), 12)
            intersection_meter.update(intersection)
            union_meter.update(union)
            val_loss = criterion(pred_mask, y.to(device))
            val_loss_list.append(val_loss.cpu().detach())
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        mIoU = torch.mean(iou_class)            
        print(' epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val iou : {:.2f}'.format(epoch, 
                                                                                                        np.mean(loss_list), 
                                                                                                        np.mean(acc_list), 
                                                                                                        np.mean(val_loss_list),
                                                                                                        mIoU))
        plot_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

        is_best = mIoU < max_mIoU
        if is_best == True:
            scheduler_counter = 0
            max_mIoU = min(mIoU, max_mIoU)
            torch.save(model.state_dict(), './saved_models/unet_epoch_{}_{:.5f}.pt'.format(epoch,mIoU))
        

    # plot loss
    plot_losses = np.array(plot_losses)
    plt.figure(figsize=(12,8))
    plt.plot(plot_losses[:,0], plot_losses[:,1], color='b', linewidth=4)
    plt.plot(plot_losses[:,0], plot_losses[:,2], color='r', linewidth=4)
    plt.title(args.loss, fontsize=20)
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('loss',fontsize=20)
    plt.grid()
    plt.legend(['training', 'validation']) # using a named size
    plt.savefig('loss_plots.png')


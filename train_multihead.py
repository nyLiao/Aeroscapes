import numpy as np
import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import segmentation_models_pytorch as smp

from losses import *
from model import *
from dataloader import DatasetTrain,DatasetVal
from utils import *
from logger import Logger, ModelLogger, prepare_opt


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, default='/home/ubuntu/data/aeroscapes/', help='path to your dataset')
    # parser.add_argument('--data', type=str, default='/fs/resource/dataset/cv/aeroscapes/', help='path to your dataset')
    parser.add_argument('--data', type=str, default='./data/', help='path to your dataset')
    parser.add_argument('--config', type=str, default='./config/unet.json', help='path to config JSON')
    # parser.add_argument('--num_epochs', type=int, default=30, help='dnumber of epochs')
    # parser.add_argument('--batch', type=int, default=4, help='batch size')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--loss', type=str, default='focalloss', help='focalloss | iouloss | crossentropy')
    parser.add_argument('--local_rank', type=int, default=-1)
    return prepare_opt(parser)

def acc(y, pred_mask):
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
    return seg_acc

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


if __name__ == '__main__':
    dist.init_process_group(backend="nccl")
    args = get_args()
    local_rank = int(args.local_rank)
    device = torch.device(f"cuda:{local_rank}")
    N_EPOCHS = args.num_epochs
    BACH_SIZE = args.batch

    # ===== Data =====
    train_dataset = DatasetTrain(args.data)
    val_dataset = DatasetVal(args.data)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,batch_size=BACH_SIZE, num_workers=28)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=28)

    # ===== Loss =====
    if args.loss == 'focalloss':
        # criterion = WeightedFocalLoss(gamma=3/4).to(device)
        alpha = [0.25]
        for i in range(11):
            alpha.append(0.75)
        # print(len(alpha))
        criterion = FocalLoss(gamma=3/4,alpha=[0.5,1,2,1,1,1,1,2,0.75,0.75,0.75,0.75]).to(device)
        # stuff_criterion = FocalLoss(gamma=3/4,alpha=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1]).to(device)
        # thing_criterion = FocalLoss(gamma=3/4,alpha=[0.5,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5]).to(device)
    elif args.loss == 'iouloss':
        criterion = mIoULoss(n_classes=12).to(device)
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().to(device)
        stuff_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1,1,1,1,2,2,2,2])).to(device)
        thing_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1,1,1,1,2,2,2,2])).to(device)
    else:
        print('Loss function not found!')

    # ===== Model =====
    if args.model == 'unet':
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=12,
        )
    elif args.model == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=12,
        )
    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank],
                                                      output_device=local_rank)
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
    plot_losses = []

    for epoch in range(N_EPOCHS):
        # training
        model.train()
        train_sampler.set_epoch(epoch)
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(train_dataloader):
            if x.shape[0] < 2:
                continue
            pred_mask = model(x.to(device))
            # stuff_pred_mask,thing_pred_mask = model(x.to(device))
            # print(pred_mask.shape,y.shape)
            # stuff_y, thing_y = divide_labels(y.to(device))
            loss = criterion(pred_mask, y.to(device))
            # stuff_loss = stuff_criterion(stuff_pred_mask, y.to(device))
            # thing_loss = thing_criterion(thing_pred_mask, y.to(device))
            # print(thing_pred_mask.shape,stuff_pred_mask.shape)
            # pred_mask = stuff_pred_mask+thing_pred_mask
            # pred_mask = torch.cat((thing_pred_mask,stuff_pred_mask),dim=1)
            # loss= stuff_loss+thing_loss
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
        # testing
        if local_rank==0:
            model.eval()
            val_loss_list = []
            val_iou_list = []
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            for batch_i, (x, y) in enumerate(val_dataloader):
                with torch.no_grad():
                    pred_mask = model(x.to(device))
                    # stuff_pred_mask,thing_pred_mask = model(x.to(device))
                # pred_mask = stuff_pred_mask+thing_pred_mask
                # back_pred_mask = (stuff_pred_mask[:,0,:,:]+thing_pred_mask[:,0,:,:])/2
                # back_pred_mask = back_pred_mask.unsqueeze(dim=1)
                # stuff_pred_mask = stuff_pred_mask[:,8:,:,:]
                # thing_pred_mask = thing_pred_mask[:,1:8,:,:]
                # print(back_pred_mask.shape,thing_pred_mask.shape,stuff_pred_mask.shape)
                # pred_mask = torch.cat((back_pred_mask,thing_pred_mask,stuff_pred_mask),dim=1)
                pred_mask = torch.softmax(pred_mask, dim=1)
                pred = torch.argmax(pred_mask, dim=1)
                intersection, union, target = intersectionAndUnionGPU(pred, y.to(device), 12)
                intersection_meter.update(intersection)
                union_meter.update(union)
                val_loss = criterion(pred_mask, y.to(device))
                val_loss_list.append(val_loss.cpu().detach())
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            mIoU = torch.mean(iou_class)
            print(' epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val iou : {:.3f}'.format(epoch,
                                                                                                        np.mean(loss_list),
                                                                                                        np.mean(acc_list),
                                                                                                        np.mean(val_loss_list),
                                                                                                        mIoU))
            for i,iou in enumerate(iou_class):
                print('{}:{}'.format(i,iou))
            plot_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

            is_best = mIoU > max_mIoU
            if is_best == True:
                max_mIoU = max(mIoU, max_mIoU)
                torch.save(model.state_dict(), './saved_models/unet_epoch_{}_{:.5f}.pt'.format(epoch, mIoU))


    # plot loss
    # plot_losses = np.array(plot_losses)
    print(plot_losses)
    # plt.figure(figsize=(12,8))
    # plt.plot(plot_losses[:,0], plot_losses[:,1], color='b', linewidth=4)
    # plt.plot(plot_losses[:,0], plot_losses[:,2], color='r', linewidth=4)
    # plt.title(args.loss, fontsize=20)
    # plt.xlabel('epoch',fontsize=20)
    # plt.ylabel('loss',fontsize=20)
    # plt.grid()
    # plt.legend(['training', 'validation']) # using a named size
    # plt.savefig('loss_plots.png')

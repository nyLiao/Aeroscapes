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
    parser.add_argument('-d', '--data', type=str, default='./data/', help='path to your dataset')
    parser.add_argument('-c', '--config', type=str, default='./config/unet.json', help='path to config JSON')
    # parser.add_argument('--num_epochs', type=int, default=30, help='dnumber of epochs')
    # parser.add_argument('--batch', type=int, default=4, help='batch size')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--loss', type=str, default='focalloss', help='focalloss | iouloss | crossentropy')
    parser.add_argument('--nworker', type=int, default=28)
    parser.add_argument('--local_rank', type=int, default=-1)
    return prepare_opt(parser)


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
    train_dataloader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,batch_size=BACH_SIZE, num_workers=args.nworker)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=args.nworker)

    flag_run = "{}_date".format(args.loss)
    logger = Logger(prj_name=args.model, flag_run=flag_run)
    logger.save_opt(args)
    model_logger = ModelLogger(logger, state_only=True)
    model_logger.metric_name = 'iou'

    # ===== Loss =====
    criterion = get_loss(args.loss)
    criterion = criterion.to(device)

    # ===== Model =====
    model = get_model(args.model)
    model_logger.regi_model(model, save_init=False)
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

    for epoch in range(N_EPOCHS):
        # ===== training =====
        model.train()
        train_sampler.set_epoch(epoch)
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(train_dataloader):
            # BatchNorm crash when batch size < 2
            if x.shape[0] < 2:
                continue
            pred_mask = model(x.to(device))
            loss = criterion(pred_mask, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(acc(y,pred_mask).numpy())

            msg = f"\rEpoch {epoch+1}/{N_EPOCHS} | Batch {batch_i+1}/{len(train_dataloader)} | Loss {loss.cpu().detach().numpy():.6f} / {np.mean(loss_list):.6f}"
            sys.stdout.write(msg)

        print()
        # ===== testing =====
        if local_rank == 0:
            model.eval()
            val_loss_list = []
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            for batch_i, (x, y) in enumerate(val_dataloader):
                with torch.no_grad():
                    pred_mask = model(x.to(device))
                pred_mask = torch.softmax(pred_mask, dim=1)
                pred = torch.argmax(pred_mask, dim=1)
                intersection, union, target = intersectionAndUnionGPU(pred, y.to(device), 12)
                intersection_meter.update(intersection)
                union_meter.update(union)
                val_loss = criterion(pred_mask, y.to(device))
                val_loss_list.append(val_loss.cpu().detach())

            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            mIoU = torch.mean(iou_class)
            msg = f"Epoch:{epoch+1:04d} | trn loss:{np.mean(loss_list):.5f}, trn acc:{np.mean(acc_list):.4f} | val loss:{np.mean(val_loss_list):.5f}, val iou:{mIoU:.4f}"
            logger.print(msg)
            msg = '\tiou class:' + ','.join('{:.6f}'.format(i) for i in iou_class)
            logger.print(msg)

            model_logger.save_best(mIoU, epoch=epoch)
            model_logger.save_epoch(epoch=epoch, period=N_EPOCHS)

    logger.print_on_top(f"Val best: {model_logger.metric_best:0.4f}, epoch: {model_logger.epoch_best}")

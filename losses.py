import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def get_loss(name):
    if name == 'focalloss':
        # criterion = WeightedFocalLoss(gamma=3/4)
        return FocalLoss(gamma=3/4,alpha=[0.5,1,2,1,1,1,1,2,0.75,0.75,0.75,0.75])
    elif name == 'iouloss':
        return mIoULoss(n_classes=12)
    elif name == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif name == 'mhcrossentropy':
        stuff_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1,1,1,1,2,2,2,2]))
        thing_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1,1,1,1,2,2,2,2]))
        return stuff_criterion, thing_criterion
    else:
        raise NotImplementedError("Loss {} not found!".format(name))


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.75, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target.type(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        # print(loss.shape)
        if self.size_average: return loss.mean()
        else: return loss.sum()


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n,h,w = tensor.size()
        one_hot = torch.zeros(n,self.classes,h,w).to(tensor.device).scatter_(1,tensor.view(n,1,h,w),1)
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs,dim=1)

        # Numerator Product
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)

        #Denominator
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)

        loss = inter/union

        ## Return average loss over classes and batch
        return 1-loss.mean()


class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        input = F.log_softmax(input, dim=-1)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)
        loss = -1 * (1-pt)**self.gamma * logpt
        # print(loss.shape)
        if self.size_average: return loss.mean()
        else: return loss.sum()

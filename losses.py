import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def get_loss(name, args):
    if name == 'focalloss':
        # criterion = WeightedFocalLoss(gamma=3/4)
        # return FocalLoss(gamma=3/4,alpha=[0.5,1,2,1,1,1,1,2,0.75,0.75,0.75,0.75])
        criterion = FocalLoss(gamma=args.gamma,alpha=[0.5,1,2,1,1,1,1,2,0.75,0.75,0.75,0.75])
        flag = 'focalloss_{:.2f}'.format(args.gamma)
        return criterion, flag
    elif name == 'iouloss':
        return mIoULoss(n_classes=12), 'iouloss'
    elif name == 'iouloss2':
        return mIoULoss2(n_classes=12), 'iouloss2'
    elif name == 'crossentropy':
        return nn.CrossEntropyLoss(), 'crossentropy'
    elif name == 'mhcrossentropy':
        stuff_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1,1,1,1,2,2,2,2]))
        thing_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1,1,1,1,2,2,2,2]))
        return (stuff_criterion, thing_criterion), 'mhcrossentropy'
    elif name == 'dice':
        return DiceLoss(), 'dice'
    elif name == 'tversky':
        criterion = TverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        flag = 'tversky_{:.2f}_{:.2f}_{:.2f}'.format(args.alpha, args.beta, args.gamma)
        return criterion, flag
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


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=12):
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


class mIoULoss2(nn.Module):
    def __init__(self, weight=None, n_classes=12):
        super(mIoULoss2, self).__init__()
        self.classes = n_classes
        self.eps = 1e-7

    def compute_score(self, output, target, smooth, eps, dims):
        assert output.size() == target.size()
        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)

        union = cardinality - intersection
        jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
        return jaccard_score

    def forward(self, input, target):
        input = input.log_softmax(dim=1).exp()
        bs = target.size(0)
        num_classes = input.size(1)
        dims = (0, 2)

        target = target.view(bs, -1)
        input = input.view(bs, num_classes, -1)

        target = F.one_hot(target, num_classes)  # N,H*W -> N,H*W, C
        target = target.permute(0, 2, 1)  # H, C, H*W

        scores = self.compute_score(input, target.type(input.dtype), smooth=0., eps=self.eps, dims=dims)
        loss = 1.0 - scores
        mask = target.sum(dims) > 0
        loss *= mask.float()
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.0, eps=1e-7, n_classes=12):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        self.n_classes = n_classes

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth, eps, dims):
        assert output.size() == target.size()
        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        return dice_score

    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        bs = target.size(0)
        dims = (0, 2)

        target = target.view(bs, -1)
        input = input.view(bs, self.n_classes, -1)
        target = F.one_hot(target, self.n_classes)  # N,H*W -> N,H*W, C
        target = target.permute(0, 2, 1)  # H, C, H*W
        scores = self.compute_score(input, target.type_as(input), smooth=self.smooth, eps=self.eps, dims=dims)
        loss = 1.0 - scores
        mask = target.sum(dims) > 0
        loss *= mask.to(loss.dtype)
        return self.aggregate_loss(loss)


class TverskyLoss(DiceLoss):
    def __init__(self, smooth=0.0, eps=1e-7, alpha=0.5, beta=0.5, gamma=1.0):
        super(TverskyLoss, self).__init__(smooth, eps)
        self.alpha = alpha  # false negative
        self.beta = beta    # false positive
        self.gamma = gamma

    def compute_score(self, output, target, smooth, eps, dims):
        assert output.size() == target.size()
        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)  # TP
            fp = torch.sum(output * (1. - target), dim=dims)
            fn = torch.sum((1 - output) * target, dim=dims)
        else:
            intersection = torch.sum(output * target)  # TP
            fp = torch.sum(output * (1. - target))
            fn = torch.sum((1 - output) * target)

        tversky_score = (intersection + smooth) / (intersection + self.alpha * fn + self.beta * fp + smooth).clamp_min(eps)
        return tversky_score

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, input, target):
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor).to(input.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = input - batch_m

        output = torch.where(index, x_m, input)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

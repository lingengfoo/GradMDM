"""
Training file for HRL stage. Support Pytorch 3.0 and multiple GPUs.
"""

from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import autograd
from torch import optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse
import time
import logging
import models
import sys

from utils import ListAverageMeter, AverageMeter, more_config, accuracy


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet training with gating')
    parser.add_argument('--model-type', metavar='ARCH', default='rl', choices=['rl','sp'])
    parser.add_argument('--cmd', choices=['train', 'test'])
    parser.add_argument('--gate-type', default='rnn',
                        choices=['rnn'], help='gate type,only support RNN Gate')
    parser.add_argument('--data', '-d', default='dataset/imagenet/',
                        type=str, help='path to the imagenet data')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of total epochs (default: 120)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=20, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum used in SGD')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained',
                        action='store_true', help='use pretrained model')
    parser.add_argument('--save-folder', default='save_checkpoints',
                        type=str, help='folder to save the checkpoints')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='cropping size of the input')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scaling size of the input')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--alpha', default=0.01, type=float,
                        help='tuning hyper-parameter in the hybrid loss')
    parser.add_argument('--rl-weight', default=0.01, type=float,
                        help='scaling weight for rewards')
    parser.add_argument('--restart', action='store_true', help='restart ckpt')
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for gate parameter initialization')
    parser.add_argument('--gamma', type=float, default=100)
    parser.add_argument('--acc-maintain', action='store_true', help='to disturb the sample maintaining the accuracy')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model_type == 'rl':
        args.arch = "imagenet_rnn_gate_rl_101"
        args.resume = "resnet-101-rnn-imagenet.pth.tar"
        args.alpha = [50,3.5]
    elif args.model_type == 'sp':
        args.arch = "imagenet_rnn_gate_101"
        args.resume = "resnet-101-rnn-sp-imagenet.pth.tar"
        args.alpha = [0.35,3.5]
    more_config(args)
    print(args)
    logging.info('CMD: '+' '.join(sys.argv))

    test_model(args)

def test_model(args):
    # create model
    model = models.__dict__[args.arch]()
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t = transforms.Compose([
        transforms.Scale(args.scale_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, t),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    validate(args, val_loader, model, criterion, args.start_epoch)

def tanh_rescale(x, x_min=[-2.4291,-2.4183,-2.2214], x_max=[2.5141,2.5968,2.7537], type=False):
    x_min, x_max = torch.tensor(x_min)[None,:,None,None], torch.tensor(x_max)[None,:,None,None]
    return (torch.tanh(0.8*x) + 1) / 2 * (x_max - x_min) + x_min

def gate_loss(logprobs, threshold=0.5, upper_bound=1.0, alpha=[50,3.5]):

    alpha_pos = alpha[0]
    gateloss_pos = upper_bound-torch.clamp(logprobs, min=threshold)
    gateloss_pos = gateloss_pos.norm(p=alpha_pos, dim=[1])
    gateloss_pos = gateloss_pos.sum()

    alpha_neg = alpha[1]
    gateloss_neg = torch.clamp(logprobs, max=threshold)
    gateloss_neg = gateloss_neg.norm(p=alpha_neg, dim=[1])
    gateloss_neg = -1*gateloss_neg.sum()

    return gateloss_pos, gateloss_neg

def save_img(img, name):
    from PIL import Image
    import numpy
    mean = torch.tensor([[[[0.4914]], [[0.4822]], [[0.4465]]]])
    std = torch.tensor([[[[0.2023]], [[0.1994]], [[0.2010]]]])
    img = (img * std + mean) * 255
    img = img.squeeze()
    img = img.permute(1,2,0)
    img = img.detach().numpy().astype(numpy.uint8)
    img = Image.fromarray(img)
    img.save(name)

def validate(args, val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    skip_ori_ratios = ListAverageMeter()
    skip_ratios = ListAverageMeter()
    prec1s_ori = AverageMeter()
    prec1s_mod = AverageMeter()
    vars = AverageMeter()


    # switch to evaluation mode
    model.eval()
    index_modified = 0
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = input

        modifier = torch.zeros(input_var.size()).float()
        modifier_var = autograd.Variable(modifier, requires_grad=True)
        optimizer = optim.Adam([modifier_var], lr=args.lr)

        output, masks, logprobs, hidden  = model(input_var)
        prec1, = accuracy(output.data, target, topk=(1,))
        
        skips_ori = masks.detach().mean(0)
        skip_ori_ratios.update(skips_ori, input.size(0))
        prec1s_ori.update(prec1)

        # save original image
        # img = input_var
        # for i in range(len(img)):
        #     save_img(img[i], "output/original{:02d}.jpg".format(i))
        
        for iter in range(100):
            skips, var, sample_num, prec1 = optimize(optimizer, model, input_var, modifier_var, target, output.data, iter, args)

        skip_ratios.update(skips, sample_num)
        prec1s_mod.update(prec1, sample_num)
        vars.update(var, sample_num)
        
        # break
        # # save modified image
        # input_adv = modifier_var + input_var
        # input_adv = tanh_rescale(input_adv)
        # import math
        # path = "output/sample_ours_k{}_rl".format(int(math.log10(args.gamma)))
        # if not os.path.exists(path):
        #     os.mkdir(path)
        # for sample in input_adv:
        #     save_img(sample, path+"/{:05d}.jpg".format(index_modified))
        #     index_modified = index_modified + 1


        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or (i == (len(val_loader) - 1)):
            logging.info(
                'Test: Epoch[{0}][{1}/{2}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                # 'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                # 'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'
                # 'Prec@5: {top5.val:.3f}({top5.avg:.3f})\t'
                    .format(epoch, i, len(val_loader),
                            batch_time=batch_time,
                            # top1=top1,
                )
            )

            cp_ori = ((sum(skip_ori_ratios.avg) + 1) / (skip_ori_ratios.len + 1)) * 100
            cp = ((sum(skip_ratios.avg) + 1) / (skip_ratios.len + 1)) * 100

            recovery = ((100-cp_ori) - (100-cp)) / (100-cp_ori) * 100
            logging.info('***Recovery: {:.3f} %'.format(recovery))

            logging.info('***Original prec: {:.5f}'.format(prec1s_ori.avg))

            logging.info('***Final prec: {:.5f}'.format(prec1s_mod.avg))

            logging.info('***Final Var: {:.5f}'.format(vars.avg))

    # always keep the first block
    cp = ((sum(skip_ori_ratios.avg) + 1) / (skip_ori_ratios.len + 1)) * 100
    logging.info('***Original Computation Percentage: {:.3f} %'.format(cp))

    cp = ((sum(skip_ratios.avg) + 1) / (skip_ratios.len + 1)) * 100
    logging.info('***Final Computation Percentage: {:.3f} %'.format(cp))



    # return top1.avg

def optimize(optimizer, model, input_var, modifier_var, target, ori_output, iter, args):
    autograd = True
    alpha = args.alpha
    if iter < 10:
        autograd = False
        alpha = [1,1]

    # compute output
    input_adv = tanh_rescale(modifier_var + input_var)
    output, masks, gates, hidden = model(input_adv, autograd=autograd)
    
    if args.acc_maintain:
        l2_dist = ((input_adv-input_var)**2).sum() + ((ori_output-output)**2).sum() * 100000
    else:
        l2_dist = ((input_adv-input_var)**2).sum()
    gateloss_pos, gateloss_neg = gate_loss(gates, 0.5, alpha=alpha)
    
    # L2 gradient calculating
    optimizer.zero_grad()
    l2_dist.backward(retain_graph=True)
    l2_dist_grad = modifier_var.grad.clone().detach()

    # postive gradient calculating
    optimizer.zero_grad()
    gateloss_pos.backward(retain_graph=True)
    gradient_pos = modifier_var.grad.clone().detach()

    # negative gradient calculating
    optimizer.zero_grad()
    gateloss_neg.backward()
    gradient_neg = modifier_var.grad.clone().detach()

    # optimizing
    modifier_var.grad = 1.5e+5 * gradient_neg + l2_dist_grad * args.gamma
    optimizer.step()

    img_size = input_var.shape[1] * input_var.shape[2] * input_var.shape[3]
    vars = ((input_adv-input_var)**2).sum([1,2,3])/img_size

    index = vars!=torch.nan
    sample_num = index.sum()
    skips = masks[index].detach().mean(0)
    vars = vars[index].detach().mean(0)

    prec1, = accuracy(output.data, target, topk=(1,))

    return skips, vars, sample_num, prec1

if __name__ == '__main__':
    main()

import os
from PIL import Image
import numpy
import torch
from utils import AverageMeter


mean = torch.tensor([0.4914, 0.4822, 0.4465])[:,None,None]
std = torch.tensor([0.2023, 0.1994, 0.2010])[:,None,None]


def l2_cal(ori_img, mod_img):
    ori_img = torch.FloatTensor(numpy.array(ori_img)).permute(2,0,1) / 255
    ori_img = (ori_img  - mean)/std

    mod_img = torch.FloatTensor(numpy.array(mod_img)).permute(2,0,1) / 255
    mod_img = (mod_img  - mean)/std

    dis = ori_img-mod_img
    dis = (dis**2).mean()
    return dis

filelist = os.listdir('output/original/')
filelist.sort()
ours_l2s = AverageMeter()
ilfo_l2s = AverageMeter()

for idx, file in enumerate(filelist):
    ori_img = Image.open("output/original/"+file)

    # ours_img = Image.open("output/ours_k-1_rl/"+file)
    ours_img = Image.open("AutoGrad/skipnet/ours_k-2_rl/"+file)
    ours_l2 = l2_cal(ori_img, ours_img)
    ours_l2s.update(ours_l2)

    ilfo_img = Image.open("output/ilfo_rl/"+file)
    ilfo_l2 = l2_cal(ori_img, ilfo_img)
    ilfo_l2s.update(ilfo_l2)

    if idx % 100 == 0:
        print("ilfo:{:.5f}, ours:{:.5f}".format(ilfo_l2s.avg, ours_l2s.avg))


    

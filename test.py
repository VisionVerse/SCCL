#!/usr/bin/env python
# coding: utf-8

import os
import torch
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"


import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import cpu_count
from thop import profile
import logging
from PIL import Image

from models.unet import SCCL
from toolbox.datasets import SegDataset
from toolbox.utils import save_ckpt_zh
from toolbox import loss_iccl
from toolbox.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from options import opt

from torch.optim.lr_scheduler import LambdaLR
from toolbox.Ranger import Ranger

ROOT = opt.root

train_dataset = SegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, img_size=opt.image_size, trans=True)
valid_dataset = SegDataset(VAL_IMG_DIR, VAL_MASK_DIR)
test_dataset = SegDataset(TEST_IMG_DIR, TEST_MASK_DIR, trans=True)

# ----------  Model
model = SCCL(img_channel, opt.num_class)

if opt.parallel and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model.cuda()


# ============  TEST LOOP
def test(model, test_dataloader, save_path):

    model.eval()

    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    MAE_ = MAE()

    total_step = len(test_dataloader)
    n = 0
    with torch.no_grad():
        pbar = enumerate(test_dataloader)
        pbar = tqdm(pbar, total=total_step)

        for i, data in pbar:
            # get the inputs
            images, labels, names = data['image'], data['mask'], data['name']

            images = images.cuda()
            labels = labels.cuda()

            n += 1

            # run the model and apply sigmoid
            outputs_dict = model(images)
            outputs = outputs_dict['out'].argmax(1)

            res = outputs.sigmoid().data.cpu().numpy().squeeze()  # sigmoid
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # 归一化
            res = res * 255
          
            gt = labels.cpu().numpy().squeeze() * 255

            FM.step(pred=res, gt=gt)
            WFM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)
            MAE_.step(pred=res, gt=gt)

            predict = Image.fromarray(np.uint8(res))
            predict.save(os.path.join(save_path, name[0]))

    f_metrics = WFM.get_results()["wfm"]
    s_metrics = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    e_metrics = em["curve"].mean()
    mae = MAE_.get_results()["mae"]



if __name__ == '__main__':
    file = 'best_model.pth'
    model_file = MODEL_SAVE_PATH + file
    save_img_path = model_file[:-4]
    model.load_state_dict(torch.load(model_file))
    test(model, test_dataloader, save_img_path)




import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import pandas as pd
from time import time
import argparse

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool, DinkNet34_logits, DinkNet34_abn, DinkNet34_context_abn, DinkNet34_psp, DinkNet34_psp64, DinkNet34_psp64_hdc, DinkNet34_abn_psp128
from framework import MyFrame
from loss import dice_bce_loss, FocalLoss, dice_focal_loss, dice_loss2, bce_with_logits_loss, dice_bce_loss2
from data import ImageFolder

parser = argparse.ArgumentParser(description='Train a PP-Linknet model on a folder.')
parser.add_argument('-t', '--train_folder', type=str, required=True, help='Dir to training jpg and mask images.')
parser.add_argument('-n', '--name', type=str, required=True, help='Name of the result model.')
parser.add_argument('-s', '--shape', type=int, default=1024, help='Shape of the input images.')
parser.add_argument('-b', '--batchsize', type=int, default=9, help='Batchsize per card.')
args = vars(parser.parse_args())

SHAPE = (args['shape'], args['shape'])
ROOT = args['train_folder']
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = map(lambda x: x[:-8], imagelist)
trainlist = list(trainlist)
print('Number of training images %d\n' % len(trainlist))
NAME = args['name']
BATCHSIZE_PER_CARD = args['batchsize']
power = 0.9

solver = MyFrame(DinkNet34_psp64, FocalLoss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT, SHAPE)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=torch.cuda.device_count()*8)

history = {'train_loss':[]}
mylog = open('logs/'+NAME+'.log','w')
mylog.write('Number of training images %d\n' % len(trainlist))
tic = time()
no_optim = 0
total_epoch = 90
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        #print(train_loss)
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    mylog.write('********\n')
    mylog.write('epoch:%d    time:%d\n' % (epoch,int(time()-tic)))
    mylog.write('train_loss:%f\n' % train_epoch_loss)
    mylog.write('SHAPE: (%d, %d)' % (SHAPE[0], SHAPE[1]))
    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)
    history['train_loss'].append(train_epoch_loss)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save(os.path.join('weights/', NAME))
    if no_optim > 6:
        mylog.write('early stop at %d epoch' % epoch)
        print('early stop at %d epoch' % epoch)
        break
    # TODO: implementing poly learning rate here!
    solver.update_poly_lr(poly_rate=(1 - float(epoch)/total_epoch)**power, mylog = mylog)
#     if no_optim > 3:
#         if solver.old_lr < 5e-7:
#             break
#         solver.load('weights/'+NAME+'.th')
#         solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()

mylog.write('Finish!')
print('Finish!')
mylog.close()

df = pd.DataFrame(data = history)
df.to_csv('logs/loss_%s.csv' % NAME, index=False)
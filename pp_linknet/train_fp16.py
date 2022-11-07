import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
from time import time
import pandas as pd

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool, DinkNet34_fp16
from framework import MyFrame
from loss import dice_bce_loss_fp16
from data import ImageFolder

SHAPE = (1024,1024)
ROOT = '/home/antran/map-workspace/data/deep-globe/train/'
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = map(lambda x: x[:-8], imagelist)
trainlist = list(trainlist)
NAME = 'dink34_fp16_retrain01'
BATCHSIZE_PER_CARD = 12

solver = MyFrame(DinkNet34_fp16, dice_bce_loss_fp16, 2e-4, fp16=True)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)

history = {'train_loss':[]}
mylog = open('logs/'+NAME+'.log','w')
tic = time()
no_optim = 0
total_epoch = 64
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        img = img.to(dtype=torch.float16)
        #mask = mask.to(dtype=torch.float16)
        solver.set_input(img, mask)
        train_loss = solver.optimize()
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
        solver.save('weights/'+NAME+'.th')
    if no_optim > 6:
        mylog.write('early stop at %d epoch' % epoch)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()

mylog.write('Finish!')
print('Finish!')
mylog.close()

df = pd.DataFrame(data = history)
df.to_csv('logs/loss_%s.csv' % NAME, index=False)
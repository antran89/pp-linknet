import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import argparse
from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool, DinkNet34_abn, DinkNet34_context_abn, DinkNet34_psp, DinkNet34_psp64, DinkNet34_psp64_hdc, DinkNet34_abn_psp128, DinkNet34_logits

BATCHSIZE_PER_CARD = 4

def pad(img, pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def test_one_img_from_path(self, path, evalmode = True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        return self.test_one_img_from_path_2(path)
#         if batchsize >= 8:
#             return self.test_one_img_from_path_1(path)
#         elif batchsize >= 4:
#             return self.test_one_img_from_path_2(path)
#         elif batchsize >= 2:
#             return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img, pads = pad(img)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        mask2 = unpad(mask2, pads)
        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img, pads = pad(img)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        mask2 = unpad(mask2, pads)
        return mask2
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = np.array(img, np.float32)/255.0
        img = (img - (0.3262061, 0.39206991, 0.41451698)) / (0.11344261, 0.1211441, 0.14105087)
        img, pads = pad(img)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        #img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5.copy()).cuda())
        img6 = img4.transpose(0,3,1,2)
        #img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6.copy()).cuda())
        
        maska = torch.sigmoid(self.net.forward(img5)).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = torch.sigmoid(self.net.forward(img6)).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        mask3 = unpad(mask3, pads)
        return mask3
    
    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img, pads = pad(img)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        mask3 = unpad(mask3, pads)
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

parser = argparse.ArgumentParser(description='Testing a model.')
parser.add_argument('-w', '--weight', type=str, required=True, help='Dir to training jpg and mask images.')
parser.add_argument('-b', '--jpg_folder', type=str, required=True, help='Path to test jpg folder.')
parser.add_argument('-n', '--mask_folder', type=str, default=None, help='Name of the result model.')
parser.add_argument('-t', '--threshold', type=float, default=4.0, help='Threshold to flag it a road or non-road.')
parser.add_argument('-s', '--start_index', type=int, default=0, help='Start index, default, 0.')
parser.add_argument('-e', '--end_index', type=int, default=None, help='End index, default, None, means to the end.')
args = vars(parser.parse_args())

solver = TTAFrame(DinkNet34_psp64)
weight_path = args['weight']
solver.load(weight_path)
tic = time()

jpg_folder = args['jpg_folder']
if args['mask_folder'] != None:
    mask_folder = args['mask_folder']
else:
    mask_folder = jpg_folder
target = os.path.join(jpg_folder, '..', 'results', os.path.basename(weight_path)[:-3] + '_thresh%0.1f'%args['threshold'])
if not os.path.exists(target):
    os.makedirs(target)
val = glob.glob(os.path.join(jpg_folder, '*.jpg'))

for i,path in enumerate(val[args['start_index']:args['end_index']]):
    if i%10 == 0:
        print('%d    %.2f' % (i/10, time()-tic))
    name = os.path.basename(path)
    outfile = os.path.join(target, name[:-7]+'mask.png')
    if os.path.exists(outfile):
        continue
    
    mask = solver.test_one_img_from_path(path)
    mask[mask>args['threshold']] = 255
    mask[mask<=args['threshold']] = 0
    mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
    cv2.imwrite(outfile, mask.astype(np.uint8))

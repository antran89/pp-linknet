'''
Author: an.tran@grab.com
Predicting the road mask in the satellite image.
'''
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
import logging
import pandas as pd

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool, DinkNet34_abn, DinkNet34_context_abn, DinkNet34_psp, DinkNet34_psp64, DinkNet34_psp64_hdc, DinkNet34_abn_psp128, DinkNet34_logits
import apollo_python_common.io_stream.io_utils as io_utils
from io_interface import IO_Interface

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
    def __init__(self, net, io):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.io = io
        
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
        # img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = self.io.get_image(path)
        img = img[:,:,::-1]
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
        #img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = self.io.get_image(path)
        img = img[:,:,::-1]
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
        #img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = self.io.get_image(path)
        img = img[:,:,::-1]
        img = np.array(img, np.float32)/255.0
        # img must be in BGR order
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
        #img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = self.io.get_image(path)
        img = img[:,:,::-1]
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


class RoadMaskPredictorAzure(object):
    '''Predict road masks.
    '''

    def __init__(self, config):
        config = self.init_config(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(filename=self.config['log_file'], level=logging.INFO)
        self.io = IO_Interface(config)

    def init_config(self, config: dict) -> dict:
        """ Initilize the configuration and create folders
        """
        if config['city_id'][-1] == '/':
            config['city_id'] = config['city_id'][:-1]
        config['log_file'] = os.path.join(config['dataset_root_path'], config['city_id'], '%s.log' % config['city_id'])
        jpg_out_folder = os.path.join(config['dataset_root_path'], config['city_id'], 'overlapping-jpg')
        config['jpg_out_folder'] = jpg_out_folder
        config['geohash_assignment_file_all'] = os.path.join(config['dataset_root_path'], config['city_id'], '%s_gh5_all.csv' % config['city_id'])
        return config

    def generate_dataset(self) -> None:
        solver = TTAFrame(DinkNet34_psp64, self.io)
        weight_path = self.config['weight']
        solver.load(weight_path)
        tic = time()

        jpg_folder = self.config['jpg_out_folder']
        if 'mask_folder' in self.config:
            mask_folder = self.config['mask_folder']
        else:
            mask_folder = jpg_folder
        target = os.path.join(jpg_folder, '..', 'results', os.path.basename(weight_path)[:-3] + '_thresh%0.1f'%self.config['threshold'])
        # if not os.path.exists(target):
        #     os.makedirs(target)
        # handling prediction at geohash level
        gh = self.config['geohash']
        # temporarily don't need to get the geohash_assignment_file
        # if gh:
        #     df = self.io.get_csv_df(self.config['geohash_assignment_file_all'])
        #     arr = df.loc[df.loc[:, 'GeohashAssignment']==gh, 'JPGImageName'].to_list()
        #     val = [os.path.join(jpg_folder, file_name) for file_name in arr]
        # else:
        #     val = self.io.get_files_from_folder(jpg_folder)
        #     val = [p for p in val if os.path.splitext(p)[1]=='.jpg']
        val = self.io.get_files_from_folder(jpg_folder)
        val = [p for p in val if os.path.splitext(p)[1]=='.jpg']
        print(f"Predicting road masks for {len(val)} jpg images for geohash {gh} of city {self.config['city_id']}...")
        self.logger.info(f"Predicting road masks for {len(val)} jpg images for geohash {gh} of city {self.config['city_id']}...")

        for i,path in enumerate(val[self.config['start_index']:self.config['end_index']]):
            if i%10 == 0:
                print('%d    %.2f' % (i/10, time()-tic))
            name = os.path.basename(path)
            outfile = os.path.join(target, name[:-7]+'mask.png')
            if self.io.file_exists(outfile):
                continue
            
            mask = solver.test_one_img_from_path(path)
            mask[mask>self.config['threshold']] = 255
            mask[mask<=self.config['threshold']] = 0
            mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
            self.io.save_image_to_disk(mask.astype(np.uint8), outfile)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",
                        help="config file path", type=str, required=True)
    return parser.parse_args()


def run_dataset_builder(conf_file: str) -> None:
    conf = dict(io_utils.config_load(conf_file))
    RoadMaskPredictorAzure(conf).generate_dataset()

if __name__ == '__main__':
    args = parse_arguments()
    run_dataset_builder(args.config_file)

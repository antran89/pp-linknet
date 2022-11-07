import torch
import torch.nn as nn
from torch.autograd import Variable as V
# from apex.fp16_utils import FP16_Optimizer

import cv2
import numpy as np

def BN_convert_float(module):
    '''
    Designed to work with network_to_half.
    BatchNorm layers need parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False, fp16 = False, finetune_weights=None):
        self.net = net().cuda()
        # temp stuff
        if finetune_weights:
            state = torch.load(finetune_weights)
            state = {key.replace('module.', ''): value for key, value in state.items()}
            self.net.load_state_dict(state)
            print('Fine-tuning models from %s' % finetune_weights)
        
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.fp16 = fp16
        if fp16:
            raise Exception('We do not handle fp16 for now.')
#             self.net = BN_convert_float(self.net.half())
#             self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
#             self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr, momentum=0.95)
#             self.optimizer = FP16_Optimizer(self.optimizer, dynamic_loss_scale=True)
        else:
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        if self.fp16:
            loss = self.loss(pred.float(), self.mask.float())
            self.optimizer.backward(loss)
        else:
            loss = self.loss(pred, self.mask)
            loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        mylog.write('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def update_poly_lr(self, poly_rate, mylog):
        new_lr = self.old_lr * poly_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        mylog.write('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
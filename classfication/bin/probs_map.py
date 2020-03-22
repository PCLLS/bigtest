import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import logging,openslide
import time,glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from classfication.utils.config import TESTSET

class ProbsMap:
    def __init__(self,dataset,dataloader,save, model, level, tif_folder=TESTSET):
        '''
        dataset: contains all points for prediction
        probs_size: a dict saving all mask dimensions
        save: save path
        level: level
        '''
        self.dataset = dataset
        self.dataloader = dataloader
        self.save = save
        self.model = model
        self.level = level
        self.mag = pow(2,self.level)
        self.tif_folder = tif_folder
        self.queue = []
        self._preprocess()
    
    def _preprocess(self):
        for tif in glog.glob(os.path.join(self.tif_folder,'*/*.tif')):
            slide = openslide.OpenSlide(tif)
            basename = os.path.basename(slide)
            self.dimensions[basename] = (slide.dimensions[0]/self.mag,slide.dimensions[1]/self.mag)
        self.dimensions

    def gen_heatmap(self):
        '''
        outputs: model output wize shape [N,Channel,row,col]
        '''
        for (data,targets,indexes) in self.dataloader:
            data = Variable(data.cuda(async=True), volatile=True)
            outputs = self.model(data)
            n,channel,h,w=outputs.shape
            outputs = F.softmax(outputs[i],dim=1)
            for i in range(outputs.shape[0]):
                slidename, x, y, label = self.dataset.table.loc[index]
                if slidename not in self.queue:
                    if self.queue[-1]:
                        save = os.path.join(self.save,self.queue[-1])
                        np.save(f'{save.npy}',probs_map)
                    probs_map = np.zeros(self.dimensions[slidename])
                    self.queue.append(slidename)
                col, row = x//self.mag, y//self.mag # converted to numpy array,relation row-->_y-->h,col-->_x--->w
                if w>1 and h>1: # is segmentation results n,c,h,w
                    probs_maps[slidename][row-h/2:row+h/2,col-w/2:col+w/2]=outputs[i][1].cpu().data.numpy()
                else: # is binary classfication results
                    probs_map[slidename][row,col]=outputs[i][1].cpu().data.numpy()
        save = os.path.join(self.save,self.queue[-1])
        np.save(f'{save.npy}',probs_map)
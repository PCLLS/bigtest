import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import logging,openslide
import time,glob
import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd
import torch.nn.functional as F
from classfication.utils.config import TESTSET
from classfication.preprocess.wsi_ops import wsi
from openslide import OpenSlide
from classfication.utils.metrics import  Metric
class ProbsMap:
    def __init__(self,dataset,dataloader,save, model, grid_size, tif_folder=TESTSET):
        '''
        dataset: contains all points for prediction
        probs_size: a dict saving all mask dimensions
        save: save path
        grid_size: grid_size
        '''
        self.dataset = dataset
        self.dataloader = dataloader
        self.save = save
        self.model = model
        self.grid_size = grid_size
        self.tif_folder = tif_folder
        self.queue = []
        self._preprocess()
    
    def _preprocess(self):
        '''
        initilize numpy array shape for heatmap
        '''
        self.dimensions={}
        for tif in glob.glob(os.path.join(self.tif_folder,'*/*.tif')):
            slide = openslide.OpenSlide(tif)
            basename = os.path.basename(tif).rstrip('.tif')
            self.dimensions[basename] = (slide.dimensions[1]//self.grid_size,slide.dimensions[0]//self.grid_size)

    def gen_heatmap(self):
        '''
        outputs: model output wize shape [N,Channel,row,col]
        '''
        TP,FP,TN,FN =0,0,0,0
        metric=Metric()
        for (data,targets,indexes) in self.dataloader:
            #data = Variable(data.cuda(async=True), volatile=True)
            data, targets = data.cuda(),targets.cpu()
            outputs = self.model(data).cpu()
            #print ('---checking out put shape',outputs.shape)
            #n,channel,h,w=outputs.shape
            if len(outputs.shape)==4:
                n, c, h, w=outputs.shape
            else:
                n, c = outputs.shape
                h, w = 1, 1
            outputs = F.softmax(outputs,dim=1)[:,1]
            metric.add_data(outputs,targets,indexes)
            for i in range(outputs.shape[0]):
                slidename, x, y, label = self.dataset.table.loc[int(indexes[i])]
                if slidename not in self.queue:
                    if self.queue !=[]:
                        save = os.path.join(self.save,self.queue[-1])
                        np.save(f'{save}.npy',probs_map)
                    probs_map = np.zeros(self.dimensions[slidename])
                    self.queue.append(str(slidename))
                col, row = x//self.grid_size, y//self.grid_size # converted to numpy array,relation row-->_y-->h,col-->_x--->w
                if w>1 and h>1: # is segmentation results n,c,h,w
                    probs_map[row-h/2:row+h/2,col-w/2:col+w/2]=outputs[i][1].cpu().data.numpy()
                else: # is binary classfication results
                    try:
                        probs_map[row,col]=outputs[i,1].cpu().data.numpy()
                    except:
                        print(slidename, x, y, label,col,row)
        save = os.path.join(self.save,self.queue[-1])
        np.save(f'{save}.npy',probs_map)
        logging.info(f"FP:{metric.FP},TP:{metric.TP},TN:{metric.TN},FN:{metric.FN}")
        logging.info(f"sensitive:{metric.get_sensitivity()}\taccuracy:{metric.get_accuracy()}\tprecision:{metric.get_precision()}")
        logging.info(f"F1:{metric.get_F1()}\tspecificity:{metric.get_specificity()}")

class OpHeatmap(object):
    @staticmethod
    def visualization(heatmap,grid_size:int,mask_slide:OpenSlide, origin_slide:OpenSlide,save):
        mask = mask_slide.get_thumbnail((mask_slide.dimensions[0]/grid_size,mask_slide.dimensions[0]/grid_size))
        origin = origin_slide.get_thumbnail((origin_slide.dimensions[0]/grid_size,origin_slide.dimensions[0]/grid_size))
        otsu = wsi.otsu(origin_slide, grid_size)
        fig, axes = plt.subplots(1, 4, figsize=(20, 20), dpi=100)
        axes[0].imshow(heatmap)
        axes[0].set_title('Heatmap')
        axes[1].imshow(np.asarray(mask)[:,:,0])
        axes[1].set_title('GT')
        axes[2].imshow(np.asarray(origin))
        axes[2].set_title('Origin')
        axes[3].imshow(otsu)
        axes[3].set_title('Tissue')
        plt.savefig(f'{save}.png')
        
    @staticmethod
    def pool(heatmap,size:int):
        rows,cols=heatmap.size()
        data = np.zeros((rows//size,cols//size))
        for i in range(rows//size):
            for j in range(cols//size):
                data[i,j]=np.max(heatmap[size*i:size*i+size,size*j:size*j+size])
        return data

    @staticmethod
    def to_csv(heatmap:np.array,start_point:int,interval:int,save:str,threshold=0.5):
        x_start,y_start = start_point
        outfile = open(save, 'w')
        if threshold:
            Xs,Ys = np.where(heatmap>threshold)
        else:
            Xs,Ys = np.where(heatmap>0)
        for i,j in zip(Xs,Ys):
            outfile.write('{:0.5f},{},{}'.format(heatmap[i,j], x_start+i*interval, y_start+j*interval) + '\n')
        outfile.close()

    @staticmethod
    def csv_to_heatmap(csvfile:str,size,interval:int,save:str):
        heatmap = np.zeros(size)
        with open(csvfile,'r')as f:
            for i in f.readlines():
                probs,x,y = i.rstrip().split(',')
                probs,i ,j=float(probs),float(x)//interval,float(y)//interval
                heatmap[i,j]=probs
        np.save(save,heatmap)

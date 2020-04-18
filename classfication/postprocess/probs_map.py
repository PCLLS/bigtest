import sys, os, tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import torch
import logging, openslide
import time, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from classfication.utils.config import TESTSET
from classfication.preprocess.wsi_ops import wsi
from openslide import OpenSlide
from classfication.utils.metrics import Metric


class ProbsMap:
    def __init__(self, dataset, dataloader, save, net, grid_size, out_fn, tif_folder=TESTSET):
        '''
        dataset: contains all points for prediction
        probs_size: a dict saving all mask dimensions
        save: save path
        grid_size: grid_size
        out_fn: outputs --> probs
        '''
        self.dataset = dataset
        self.dataloader = dataloader
        self.save = save
        self.net = net
        self.grid_size = grid_size
        self.tif_folder = tif_folder
        self.queue = []
        self._preprocess()
        self.out_fn = out_fn

    def _preprocess(self):
        '''
        initilize numpy array shape for heatmap
        '''
        self.np_shapes = {}
        for tif in glob.glob(os.path.join(self.tif_folder, '*/*.tif')):
            slide = openslide.OpenSlide(tif)
            basename = os.path.basename(tif).rstrip('.tif')
            self.np_shapes[basename] = (slide.dimensions[1] // self.grid_size, slide.dimensions[0] // self.grid_size)

    def gen_heatmap(self):
        r'''
        outputs: model output wize shape [N,Channel,row,col]
        npy file would be saved in self.save as well as log file with stats
        '''
        logfile = open(os.path.join(self.save, 'log.txt'), 'w')
        TP, FP, TN, FN = 0, 0, 0, 0
        metric = Metric()
        self.net.eval()  # freeze BN
        with torch.no_grad():  # save memory
            qbar = tqdm.tqdm(self.dataloader, dynamic_ncols=True, leave=False)
            for (data, targets, indexes) in qbar:
                # data = Variable(data.cuda(async=True), volatile=True)
                data, targets = data.cuda(), targets.cpu()
                outputs = self.net(data).cpu()
                # print ('---checking out put shape',outputs.shape)
                # n,channel,h,w=outputs.shape
                probs = self.out_fn(outputs)
                metric.add_data(probs, targets, indexes)
                if len(probs.shape) == 3:
                    n, h, w = probs.shape
                elif len(probs.shape) == 1:
                    n = probs.shape[0]
                    w, h = None, None
                probs = probs.numpy()
                for i in range(n):
                    slidename, x, y, label = self.dataset.table.loc[int(indexes[i])]
                    if slidename not in self.queue:
                        if self.queue != []:
                            save = os.path.join(self.save, self.queue[-1])
                            np.save(f'{save}.npy', probs_map)
                        probs_map = np.zeros(self.np_shapes[slidename])
                        self.queue.append(str(slidename))
                    col, row = x // self.grid_size, y // self.grid_size  # converted to numpy array,relation row-->_y-->h,col-->_x--->w
                    if w and h:  # is segmentation results n,c,h,w
                        probs_map[int(row - h / 2):int(row + h / 2), int(col - w / 2):int(col + w / 2)] = probs[i]
                    else:
                        probs_map[row, col] = probs[i]
        save = os.path.join(self.save, self.queue[-1])
        np.save(f'{save}.npy', probs_map)
        logfile.write(f"FP:{metric.FP},TP:{metric.TP},TN:{metric.TN},FN:{metric.FN}\n")
        logfile.write(
            f"sensitive:{metric.get_sensitivity()}\taccuracy:{metric.get_accuracy()}\tprecision1:{metric.get_precision()}\n")
        logfile.write(
            f"F1:{metric.get_F1()}\tspecificity:{metric.get_specificity()}\tprecision2:{metric.get_precision2()}\n")
        logfile.close()


class OpHeatmap(object):
    @staticmethod
    def visualization(heatmap, grid_size: int, mask_slide, level: int, origin_slide, save=None):
        origin = origin_slide.get_thumbnail(
            (origin_slide.dimensions[0] / grid_size, origin_slide.dimensions[1] / grid_size))
        otsu = wsi.otsu(origin_slide, grid_size)
        fig, axes = plt.subplots(1, 4, figsize=(20, 20), dpi=100)
        axes[0].imshow(heatmap)
        axes[0].set_title('Heatmap')
        if mask_slide:
            mask = mask_slide.get_thumbnail((mask_slide.level_dimensions[level]))
            mask = np.asarray(mask)[:, :, 0]
        else:
            mask = np.zeros_like(heatmap)
        axes[1].imshow(mask)
        axes[1].set_title('GT')
        axes[2].imshow(np.asarray(origin))
        axes[2].set_title('Origin')
        axes[3].imshow(otsu)
        axes[3].set_title('Tissue')
        plt.savefig(f'{save}.png')

    @staticmethod
    def pool(heatmap, size: int):
        rows, cols = heatmap.size()
        data = np.zeros((rows // size, cols // size))
        for i in range(rows // size):
            for j in range(cols // size):
                data[i, j] = np.max(heatmap[size * i:size * i + size, size * j:size * j + size])
        return data

    @staticmethod
    def to_csv(heatmap: np.array, start_point: int, interval: int, save: str, threshold=0.5):
        x_start, y_start = start_point
        outfile = open(save, 'w')
        if threshold:
            Xs, Ys = np.where(heatmap > threshold)
        else:
            Xs, Ys = np.where(heatmap > 0)
        for i, j in zip(Xs, Ys):
            outfile.write('{:0.5f},{},{}'.format(heatmap[i, j], x_start + i * interval, y_start + j * interval) + '\n')
        outfile.close()

    @staticmethod
    def csv_to_heatmap(csvfile: str, size, interval: int, save: str):
        heatmap = np.zeros(size)
        with open(csvfile, 'r')as f:
            for i in f.readlines():
                probs, x, y = i.rstrip().split(',')
                probs, i, j = float(probs), float(x) // interval, float(y) // interval
                heatmap[i, j] = probs
        if save:
            np.save(save, heatmap)
        else:
            return heatmap

    @staticmethod
    def gt_to_heatmap(csvfile: str, size, interval: int, save: str):
        heatmap = np.zeros(size)
        with open(csvfile, 'r')as f:
            for line in f.readlines()[1:]:
                index, slidename, x, y, label = line.rstrip().split(',')
                label, i, j = float(label), float(y) // interval, float(x) // interval
                heatmap[int(i), int(j)] = label
        if save:
            np.save(save, heatmap)
        else:
            return heatmap
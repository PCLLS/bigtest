import sys
import os
import argparse
import logging
import json
import time

import torch
import pandas as pd
import tqdm
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from  classfication.utils.metrics import *

class Train:
    def __init__(self,optimizer,net,workspace,criterion, dataset,train_dataloader, eval_dataloader, out_fn=None):
        self.optimizer = optimizer
        self.net = net
        self.criterion = criterion
        self.out_fn = out_fn
        self.workspace = workspace
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.dataset=dataset

    def train_epoch(self):
        self.net.train()
        losses = Counter()
        qbar=tqdm.tqdm(self.train_dataloader, dynamic_ncols=True, leave=False)
        for i,data in enumerate(qbar,0):
            inputs, labels, patch_list = data
            print(f"input size: {inputs.size()}")
            inputs, labels = inputs.cuda(), labels.cpu()
            outputs = self.net(inputs).cpu()
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            qbar.set_description(f'Train Loss:{loss:.4f}')
            losses.addval(loss.item(), len(outputs))
        return losses.avg

    def eval_epoch(self):
        metrics = Metric()
        self.net.eval()
        losses = Counter()
        qbar = tqdm.tqdm(self.eval_dataloader, dynamic_ncols=True, leave=False)
        hard_neg_example=[]
        for i, data in enumerate(qbar, 0):
            inputs, labels, indexes = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.net(inputs).squeeze().cuda()
            loss = self.criterion(outputs, labels)
            if self.out_fn != None:
                outputs = self.out_fn(outputs)
            metrics.add_data(outputs,labels,indexes)
            losses.addval(loss.item(), len(outputs))
        logging.info(f'accuracy:{metrics.get_accuracy()},\n precision:{metrics.get_precision()},\n sensitivity:{metrics.get_sensitivity()},\nF1:{metrics.get_F1()}')
        return metrics, losses.avg

# TODO: 如何导入hard——example的代码还需要仔细思考再去复现
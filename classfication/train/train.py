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
import torch.nn.functional as F
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
            # print(outputs.size())
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            qbar.set_description(f'Train Loss:{loss:.4f}')
            losses.addval(loss.item(), len(outputs))
        return losses.avg

    def eval_epoch(self,save_hard_example=None):
        self.net.eval()
        TPs, FPs, TNs, FNs=[],[],[],[]
        total_pos=0
        total_neg=0
        losses = Counter()
        qbar = tqdm.tqdm(self.eval_dataloader, dynamic_ncols=True, leave=False)
        hard_neg_example=[]
        for i, data in enumerate(qbar, 0):
            inputs, labels, patch_list = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.net(inputs).squeeze().cuda()
            loss = self.criterion(outputs, labels)
            if self.out_fn != None:
                outputs = self.out_fn(outputs)
            tps, fps, tns, fns, pos, neg = acc_metric(outputs, labels, 0.5)
            TPs +=tps
            FPs +=fps
            TNs += tns
            FNs += fns
            total_pos += pos
            total_neg += neg
            losses.addval(loss.item(), len(outputs))
        TP = len(TPs)
        TN = len(TNs)
        total = total_pos + total_neg
        total_acc = (TP + TN) / total
        pos_acc = TP / total_pos
        neg_acc = TN / total_neg
        logging.info(f'pos_acc:{total_pos},neg_acc:{total_neg}, acc:{total_acc} in validation')
        hard_neg_example=pd.DataFrame()
        if save_hard_example:
            hard_neg_example=self.dataset.table.loc[FPs]
        return total_acc, pos_acc, neg_acc,losses.avg,hard_neg_example

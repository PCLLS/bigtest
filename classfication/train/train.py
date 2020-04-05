import sys
import os
import tqdm
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from  classfication.utils.metrics import *

class Train:
    def __init__(self,optimizer,net,workspace,criterion,out_fn=None):
        self.optimizer = optimizer
        self.net = net
        self.criterion = criterion
        self.out_fn = out_fn
        self.workspace = workspace

    def train_epoch(self,train_dataloader):
        self.net.train()
        metrics = Metric()
        qbar=tqdm.tqdm(train_dataloader, dynamic_ncols=True, leave=False)
        for i,data in enumerate(qbar,0):
            inputs, labels, indexes = data
            inputs, labels = inputs.cuda(), labels.cpu()
            outputs = self.net(inputs).cpu()
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            probs = self.out_fn(outputs)
            metrics.add_data(probs, labels, indexes,loss)
            qbar.set_description(f'Train Loss:{loss:.4f},acc:{metrics.get_accuracy():.4f},pos:{metrics.get_precision():.4f},neg:{metrics.get_precision2():.4f}-tumor_ratio:{metrics.tumor_ratio():.4f}')
            outputs = self.out_fn(outputs)
        return metrics

    def eval_epoch(self,eval_dataloader):
        metrics = Metric()
        self.net.eval()
        qbar = tqdm.tqdm(eval_dataloader, dynamic_ncols=True, leave=False)
        hard_neg_example=[]
        for i, data in enumerate(qbar, 0):
            inputs, labels, indexes = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.net(inputs).squeeze().cuda()
            loss = self.criterion(outputs, labels)
            probs = self.out_fn(outputs).cpu()
            metrics.add_data(probs, labels, indexes,loss)
            qbar.set_description(f'accuracy:{metrics.get_accuracy():.4f}, pos:{metrics.get_precision():.4f},neg:{metrics.get_precision2():.4f}-tumor_ratio:{metrics.tumor_ratio():.4f}')
        return metrics

    def hard_epoch(self,hard_dataloader):
        metrics = Metric()
        self.net.eval()
        qbar = tqdm.tqdm(hard_dataloader, dynamic_ncols=True, leave=False)
        hard_neg_example=[]
        for i, data in enumerate(qbar, 0):
            inputs, labels, indexes = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.net(inputs).squeeze().cuda()
            loss = self.criterion(outputs, labels)
            probs = self.out_fn(outputs).cpu()
            metrics.add_data(probs, labels, indexes,loss)
            qbar.set_description(f'accuracy:{metrics.get_accuracy():.4f}, pos:{metrics.get_precision():.4f},neg:{metrics.get_precision2():.4f}-tumor_ratio:{metrics.tumor_ratio():.4f}')
        return metrics

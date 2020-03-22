import sys,os,logging,glob,random,tqdm
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from classfication.utils.config import TRAINSET,MASK_FOLDER
import torch
from torch.optim import *
import torch.nn as nn
import matplotlib.pyplot as plt
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from classfication.modeling.deeplab import DeepLab
from classfication.modeling.sync_batchnorm.replicate import patch_replication_callback
from classfication.preprocess.extract_patches import ExtractPatch
from classfication.data.dataset import MaskDataset
from classfication.data.sampler import RandomSampler
from classfication.utils import Checkpointer
from classfication.utils.loss import SegmentationLosses
workspace = '/root/workspace/renqian/test/deeplab1024/'
logging.basicConfig(level=logging.INFO,filename=os.path.join(workspace,'log.txt'))

level = 3 # working level
patch_size = 1280  # size in working level
crop_size = 1024 #size in working level
# sample_level = 7  # 采样倍数
dataset_path=os.path.join(workspace,'patchlist')

# 读取数据集
normal_csv = glob.glob(os.path.join(dataset_path, 'normal*.csv'))
tumor_csv = glob.glob(os.path.join(dataset_path, 'tumor_*.csv'))
csv_list = normal_csv + tumor_csv
tables=[]
# 此处从生成的patch的csv文件中导入，如果调用extractor，则是从返回值获取即可
qbar = tqdm.tqdm(csv_list)
for csv in qbar:
    qbar.set_description(f'loading csv: {csv}')
    tables.append(pd.read_csv(csv,index_col=0,header=0))
table=pd.concat(tables).reset_index(drop=True)

dataset = MaskDataset(TRAINSET,MASK_FOLDER,level,patch_size * pow(2,level),crop_size * pow(2,level),table) # 训练集所有数据导入
rate=0  # 测试集和验证集比例
## check 测试集


## train set
train_slides = {}
train_normal_slides = normal_csv[:int(len(normal_csv)*(1-rate))]
train_tumor_slides = tumor_csv[:int(len(tumor_csv)*(1-rate))]
train_slides[0] = [os.path.basename(csv).rstrip('.csv')  for csv in train_normal_slides + train_tumor_slides]
train_slides[1] = [os.path.basename(csv).rstrip('.csv')  for csv in train_tumor_slides]
train_sampler=RandomSampler(data_source=dataset,slides=train_slides,num_samples=2000)




# 模型训练参数
LR = 0.001
batch_size=4
num_workers=16
net = DeepLab(num_classes=2, backbone='mobilenet',sync_bn=True, output_stride = 16,freeze_bn=False)
if torch.cuda.device_count() > 1:
  logging.info("Let's use", torch.cuda.device_count(), "GPUs!")
net = nn.DataParallel(net)
patch_replication_callback(net)
out_fn = lambda x: x
train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
# valid_dataloader =  DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
evaluate = False
valid_dataloader = None
# optimizer=SGD(net.parameters(),lr=LR,
#               weight_decay=0.9)
optimizer=Adam(net.parameters(),lr=LR)
start = 0 #    start epoch
end = 3000
criterion = SegmentationLosses(weight=None, cuda=False).build_loss(mode='ce')
train_model_save = os.path.join(workspace,'train','model')
train_visual = os.path.join(workspace,'train','visualization')

#  ===================no need change===================
### 保存模型
if not os.path.exists(train_model_save):
        os.system(f"mkdir -p {train_model_save}")
ckpter = Checkpointer(train_model_save)
ckpt = ckpter.load(start)
last_epoch = -1
if ckpt[0]:
    net.load_state_dict(ckpt[0])
    optimizer.load_state_dict(ckpt[1])
    start = ckpt[2]+1
    last_epoch = ckpt[2]
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    start = ckpt[2]+1
train_writer=SummaryWriter(train_visual)
net.cuda()
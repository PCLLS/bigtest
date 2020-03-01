import sys,os,logging,glob,random,tqdm
import pandas as pd
sys.path.append('../../')
from classfication.bin.config import *
import torch
from torch.optim import *
import torch.nn as nn
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

workspace='/root/workspace/renqian/20200229deeplab/'
logging.basicConfig(level=logging.INFO,filename=os.path.join(workspace,'log.txt'))
patch_size = 1500
crop_size = 1280
sample_level = 10  # 采样倍数
dataset_path=os.path.join('/root/workspace/renqian/20200221test/','patchlist')
win_size=800
extractor=ExtractPatch(tif_folder,mask_folder,sample_level,dataset_path,win_size)
extractor.extract_all_sample_together(10)

# 读取数据集
# dataset_path = os.path.join('/root/workspace/renqian/20200221test/','patchlist')  # 存放dataset表格的文件夹
normal_csv = glob.glob(os.path.join(dataset_path, 'normal*.csv'))
tumor_csv = glob.glob(os.path.join(dataset_path, 'tumor_*.csv'))
csv_list = normal_csv + tumor_csv
tables=[]
qbar=tqdm.tqdm(csv_list)

# 此处从生成的patch的csv文件中导入，如果调用extractor，则是从返回值获取即可
for csv in qbar:
    qbar.set_description(f'loading csv: {csv}')
    tables.append(pd.read_csv(csv,index_col=0,header=0))
table=pd.concat(tables).reset_index(drop=True)
level = 3 # 训练集大小
dataset = MaskDataset(tif_folder,mask_folder,level,patch_size,crop_size,table) # 训练集所有数据导入
# 随机分割验证集和训练集
random.shuffle(normal_csv)
random.shuffle(tumor_csv)
rate=0.2  # 测试集和验证集比例

## 训练集
train_slides = {}
train_normal_slides = normal_csv[:int(len(normal_csv)*(1-rate))]
train_tumor_slides = tumor_csv[:int(len(tumor_csv)*(1-rate))]
train_slides[0] = [os.path.basename(csv).rstrip('.csv')  for csv in train_normal_slides + train_tumor_slides]
train_slides[1] = [os.path.basename(csv).rstrip('.csv')  for csv in train_tumor_slides]
train_sampler=RandomSampler(data_source=dataset,slides=train_slides,num_samples=200)

## 验证集
evaluate=False
valid_slides = {}
valid_normal_slides = normal_csv[:int(len(normal_csv)*rate)]
valud_tumor_slides = tumor_csv[:int(len(tumor_csv)*rate)]
valid_slides[0] =  [os.path.basename(csv).rstrip('.csv')  for csv in valid_normal_slides + valud_tumor_slides ]
valid_slides[1] = [os.path.basename(csv).rstrip('.csv')  for csv in valud_tumor_slides ]
valid_sampler=RandomSampler(data_source=dataset,slides=valid_slides,num_samples=40)


# 模型训练参数
LR = 0.01
device_ids=[0,1,2,3]
batch_size=4
num_workers=4
net = DeepLab(num_classes=2, backbone='resnet',sync_bn=True, output_stride = 16,freeze_bn=False)
net = nn.DataParallel(net,device_ids=device_ids)
patch_replication_callback(net)
net=net.cuda()
out_fn = None
train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_dataloader =  DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
optimizer=SGD(net.parameters(),lr=LR)
start = 0 #    起始epoch
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
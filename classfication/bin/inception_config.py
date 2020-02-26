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

from classfication.models.inceptionv3 import inception_v3
out_fn=None
from classfication.preprocess.extract_patches import ExtractPatch
from classfication.data.dataset import ListDataset
from classfication.data.sampler import RandomSampler
from classfication.utils import Checkpointer

'''
生成结果文件布局：
workspace：
- patchlist: 存放所有提取点的csv
- train ：包含训练保存的模型以及可视化模型 
-- models 
-- visualzation
 
'''

# basic config for training inception_v3
workspace='/root/workspace/renqian/20200221test/'
logging.basicConfig(level=logging.INFO,filename=os.path.join(workspace,'log.txt'))


# 提取数据集参数 （如果已生成csv文件的话，则不用设置，无需调用extractor）
#extract_level = 8 # 提取样本时，采用的倍数
#label_win_size = 150  # label根据的win大小

patch_size = 512
crop_size = 299
level = 0  # 训练图片的倍数
# 读取数据集
dataset_path = os.path.join(workspace,'patchlist')  # 存放dataset表格的文件夹
normal_csv = glob.glob(os.path.join(dataset_path, 'normal*.csv'))
tumor_csv = glob.glob(os.path.join(dataset_path, 'tumor_*.csv'))
csv_list = normal_csv + tumor_csv
tables=[]
qbar=tqdm.tqdm(csv_list)

# 此处从生成的patch的csv文件中导入，如果调用extractor，则是从返回值获取即可
for csv in qbar:
    qbar.set_description(f'loading csv: {csv}')
    tables.append(pd.read_csv(csv,index_col=0))
table=pd.concat(tables).reset_index(drop=True)
dataset = ListDataset(tif_folder,mask_folder,level,patch_size,crop_size,table) # 训练集所有数据导入
# 随机分割验证集和训练集
normal_csv = random.shuffle(normal_csv)
tumor_csv = random.shuffle(tumor_csv)
rate=0.1  # 测试集和验证集比例

## 训练集
train_slides = {}
train_slides[0] = [os.path.basename(csv).rstrip('.csv')  for csv in normal_csv[:int(len(normal_csv)*(1-rate))]]
train_slides[1] = [os.path.basename(csv).rstrip('.csv')  for csv in tumor_csv[:int(len(tumor_csv)*(1-rate))]]
train_sampler=RandomSampler(data_source=dataset,slides=train_slides,num_samples=20000)

## 验证集
valid_slides = {}
valid_slides[0] =  [os.path.basename(csv).rstrip('.csv')  for csv in normal_csv[:int(len(normal_csv)*rate)]]
valid_slides[1] = [os.path.basename(csv).rstrip('.csv')  for csv in tumor_csv[:int(len(tumor_csv)*rate)]]
valid_sampler=RandomSampler(data_source=dataset,slides=valid_slides,num_samples=4000)


# 模型训练参数
LR = 0.01
device_ids=[0,1,2,3]
batch_size=32
num_workers=20
net = nn.DataParallel(inception_v3(True,num_class=2),device_ids=device_ids)
train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_dataloader =  DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
optimizer=SGD(lr=LR)
start = 0 #    起始epoch
end = 3000
loss = nn.CrossEntropyLoss()
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
    net.optim(ckpt[0])
    optimizer.load_state_dict(ckpt[1])
    start = ckpt[2]+1
    last_epoch = ckpt[2]
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    start = ckpt[2]+1
train_writer=SummaryWriter(train_visual)
#  ====================================================
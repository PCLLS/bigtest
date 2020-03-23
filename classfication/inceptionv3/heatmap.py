import sys, os, logging, glob, random, tqdm
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from classfication.utils.config import TESTSET,MASK_FOLDER
import torch.nn as nn
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import Inception3
from classfication.models.inceptionv3 import inception_v3
from classfication.data.dataset import ListDataset
from classfication.postprocess.probs_map import ProbsMap
from classfication.utils import Checkpointer

# basic config for training inception_v3
workspace='/root/workspace/renqian/test/inceptionv3'
logging.basicConfig(level=logging.INFO,filename=os.path.join(workspace,'heatmap_log.txt'))

# 提取数据集参数 （如果已生成csv文件的话，则不用设置，无需调用extractor）
#extract_level = 8 # 提取样本时，采用的倍数
#label_win_size = 150  # label根据的win大小

level = 0  # 训练图片的倍数
patch_size = 299
crop_size = 299
grid_size = 128
# 读取数据集
dataset_path = os.path.join(workspace,'test/patchlist')  # 存放dataset表格的文件夹
test_csv = glob.glob(os.path.join(dataset_path, 'test*.csv'))
csv_list = test_csv
tables=[]
qbar=tqdm.tqdm(csv_list)

# 此处从生成的patch的csv文件中导入，如果调用extractor，则是从返回值获取即可
for csv in qbar:
    qbar.set_description(f'loading csv: {csv}')
    tables.append(pd.read_csv(csv,index_col=0,header=0))
table=pd.concat(tables).reset_index(drop=True)
dataset = ListDataset(TESTSET,MASK_FOLDER,level,patch_size * pow(2,level),crop_size * pow(2,level),table) # 训练集所有数据导入

# print(dataset.table.head())
test_slides = {}
test_slides_csv = test_csv
test_slides[0] = [os.path.basename(csv).rstrip('.csv')  for csv in test_slides_csv]


start = 54
batch_size=8
num_workers= 10

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
train_model_save = '/root/workspace/renqian/test/inceptionv3/train/model'
save = '/root/workspace/renqian/test/inceptionv3/heatmap128/'
ckpter = Checkpointer(train_model_save)
ckpt = ckpter.load(start)
net = nn.DataParallel(Inception3(num_classes=2,aux_logits=False))
net.load_state_dict(ckpt[0])
net = net.cuda()
test_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,shuffle=False)
heatmap = ProbsMap(dataset,test_dataloader,save, net, grid_size, tif_folder=TESTSET)
heatmap.gen_heatmap()

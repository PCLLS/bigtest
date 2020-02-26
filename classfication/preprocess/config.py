import os.path as osp
import os
import logging
import glob
logging.basicConfig(level=logging.INFO)
# only one thing u need to do is setting variable between two comments
data_path="/root/workspace/dataset/CAMELYON16/"  # Camelyon16的数据集路径
camelyon17_type_mask = False # 判断是否是Camelyon17的注释文件

# ================================================no need change=============
train_tumor_tif = osp.join(data_path,'training','tumor')
train_tumor_anno = osp.join(data_path,'training','lesion_annotations')
logging.info(f'train_tumor_anno in{train_tumor_anno}')
logging.info(f'train_tumor_tif in {train_tumor_tif}')
logging.info('coverting tumor annotation into Mask image as tif file')
# ================================================end==========================
# set mask files path
train_mask_path = osp.join(data_path,'training','mask') # 设置生成的Mask的地址

# ================================================no need change==============
if not osp.exists(train_mask_path):
    os.mkdir(train_mask_path)
    logging.debug(f'Mask path: {train_mask_path} doesn\'t exists and is created automatic')
train_tumor_tifs = glob.glob(f'{train_tumor_tif}/*.tif')
# ======================end=====================================================
# set otsu file path


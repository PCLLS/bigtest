from multiprocessing import Pool
import glob,os,sys
sys.path.append('../../')
import os.path as osp
import logging
from classfication.preprocess.generateMask import generate_mask
# deploy code in server
tif_folder='/root/workspace/dataset/CAMELYON16/training/*/'
mask_folder='/root/workspace/dataset/CAMELYON16/mask/'
camelyon17_type_mask = False # 判断是否是Camelyon17的注释文件

# ================generate Mask file==================no need change=============
train_tumor_tif = osp.join(tif_folder,'training','tumor')
train_tumor_anno = osp.join(tif_folder,'training','lesion_annotations')
logging.info(f'train_tumor_anno in{train_tumor_anno}')
logging.info(f'train_tumor_tif in {train_tumor_tif}')
logging.info('coverting tumor annotation into Mask image as tif file')
if not osp.exists(mask_folder):
    os.mkdir(mask_folder)
    logging.debug(f'Mask path: {mask_folder} doesn\'t exists and is created automatic')
    train_tumor_tifs = glob.glob(f'{train_tumor_tif}/*.tif')
    pool = Pool(processes=10)
    for tif in sorted(train_tumor_tifs):
        pool.apply_async(generate_mask,args=(tif,train_tumor_anno,mask_folder,camelyon17_type_mask,))
    pool.close()
    pool.join()
# ======================end=====================================================


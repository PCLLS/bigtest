import sys,time
sys.path.append('../../')
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import sys,os,cv2,glob,tqdm
import numpy as np
import pandas as pd
import logging
from classfication.preprocess.wsi_ops import wsi
#import multiprocessing
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

def Label(x, y, mag,level,win,mask):
    x, y = x * mag - win / 2, y * mag - win / 2
    assert isinstance(mask,OpenSlide)
    label=wsi.read_slide(mask, x, y, level, win, win)[:, :, 0]
    if np.sum(label) > 0:
        return 1
    return 0

def skip_slide(slide_name):
    skip_list = ['normal_86', 'normal_144', 'test_049', 'test_114']
    for skip_name in skip_list:
        if skip_name in slide_name:
            return True
    return False

class ExtractPatch:
    def __init__(self,tif_folder,mask_folder,level,save_path,win_size,otsu_folder=None,otsu_level=0):
        '''
        Use pandas.Framework to save result
        :param otsu_folder: contain tumor_001.npy
        :param mask_folder: contain mask.tif
        :param level: otsu level
        :param save_path:
        :param win_size:
        '''
        # assert (os.path.exists(tif_folder) or os.path.exists(otsu_folder)) and os.path.exists(mask_folder)==True
        self.otsu_folder=otsu_folder
        self.tif_folder=tif_folder
        self.mask_folder = mask_folder
        self.level = level
        self.save_path = save_path
        self.win_size = win_size
        self.table=pd.DataFrame(columns=['slide_name','x','y','label'])
        self._preprocess_mask()
        if otsu_folder:
            self._preprocess_otsu()
            self.otsu_level=otsu_level
        else:
            self._preprocess_tif()

    def _preprocess_otsu(self):
        ''' 如果使用OTSU。npy文件的话'''

        logging.info('read otsu')
        self.otsu_dict={}
        pbar =  tqdm.tqdm(glob.glob(os.path.join(self.otsu_folder,'*.npy')))
        for otsu in pbar:
            pbar.set_description("Processing otsu: %s" % otsu)
            if skip_slide(otsu):
                continue
            _basename = os.path.basename(otsu).rstrip('.npy')
            self.otsu_dict[_basename]=np.load(otsu)
        logging.info('read mask')

    def _preprocess_mask(self):
        self.mask_dict={}
        pbar = tqdm.tqdm(glob.glob(os.path.join(self.mask_folder,'*.tif')))
        for gt_mask in pbar:
            pbar.set_description("Processing mask %s" % gt_mask)
            if skip_slide(gt_mask):
                continue
            _basename = os.path.basename(gt_mask).rstrip('.tif')
            self.mask_dict[_basename]=OpenSlide(gt_mask)

    def _preprocess_tif(self):
        logging.info('detecting ROI')
        self.otsu_dict={}
        pbar = tqdm.tqdm(glob.glob(self.tif_folder+'/*/*.tif'))
        for tif in pbar:
            pbar.set_description("Processing tif %s" % tif)
            if skip_slide(tif):
                continue
            _basename = os.path.basename(tif).rstrip('.tif')
            self.otsu_dict[_basename] =wsi.otsu_rgb(OpenSlide(tif),self.level)

    def extract_from_single_slide(self,slide):
        logging.info(f'extract samples from{slide}')
        table=pd.DataFrame(columns=['slide_name','x','y','label'])
        mag = pow(2, self.level)
        count = 0
        otsu=self.otsu_dict[slide]
        _x, _y = np.where(otsu > 0)
        mask = self.mask_dict.get(slide, None)
        for x, y in zip(_x, _y):
            label = 0
            if mask:
                label = Label(x, y, mag, self.level, self.win_size, mask)
            x, y = int(x * mag), int(y * mag)
            table.loc[count] = (slide, x, y, label)
            count += 1
        save= os.path.join(self.save_path,f'{slide}.csv')
        table.to_csv(save, header=True)
        logging.info(f'samples from {slide} done!!')
        return table

    def extract_all_sample_together(self,num_works=10):
        '''
        Multithread method
        :param num_works:
        :return:
        '''
        pool = ThreadPoolExecutor(max_workers=num_works)
        futures = []
        for slide in self.otsu_dict.keys():
            future=pool.submit(self.extract_from_single_slide,slide)
            futures.append(future)
        pool.shutdown(True)
        # results=[]
        # for future in futures:
        #     results.append(future.result())
        # logging.info('save tables')
        # self.table = pd.concat(results)
        # self.table = self.table.reset_index(drop=True)
        # save = os.path.join(self.save_path, 'allsample.csv')
        # self.table.to_csv(save, header=True)

    def extract_all_sample(self):
        logging.info('extract_all_sample')
        mag = pow(2, self.level)
        count=0
        pbar=tqdm.tqdm(self.otsu_dict.items())
        for slide,otsu in pbar:
            pbar.set_description("Extracting sample from %s.tif" % slide)
            logging.info(f'extracting samples from {slide}')
            _x, _y = np.where(otsu > 0)
            mask = self.mask_dict.get(slide,None)
            # data = [(i, j) for i, j in zip(x, y)]
            for x,y in zip(_x,_y):
                label = 0
                if mask:
                    label = Label(x, y, mag, self.level, self.win_size, mask)
                x, y = int(x*mag), int(y*mag)
                self.table.loc[count]=(slide, x, y, label)
                count+=1
        save = os.path.join(self.save_path,'allsample.csv')
        logging.info(f'save samples in {self.save_path}')
        self.table.to_csv(self.save_path, header=True)

if __name__=='__main__':
    tif_folder = '/root/workspace/renqian/CAMELYON16/training/'
    mask_folder='/root/workspace/renqian/CAMELYON16/mask/'
    level=10
    save_path='/root/workspace/renqian/20200301deeplab/patchlist/'
    win_size=800
    extractor=ExtractPatch(tif_folder,mask_folder,level,save_path,win_size)
    extractor.extract_all_sample_together(10)
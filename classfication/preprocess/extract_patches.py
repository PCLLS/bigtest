import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import cv2,glob,tqdm
import numpy as np
import pandas as pd
import logging
from classfication.preprocess.wsi_ops import wsi
from classfication.utils.config import TRAINSET,MASK_FOLDER,TESTSET
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,ALL_COMPLETED
import concurrent.futures
import argparse
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def Label(mask, w, h, level,  win):
    '''
    :param w,h: center point in level 0
    :param level: 
    :param win: in level 0
    :param mask:OpenSlide
    :return:
    '''
    if mask==0:
        return mask
    assert isinstance(mask, OpenSlide)
    mask=wsi.read_slide(mask, w, h, level, win, win)
    return int(mask.sum() > 0)


def skip_slide(slide_name):
    skip_list = ['normal_86', 'normal_144', 'test_049', 'test_114']
    for skip_name in skip_list:
        if skip_name in slide_name:
            return True
    return False

def extract_tumor(slide_name):
    skip_list = []
    return True

class ExtractPatch:
    def __init__(self,tif_folder,mask_folder,level,save_path,win_size,StepSize=128,grid_size=128,otsu_level=0):
        '''
        Use pandas.Framework to save result
        :param otsu_folder: contain tumor_001.npy
        :param mask_folder: contain mask.tif
        :param level: otsu level
        :param save_path:
        :param win_size:
        '''
        # assert (os.path.exists(tif_folder) or os.path.exists(otsu_folder)) and os.path.exists(mask_folder)==True
        self.tif_folder=tif_folder
        self.mask_folder = mask_folder
        self.level = level 
        self.step_size=StepSize 
        self.grid_size=grid_size 
        self.save_path = save_path 
        self.win_size = win_size 
        self._load_mask()
        self.otsu_level=otsu_level 
        self._load_tif()

    def _load_mask(self):
        self.mask_dict={}
        pbar = tqdm.tqdm(glob.glob(os.path.join(self.mask_folder,'*.tif')))
        for gt_mask in pbar:
            if skip_slide(gt_mask):
                continue
            _basename = os.path.basename(gt_mask).rstrip('.tif')
            pbar.set_description(f"Processing mask {gt_mask} - {_basename}" )
            self.mask_dict[_basename]=gt_mask


    def _load_tif(self):
        logging.info('detecting ROI')
        self.tifs={}
        pbar = tqdm.tqdm(glob.glob(self.tif_folder+'/*/*.tif'))
        for tif in pbar:
            if skip_slide(tif):
                continue
            _basename = os.path.basename(tif).rstrip('.tif')
            pbar.set_description(f"Processing tif {tif} - {_basename}" )
            self.tifs[_basename]=tif


    def extract_from_single_slide(self,slidename):
        '''
        Sample Patch with limited stride for tissue
        :param slide:
        read_slide: return im (height, width X C)
        :return: stats about patch points
        '''
        # logging.info(f'extract samples from{slide}')
        save= os.path.join(self.save_path, f'{slidename}.csv')
        if os.path.exists(save):
            table=pd.read_csv(save,index_col=0,header=0)
            stats[0]=(table['label']==0).sum()
            stats[1]=(table['label']==1).sum()
            return stats
        table=pd.DataFrame(columns=['slide_name','x','y','label'])
        stats = {0:0,1:0}
        count = 0
        try:
            mask = OpenSlide(self.mask_dict[slidename])
        except:
            logging.info(f'mask {slidename} not exists!!')
            mask = 0
        slide=OpenSlide(self.tifs[slidename])
        W,H=slide.dimensions 
        x_center = self.grid_size // 2  
        for col in range(int((W-self.grid_size/2)//self.step_size)):
            y_center=self.grid_size// 2 
            for row in range(int((H-self.grid_size/2)//self.step_size)):
                otsu,white_flag=wsi.read_otsu(slide, x_center, y_center, self.otsu_level, self.win_size, self.win_size,  white_flag=True)
                if not white_flag:
                    label = Label(mask,x_center, y_center, self.level, win=self.win_size)
                    table.loc[count] = (slidename, x_center, y_center, label)
                    count += 1
                    stats[label] +=  1
                y_center += self.step_size 
            x_center += self.step_size  
        
        logging.info(f'samples {str(stats)} from {slidename} done!!')
        if mask != 0 and stats[1]==0:
            logging.warning(f'{slidename} cannot find tumor')
        table.to_csv(save, header=True)
        return stats
    

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--num_works",type=int,default=20,help='num of workers')
    parser.add_argument("-ol",'--otsu_level',type=int,default=0,help="otsu level")
    parser.add_argument("-l",'--level',type=int,default=0,help="work level")
#     parser.add_argument("-tf",'--tif_folder',type=str,default=TRAINSET,help="tif folder")
    parser.add_argument("-mf","--mask_folder",type=str,default=MASK_FOLDER,help="Mask folder")
    parser.add_argument("-w","--win_size",type=int,help="win size in level 0")
    parser.add_argument("-ss","--StepSize",type=int,help="stride in level 0")
    parser.add_argument("-g","--grid_size",type=int,help="grid_size,as same as win size proposed")
    parser.add_argument("-s","--save",type=str,help="save path")
    parser.add_argument("-t","--test",action='store_true',help="train mode or test mode")
    return parser.parse_args()


if __name__=='__main__':
    args=get_arg()
    num_works=args.num_works
    mask_folder = args.mask_folder
    level = args.level
    save_path= args.save
    win_size = args.win_size
    otsu_level = args.otsu_level
    stepsize = args.StepSize
    grid_size = args.grid_size
    if not os.path.exists(save_path):
        os.system(f'mkdir -p {save_path}')
    if args.test:
        tif_folder=TESTSET
    else:
        tif_folder=TRAINSET
    logging.basicConfig(level=logging.INFO,filename=os.path.join(save_path,'log.txt'))
    logging.info(str(args))
    extractor = ExtractPatch(tif_folder, mask_folder, level, save_path, StepSize=stepsize, grid_size=grid_size,win_size=win_size, otsu_level=otsu_level)
    with ProcessPoolExecutor(max_workers=num_works) as pool:
        futures= [pool.submit(extractor.extract_from_single_slide, slide) for slide in extractor.tifs.keys()]
    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    stat={0:[], 1:[]}
    for future in concurrent.futures.as_completed(futures):
        stat[0].append(future.result().get(0, 0))
        if future.result().get(1,0):
            stat[1].append(future.result()[1])
    stat[0]=np.array(stat[0])
    stat[1]=np.array(stat[1])
    # Visualize status of sampler
    figs, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].boxplot([stat[0], stat[1]], labels=['nomral', 'tumor'])
    axs[0].set_title('distribution of numbers of patches')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].text(0, 0.5, s=f'normal: {stat[0].sum():.2f}\nmean:{stat[0].mean():.2f}\nstd:{stat[0].std():.2f}\ntumor:{stat[1].sum():.2f}\nmean:{stat[1].mean():.2f}\nstd:{stat[1].std():.2f}')
    plt.savefig(os.path.join(save_path,'boxplot.png'))
    print(f"statistic :{stat}")
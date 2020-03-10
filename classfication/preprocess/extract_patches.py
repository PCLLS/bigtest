import sys,time
sys.path.append('../../')
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import sys,os,cv2,glob,tqdm
import numpy as np
import pandas as pd
import logging
from classfication.preprocess.wsi_ops import wsi
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,ALL_COMPLETED
import concurrent.futures
import argparse
import matplotlib.pyplot as plt
def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","-num_works",type=int,default=20,help='num of workers')
    parser.add_argument("-ol",'-otsu_level',type=int,default=0,help="otsu level")
    parser.add_argument("-l",'-level',type=int,default=0,help="work level")
    parser.add_argument("-tf",'-tif_folder',type=str,help="tif folder")
    parser.add_argument("-mf","-mask_folder",type=str,help="Mask folder")
    parser.add_argument("-w","-win_size",type=int,help="win size")
    parser.add_argument("-ss","-StepSize",type=int,help="stride")
    parser.add_argument("-p","-patch_size",type=int,help="patch size")
    parser.add_argument("-s","-save",type=str,help="save path")
    return parser.parse_args()

def Label(w, h, mag,level,  win, mask):
    if mask==0:
        return mask
    w, h = w * mag - win / 2, h * mag - win / 2
    assert isinstance(mask, OpenSlide)
    label=wsi.read_mask(mask, w, h, level, win, win,as_numpy=True)
    return int(np.sum(label) > 0)


def skip_slide(slide_name):
    skip_list = ['normal_86', 'normal_144', 'test_049', 'test_114']
    for skip_name in skip_list:
        if skip_name in slide_name:
            return True
    return False

class ExtractPatch:
    def __init__(self,tif_folder,mask_folder,level,save_path,win_size,StepSize=229,patch_size=229,otsu_level=0):
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
        self.level = level # 工作level，如inceptionV3在level 0，
        self.step_size=StepSize # 滑块滑动步长
        self.patch_size=patch_size # 滑块在level-0的大小
        self.save_path = save_path # 存储路径
        self.win_size = win_size # 标签大小
        self.table=pd.DataFrame(columns=['slide_name','x','y','label'])
        self._load_mask()
        self.otsu_level=otsu_level # 滑块所处的level
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
        Ref:linhj
        思路： 基于step, patch切割图片，然后计算OTSU和是否合适用于分析，去掉含有大量脂肪区域的patch
        :param slide:
        read_slide: return im (height, width X C)
        :return:
        '''
        # logging.info(f'extract samples from{slide}')
        table=pd.DataFrame(columns=['slide_name','x','y','label'])
        stats = {}
        count = 0
        try:
            mask = OpenSlide(self.mask_dict[slidename])
        except:
            logging.info(f'mask {slidename} not exists!!')
            mask = 0
        slide=OpenSlide(self.tifs[slidename])
        W,H=slide.level_dimensions[self.level]
        mag = slide.level_downsamples[self.level]
        w = self.patch_size // 2  # 取中间点
        for col in range(int((W-self.patch_size/2)//self.step_size)):
            h=self.patch_size// 2 # 取中间点
            for row in range(int((H-self.patch_size/2)//self.step_size)):
                # 计算OTSU
                otsu,white_flag=wsi.read_otsu(slide,w,h,self.otsu_level,self.win_size,self.win_size, white=True)
                # 过滤掉白色区域
                label = Label(w, h, mag, self.level, self.win_size, mask)
                if not white_flag or label :
                    table.loc[count] = (slidename, w, h, label)
                    count += 1
                    stats[label] = stats.get(label, 0) + 1
                h += self.step_size # 下移
            w += self.step_size  # 右移
        save= os.path.join(self.save_path, f'{slidename}.csv')
        logging.info(f'samples {str(stats)} from {slidename} done!!')
        if mask != 0 and stats[1]==0:
            logging.warning(f'{slidename} cannot find tumor')
        table.to_csv(save, header=True)
        return stats




if __name__=='__main__':
    args=get_arg()
    print(args)
    num_works=args.n
    tif_folder = args.tf
    mask_folder = args.mf
    level = args.l
    save_path= args.s
    win_size = args.w
    otsu_level = args.ol
    stepsize = args.ss
    patch_size = args.p
    logging.basicConfig(level=logging.INFO,filename=os.path.join(save_path,'log.txt'))
    logging.info(f'config otsu level-{otsu_level},save:{save_path},Patch win_size:{win_size} level-{level}')
    if not os.path.exists(save_path):
        os.system(f'mkdir -p {save_path}')
    extractor = ExtractPatch(tif_folder, mask_folder, level, save_path, StepSize=stepsize, patch_size=patch_size,
                             win_size=win_size, otsu_level=otsu_level)
    with ProcessPoolExecutor(max_workers=num_works) as pool:
        futures= [pool.submit(extractor.extract_from_single_slide, slide) for slide in extractor.tifs.keys()]
    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    stat={0:[], 1:[]}
    for future in concurrent.futures.as_completed(futures):
        stat[0].append(future.result().get(0, 0))
        if futures.get(1,0):
            stat[1].append(future.result()[1])
    stat[0]=np.array(stat[0])
    stat[1]=np.array(stat[1])
    # 可视化结果
    figs, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].boxplot([stat[0], stat[1]], labels=['nomral', 'tumor'])
    axs[0].set_title('distribution of numbers of patches')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].text(0, 0.5, s=f'normal: {stat[0].sum()}\nmean:{stat[0].mean()},std:{stat[0].std()}\n \
                tumor:{stat[1].sum()}\nmean:{stat[1].mean()},std:{stat[1].std()}')
    plt.savefig(os.path.join(save_path,'boxplot.png'))
    print(f"statistic :{stat}")
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import os
import cv2
import multiprocessing
import numpy as np
class wsi(object):
    @staticmethod
    def read_slide(slide, x_center, y_center, level, width, height, as_numpy=False):
        '''

        :param slide:
        :param x_center,y_center : center location in level 0
        :param y:
        :param level: level
        :param width,height: in level 0
        :param : 
        :param as_float:
        :return: im (height, width X C)
        '''
        x, y = int(x_center - width/2) , int(y_center - height/2)  
        mag = pow(2,level)
        width, height= int(width/mag), int(height/mag) # scale width and height
        im = slide.read_region((x, y), level, (width, height)).convert('RGB')
        if as_numpy:
            im = np.asarray(im)
        else:
            im = np.asarray(im)
        assert im.shape == (height, width, 3)
        return im

    @staticmethod
    def otsu(slide,mag,save=None):
        '''
        Provide by 黄
        :param slide:
        :param level:
        :return:
        '''
        assert isinstance(slide, openslide.OpenSlide)
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / mag, slide.dimensions[1] / mag)).convert('RGB')
        img_RGB = np.array(thumbnail)
        img_HSV = rgb2hsv(img_RGB)
        background_R = img_RGB[:, :, 0] > 203
        background_G = img_RGB[:, :, 1] > 191
        background_B = img_RGB[:, :, 2] > 201
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > 0.1113
        '''如果仅使用用threadshold，中间会有部份白色脂肪区域被隔离'''
        rgb_min = 50
        min_R = img_RGB[:, :, 0] > rgb_min
        min_G = img_RGB[:, :, 1] > rgb_min
        min_B = img_RGB[:, :, 2] > rgb_min
        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
        return tissue_mask
    
    @staticmethod
    def find_tissue(slide,x, y,level,width,height):
        assert isinstance(slide, openslide.OpenSlide)
        x_,y_ = int(x - width//2 ),int(y - height//2) 
        tissue_mask,_=wsi.read_otsu(slide,x, y,level,width,height,white_flag=False)
        X,Y = np.where(tissue_mask==1)
        return [(x_+x*mag, y_+y*mag)for x,y in zip(X,Y)]

    @staticmethod
    def read_otsu(slide,x_center, y_center,level,width,height,white_flag=True):
        '''
        :param x_center, y_center:
        :level : otsu sample level 
        width,height: w,h in level 0
        return tissue_mask:W*H*C
        white_flag = True
        '''
        assert isinstance(slide, openslide.OpenSlide)
        img_RGB = wsi.read_slide(slide, x_center, y_center, level, width, height)
        img_HSV = rgb2hsv(img_RGB)
        background_R = img_RGB[:, :, 0] > 203
        background_G = img_RGB[:, :, 1] > 191
        background_B = img_RGB[:, :, 2] > 201
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > 0.1113
        '''remove white fat tissue'''
        rgb_min = 50
        min_R = img_RGB[:, :, 0] > rgb_min
        min_G = img_RGB[:, :, 1] > rgb_min
        min_B = img_RGB[:, :, 2] > rgb_min
        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
        # using White flags white region
        if white_flag:
            if tissue_mask.sum()!=0:
                white_flag=False
        return tissue_mask,white_flag

    @staticmethod
    def read_mask(slide, x_center, y_center, level, width, height, as_numpy=False):
        '''
        :param slide:
        :param x_center, y_center: 
        :param level: sample level 
        :param width,height: in level 0
        :param as_float:
        :return: im 
        '''
        assert isinstance(slide, openslide.OpenSlide)
        im = wsi.read_slide(slide, x_center, y_center, level, width, height)
        return im[:, :, 0]
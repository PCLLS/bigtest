import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import os
import cv2
import multiprocessing
import numpy as np
class wsi(object):
    @staticmethod
    def read_slide(slide, x, y, level, width, height, as_numpy=False):
        '''

        :param slide:
        :param x:
        :param y:
        :param level:
        :param width:
        :param height:
        :param as_float:
        :return: im (W X H X C)
        '''
        mag = pow(2, level)
        x = int(x - width * mag / 2)
        y = int(y - width * mag / 2)
        im = slide.read_region((x, y), level, (width, height)).convert('RGB')
        if as_numpy:
            im = np.asarray(im)
        else:
            im = np.asarray(im)
        assert im.shape == (height, width, 3)
        return im

    @staticmethod
    def otsu_rgb(slide,level,save=None):
        '''
        Provide by 黄
        :param slide:
        :param level:
        :return:
        '''
        mag=pow(2,level)
        assert isinstance(slide,openslide.OpenSlide)
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
import os,glob,torch
import openslide
import logging
import numpy as np
import math,random
from torchvision import transforms
from classfication.preprocess.wsi_ops import wsi
from PIL import Image
import pandas as pd
import pdb
class MaskDataset():
    def __init__(self,tif_folder,mask_folder,level,patch_size,crop_size,table):
        """
        Dataset for Mask
        :param list_file: cords file. tif_name,x,y
        :param tif_folder: /Camelyonfolder/
        :param mask_folder: /CamelyonMaskfolder/
        :param level:
        :param patch_size: patch_size
        :param transform: transform
        """
        self.table=table
        self.tif_folder=tif_folder
        self.mask_folder=mask_folder
        self.level=level
        self.patch_size=patch_size
        self.crop_size = crop_size
        self._totensor=transforms.ToTensor()
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._preprocess()

    def _preprocess(self):
        tif_list = glob.glob(os.path.join(self.tif_folder, '*.tif'))
        if tif_list==[]:
            raise ValueError('tif folder should include tif files.')
        logging.info(f"loading tifs from {self.tif_folder}")
        mask_list = glob.glob(os.path.join(self.mask_folder, '*.tif'))
        if tif_list == []:
            raise ValueError('mask folder should include mask files(.tif).')
        self.patch_size = self.patch_size
        self.level=self.level
        # 添加所有的slide缓存，从缓存中取数据
        self.slide_dict = {}
        for tif in tif_list:
            basename = os.path.basename(tif).rstrip('.tif')
            self.slide_dict[basename] = openslide.OpenSlide(tif)
        self.mask_dict={}
        for mask in mask_list:
            basename = os.path.basename(mask).rstrip('.tif')
            self.mask_dict[basename]=openslide.OpenSlide(mask)


    def __len__(self):
        """

        :param self:
        :return: length of cords
        """
        return self.table.shape[0]

    def __getitem__(self,index):
        '''
        :param index:
        :return: img (C x H x W),target
        '''
        slide_name,_x,_y = self.table[index]
        slide =  self.slide_dict[slide_name]
        img = wsi.read_slide(slide,_x,_y,self.level,self.patch_size,self.patch_size) # numpy.array
        try:
            mask = self.mask_dict[slide_name]
            target = wsi.read_slide(mask,_x,_y,0,self.patch_size,self.patch_size) #numpy.array
        except:
            target = np.zeros_like(img)
        # data augmentation
        img = Image.fromarray(img)
        target = Image.fromarray(target)
        img, target = self._random_crop(img,target)
        img = self._color_jitter(img)
        img,target = self._random_flip(img,target)
        img,target = self._random_rotate( img, target)
        # 取最后一个通道
        img, target = self._totensor(img),self._totensor(target)[0,:,:].unsqueeze(0)
        return img,target,index

    def _random_flip(self,img,target):
        # use lefy_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            taget = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img,target

    def _random_rotate(self,img,target):
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)
        target = target.rotate(90 * num_rotate)
        return img, target


    def _random_crop(self,img,target):
        '''

        :param img: PIL
        :param target: PIL
        :return:
        '''

        xloc = np.random.randint(0,self.patch_size-self.crop_size)
        yloc = np.random.randint(0,self.patch_size-self.crop_size)
        img = img.crop(xloc,yloc,xloc+self.crop_size,yloc+self.crop_size)
        target = target.crop((xloc,yloc,xloc+self.crop_size,yloc+self.crop_size))
        return img,target

class ListDataset():
    def __init__(self,tif_folder,mask_folder,level,patch_size,crop_size,table):
        '''

        :param list_file:
        :param tif_folder:
        :param mask_folder:
        :param level:
        :param patch_size:
        :param crop_size:
        '''
        self.table=table
        self.tif_folder = os.path.abspath(tif_folder)
        self.mask_folder = mask_folder
        self.level = level
        self.patch_size = patch_size
        self.crop_size = crop_size
        self._preprocess()
        self._totensor = transforms.ToTensor()
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)

    def __len__(self):
        """

        :param self:
        :return: length of cords
        """
        return self.table.shape[0]

    def _preprocess(self):
        '''

        :return:
        '''

        tif_list = glob.glob(os.path.join(self.tif_folder, '*.tif'))
        if tif_list==[]:
            raise ValueError('tif folder should include tif files.')
        logging.info(f"loading tifs from {self.tif_folder}")
        # 添加所有的slide缓存，从缓存中取数据
        self.slide_dict = {}
        for tif in tif_list:
            basename = os.path.basename(tif).rstrip('.tif')
            self.slide_dict[basename] = tif
            # self.slide_dict[basename] = openslide.OpenSlide(tif)

    def __getitem__(self, item):
        slide_name,_x,_y,target= self.table.loc[item]
        try:
            slide = openslide.OpenSlide(self.slide_dict[slide_name])
        except:
            raise ValueError(f'{slide_name}.tif not exists!')
        img = wsi.read_slide(slide,_x,_y,self.level,self.patch_size,self.patch_size)
        img = Image.fromarray(img)
        img = self._random_crop(img)
        img =  self._color_jitter(img)
        img = self._random_rotate(img)
        img = self._totensor(img)
        return img, target, item


    def _random_crop(self,img):
        '''

        :param img: PIL
        :param target: PIL
        :return:
        '''
        xloc = np.random.randint(0,self.patch_size-self.crop_size)
        yloc = np.random.randint(0,self.patch_size-self.crop_size)
        img = img.crop((xloc,yloc,xloc+self.crop_size,yloc+self.crop_size))
        return img

    def _random_rotate(self,img):
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)
        return img
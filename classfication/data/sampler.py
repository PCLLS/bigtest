import torch
import random
from torch.utils.data import  Sampler
from torch._six import int_classes as _int_classes
import numpy as np
class RandomSampler(Sampler):
    r"""Samples elements randomly. sample Tumor and normal with equal prob. sample from each slide with equal prob.
    ref: 《Detecting Cancer Metastases on Gigapixel Pathology Images》
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``. 用于设置每轮实际采样数
    """

    def __init__(self, data_source, slides=None,num_samples=None,labels=[0,1],query_hist=None):
        '''
        :param data_source:  需要处理的dataset
        :param slides (dict):  用于区分开tumor和normal的数据集，将WSI分为验证集和训练集 ,如果为空则不进行区分
        :param num_samples:  采样数，控制每轮采样数
        '''
        self.data_source = data_source
        self._num_samples = num_samples
        self.slides=slides
        self._sampled=set()
        self.labels=labels
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
        if query_hist:
            self.query_hist=query_hist
        else:
            self.query_hist = {} # pre-query dataset and this method can speed up sample
            for label in labels:
                query = data_source.table.query(f'(label=={label})')
                for slide in self.slides[label]:
                    indexes=query[query['slide_name'] == slide].index
                    self.query_hist[f'{slide}_{label}'] = indexes.tolist()
        self.sample()

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def sample(self):
        self.indexes=[]
        for label in self.labels:
            for i in range(self.num_samples//2):
                slide = np.random.choice(self.slides[label])
                indexes = self.query_hist[f'{slide}_{label}']
                index = np.random.choice(indexes)
                self.indexes.append(index)
        random.shuffle(self.indexes)
        return self.indexes

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.num_samples


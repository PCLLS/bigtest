import torch
import random
from torch.utils.data import  Sampler
import numpy as np

class Sample(object):
    def __init__(self,num,func):
        self.num=num
        self.func=func
        self.current_index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.current_index<self.num:
            self.current_index +=1
            return self.func()
        else:
            self.current_index = 0
            raise StopIteration

class RandomSampler(Sampler):
    r"""Samples elements randomly. sample Tumor and normal with equal prob. sample from each slide with equal prob.
    ref: 《Detecting Cancer Metastases on Gigapixel Pathology Images》
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``. 用于设置每轮实际采样数
    """

    def __init__(self, data_source, slides=None,num_samples=None,labels=[0,1]):
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
        self.query_hist = {} # pre-query this can speed up sample
        for label in labels:
            query = self.data_source.table.query(f'(label=={label})')
            for slide in self.slides[labels]:
                self.query_hist[(label,slide)]=query[query['slide_name'] == slide].index

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def sample(self):
        # random choice labels and slides
        label = np.random.choice(self.labels)
        slide = np.random.choice(self.slides[label])
        indexes = self.query_hist[(label,slide)]
        sample_index = np.random.choice(indexes)
        return sample_index

    def __iter__(self):
        samples=Sample(self.num_samples,self.sample)
#        samples=self.sample()  # 每次只采样一部分减少推断时间
        return samples

    def __len__(self):
        return self.num_samples
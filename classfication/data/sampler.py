import torch
import random
from torch.utils.data import  Sampler
from torch._six import int_classes as _int_classes
import numpy as np

class balanceSample(object):
    def __init__(self,num,func):
        self.num=num
        self.func=func
        self.current_index = 0

    def __iter__(self):
        return self
    def __next__(self):
        if self.current_index<self.num:
            return self.func(self.current_index%2)
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
        self.query_hist = {} # pre-query dataset and this method can speed up sample
        for label in labels:
            query = data_source.table.query(f'(label=={label})')
            for slide in self.slides[label]:
                indexes=query[query['slide_name'] == slide].index
                self.query_hist[f'{slide}_{label}'] = indexes

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def sample(self,label):
        # random choice labels and slides
        # label = np.random.choice(self.labels)
        slide = np.random.choice(self.slides[label])
        indexes = self.query_hist[f'{slide}_{label}']
        index = np.random.choice(indexes)
        return index

    def __iter__(self):
        samples = balanceSample(self.num_samples,self.sample)
        print(samples.current_label)
        return samples

    def __len__(self):
        return self.num_samples

class BalancedBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self,  data_source,batch_size, drop_last, slides=None, num_samples=None, labels=[0,1] ):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_source = data_source
        self.data_source = data_source
        self.num_samples = num_samples
        self.slides = slides
        self._sampled = set()
        self.labels = labels
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
        self.query_hist = {}  # pre-query dataset and this method can speed up sample
        for label in labels:
            query = data_source.table.query(f'(label=={label})')
            for slide in self.slides[label]:
                indexes = query[query['slide_name'] == slide].index
                self.query_hist[f'{slide}_{label}'] = indexes

    def sample(self,label):
        slide = np.random.choice(self.slides[label])
        indexes = self.query_hist[f'{slide}_{label}']
        index = np.random.choice(indexes)
        return index

    def __iter__(self):
        batch = []
        current_label=0
        for idx in range(self.batch_size):
            idx = self.sample(1-current_label)
            batch.append(idx)
            current_label = idx
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.num_samples) // self.batch_size
        else:
            return (len(self.num_samples) + self.batch_size - 1) // self.batch_size

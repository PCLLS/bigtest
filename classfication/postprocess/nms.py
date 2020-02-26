import numpy as np
from skimage import filters
class NMS(object):
    def __init__(self,level,radius=12,sigma=0,prob_thred=0.5):
        '''
        设置通用NMS处理代码
        :param level:
        :param radius:
        :param sigma:
        :param prob_thred:
        '''
        self.level=level
        self.radius=radius
        self.sigma=sigma
        self.prob_thred=prob_thred
    def run(self,probs_map_path,output_path):
        '''

        :param probs_map_path: heatmap的路径
        :param output_path:  保存为csv坐标
        :return:
        '''
        probs_map = np.load(probs_map_path)
        X, Y = probs_map.shape
        mag=pow(2,self.level)
        if self.sigma>0:
            probs_map = filters.gaussian(probs_map, sigma=self.sigma)
        outfile= open(output_path,'w')
        while np.max(probs_map) > self.prob_thred:
            prob_max = probs_map.max()
            max_idx = np.where(probs_map == prob_max)
            x_mask, y_mask = max_idx[0][0], max_idx[1][0]
            x_wsi = int((x_mask + 0.5) * mag)
            y_wsi = int((y_mask + 0.5) * mag)
            outfile.write('{:0.5f},{},{}'.format(prob_max, x_wsi, y_wsi) + '\n')

            x_min = x_mask - self.radius if x_mask - self.radius > 0 else 0
            x_max = x_mask + self.radius if x_mask + self.radius <= X else X
            y_min = y_mask - self.radius if y_mask - self.radius > 0 else 0
            y_max = y_mask + self.radius if y_mask + self.radius <= Y else Y

            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    probs_map[x, y] = 0

        outfile.close()

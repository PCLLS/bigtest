import pandas as pd
import os
class NCRFdata:
    def __init__(self,normal_train,normal_valid,tumor_train,tumor_valid,save):
        self.normal_train=normal_train
        self.tumor_train=tumor_train
        self.normal_valid=normal_valid
        self.tumor_valid=tumor_valid
        self.savepath=save
        self.train_table = self.converttrain()
        self.valid_table = self.convertvalid()

    def converttrain(self):
        save = os.path.join(self.savepath,'ncrf_train.csv')
        if os.path.exists(save):
            return pd.read_csv(save,index_col=0,header=0)
        table1 = pd.read_table(self.normal_train,sep=',',header=None)
        table1.columns = ['slide_name', 'x', 'y']
        table1['slide_name'] = table1.apply(lambda x: x[0].lower(), axis=1)
        table1['label'] = 0
        table2 = pd.read_table(self.tumor_train, sep=',', header=None)
        table2.columns = ['slide_name', 'x', 'y']
        table2['slide_name'] = table2.apply(lambda x: x[0].lower(), axis=1)
        table2['label'] = 1
        table=pd.concat([table1,table2]).reset_index(drop=True)
        table.to_csv(save,header=True)
        return table

    def convertvalid(self):
        save = os.path.join(self.savepath, 'ncrf_valid.csv')
        if os.path.exists(save):
            return pd.read_csv(save, index_col=0, header=0)
        table1 = pd.read_table(self.normal_valid, sep=',', header=None)
        table1.columns = ['slide_name', 'x', 'y']
        table1['slide_name'] = table1.apply(lambda x: x[0].lower(), axis=1)
        table1['label'] = 0
        table2 = pd.read_table(self.tumor_valid, sep=',', header=None)
        table2.columns = ['slide_name', 'x', 'y']
        table2['slide_name'] = table2.apply(lambda x: x[0].lower(), axis=1)
        table2['label'] = 1
        table = pd.concat([table1, table2]).reset_index(drop=True)
        table.to_csv(save, header=True)
        return table
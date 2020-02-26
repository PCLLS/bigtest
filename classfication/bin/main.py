import sys
sys.path.append('../..')
from classfication.bin.config import *
import classfication.train
import classfication.data as data
from classfication.train import Train
# 基于config模板导入具体参数
from classfication.bin.inception_config import *

# ==============do not need change value below if not necessary======================

logging.info('Set DataLoader')

# train with checkpoint
best_epoch = 0
best_valid_acc = 0
logging.info('set Trainer')
train=Train(optimizer,net,workspace,loss, dataset,train_dataloader, valid_dataloader, out_fn)
logging.info('train models')
for epoch in range(start,end):
    loss=train.train_epoch()
    train_writer.add_scalar(loss,epoch)
    state_dict = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "last_epoch": epoch,
    }
    total_acc, pos_acc, neg_acc,loss,hard_neg_example=train.eval_epoch()
    ckpter.save(epoch, state_dict, total_acc)
    if total_acc > best_valid_acc:
        best_epoch = epoch
        best_valid_acc = total_acc


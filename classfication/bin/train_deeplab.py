import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import classfication.train
import classfication.data as data
from classfication.train import Train
# 鍩轰簬config妯℃澘瀵煎叆鍏蜂綋鍙傛暟
from classfication.bin.deeplab_config import *

# ==============do not need change value below if not necessary======================

logging.info('Set DataLoader')

# train with checkpoint
best_epoch = 0
best_valid_acc = 0
logging.info('set Trainer')
train=Train(optimizer,net,workspace,criterion, dataset,train_dataloader, valid_dataloader, out_fn)
logging.info('train models')
for epoch in range(start,end):
    loss=train.train_epoch()
    train_writer.add_scalar('loss_in_train',loss,epoch)
    state_dict = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "last_epoch": epoch,
    }
    ckpter.save(epoch, state_dict)


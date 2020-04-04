import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from classfication.train import Train
# 基于config模板导入具体参数
from inception_config import *

# ==============do not need change value below if not necessary======================

logging.info('Set DataLoader')

# train with checkpoint
best_epoch = 0
best_valid_acc = 0
logging.info('set Trainer')
train = Train(optimizer, net, workspace, criterion, dataset, train_dataloader, valid_dataloader, out_fn)
logging.info('train models')
for epoch in range(start, end):
    loss,metric,tumor = train.train_epoch()
    train_writer.add_scalar('loss_in_train', loss, epoch)
    train_writer.add_scalar('ratio_in_train', tumors / train_sample, epoch)
    train_writer.add_scalar('acc_in_train', metric.get_accuracy(), epoch)
    state_dict = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "last_epoch": epoch,
    }
    if evaluate:
        metric,loss = train.eval_epoch()
        train_writer.add_scalar('acc_in_valid', metric.get_accuracy(), epoch)
        train_writer.add_scalar('sensitive_in_valid', metric.get_sensitivity(), epoch)
        train_writer.add_scalar('loss_in_valid', metric.get_sensitivity(), epoch)
        ckpter.save(epoch, state_dict, metric.get_accuracy())
        if total_acc > best_valid_acc:
            best_epoch = epoch
            best_valid_acc = total_acc
    else:
        ckpter.save(epoch, state_dict)
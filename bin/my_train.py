# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm 
from torch.utils.data.distributed import DistributedSampler
# 防止找不到siamrpnpp的包

import sys 
sys.path.append(os.path.abspath('.')) # 

from siamrpnpp.utils.lr_scheduler import build_lr_scheduler
from siamrpnpp.utils.log_helper import init_log, print_speed, add_file_handler
from siamrpnpp.utils.distributed import dist_init, DistModule, reduce_gradients,average_reduce, get_rank, get_world_size
from siamrpnpp.utils.model_load import load_pretrain, restore_from, load_rpn_pretrain
from siamrpnpp.utils.average_meter import AverageMeter
from siamrpnpp.utils.misc import describe, commit
from siamrpnpp.models.model_builder import ModelBuilder
from siamrpnpp.datasets.dataset import TrkDataset
from siamrpnpp.core.config import cfg


os.environ['CUDA_VISIBLE_DEVICES'] = '0' #可见的gpu编号，我的电脑配置是两块GPU
# python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --master_port=2333 \
#     ./tools/train.py

#     单步调试注意事项
# 1） 多线程模式（NUM_WORKERS>0）下也是可以进行单步调试的（但是无法跳入某些函数的内部），
#     单步调试前建议将对应的config.yaml文件里面的线程数量设置为0，即在config.yaml->TRAIN->NUM_WORKERS:0
# 2） 加载config.yaml文件的变量config_path要设置成绝对路径，
# 3） config.yaml文件中的预训练模型路径也设置成绝对路径 BACKBONE->PRETRAINED:此处添加预模型绝对路径

logger = logging.getLogger('global')

#默认加载 SiamRPNpp-dwx-alexnet模型的 配置文件
config_path='..\models\siamrpnpp_alexnet\config.yaml'

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default=config_path, help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')
args = parser.parse_args()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#数据集
def build_data_loader():
    logger.info("build train2 dataset")
    #  train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    #if get_world_size() > 1:
    #train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader

#学习率
def build_opt_lr(model, current_epoch=0):
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        # 第十轮之后 解冻
        # for layer in cfg.BACKBONE.TRAIN_LAYERS:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.rpn_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.LR.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler

#梯度，权重
def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)

    v = tensor[0] / world_size 
    return v


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):

    cur_lr = lr_scheduler.get_cur_lr()

    #rank = get_rank()
    rank = 0

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    #world_size = get_world_size()
    world_size = 1
    length = len(train_loader.dataset)  # 64000*50=3200000
    num_per_epoch = len(train_loader.dataset) // cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE)
 
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and rank == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    #logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    for idx, data in enumerate(train_loader):  # idx 每个epoch要迭代的次数
        
        if epoch != idx // num_per_epoch + start_epoch:  #一个epoch迭代完成
        
            epoch = idx // num_per_epoch + start_epoch
            if  rank == 0:
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))

            if epoch == cfg.TRAIN.EPOCH:  # 所有epoch迭代完成
                return
            
            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch+1))

        tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx+1), pg['lr'], tb_idx)

        # data_time = average_reduce(time.time() - end)
        data_time = time.time()-end

        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        outputs = model(data)
        loss = outputs['total_loss'].mean()
    
        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            #reduce_gradients(model)

            if rank == 0 and cfg.TRAIN.LOG_GRADS:
                log_grads(model.module, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end

        batch_info = {}
        #batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['batch_time' ]= batch_time
        #batch_info['data_time'] = average_reduce(data_time)
        batch_info['data_time'] = data_time
        for k, v in outputs.items():
           #batch_info[k] = average_reduce(v.data.item())
           batch_info[k] = outputs[k].mean()
        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                # info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                #             epoch+1, (idx+1) % num_per_epoch,
                #             num_per_epoch, cur_lr)
                info = "Epoch: [{}][{}/{}] lr: {:6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch, num_per_epoch)
        end = time.time()

def main():
    # (1)
    # rank, world_size = dist_init()
    rank = 0

    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)

    # rank=0代表是单节点运行
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    #(2)
    '''
    # create model
     创建模型：通过ModelBuilder类作为模型各个模块组装类，
    # 在构造函数中分别调用了get_backbone/get_neck/get_rpn_head/get_mask_head(可选)
    # ModelBuilder实现了训练时用到的前向传播forward(data)函数，
    # 输入data包含训练时的template patch和search patch，
    # 以及分类label：label_cls、预测框位置回归label：label_loc，
    # 以及位置参数权重：label_loc_weight。
    # 返回的outputs字典包括了总损失值total_loss/分类损失cls_loss/位置损失loc_loss；
    # ModelBuilder类同时实现了推断时模板分支的计算方法template(z)（backbone和neck部分）
    # 和搜索分支的计算方法track(x)（backbone和neck部分，
    # 以及与模板分支得到的结果一起送入rpn_head部分），
    # 得到并返回分类和回归结果cls/loc/mask（可选）。
    '''
    model = ModelBuilder().cuda().train()

    # model.cuda().train2()：将模型移动至GPU上，并set mode=training mode ;
    # DistModule(model)：分布式训练模型，需要将模型副本copy至各个GPU中；
    dist_model = nn.DataParallel(model, device_ids=[0])
   
    #dist_model = DistModule(model)

    # load pretrained backbone weights do 需要加载rpn_head 的参数吗（不收敛）
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cfg.BACKBONE.PRETRAINED)
        # load_pretrain(model.backbone, backbone_path)
        # 新增加载rpn_head 的参数
        # load_rpn_pretrain(model.rpn_head, backbone_path)
        # content = torch.load(backbone_path,
        #                      map_location=lambda storage, loc: storage.cpu())
        # model_dict = model.state_dict()  # 获取网络参数
        # model_dict.update(content['state_dict'])  # 更新网络参数
        # # 载入 hsi 分类器
        # checkpoint = {k.replace('backbone', 'backbone2'): v for k, v in content['state_dict'].items()}
        # model_dict.update(checkpoint)
        # model.load_state_dict(model_dict)
        # 加载Dasiamrpn 权重
        checkpoint = torch.load('../models/SiamRPNBIG.model',
                                map_location=lambda storage, loc: storage.cpu())

        checkpoint = {k.replace('featureExtract.11', 'backbone.layer4.0'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('featureExtract.12', 'backbone.layer4.1'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('featureExtract.14', 'backbone.layer5.0'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('featureExtract.15', 'backbone.layer5.1'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('featureExtract.0', 'backbone.layer1.0'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('featureExtract.1', 'backbone.layer1.1'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('featureExtract.4', 'backbone.layer2.0'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('featureExtract.5', 'backbone.layer2.1'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('featureExtract.8', 'backbone.layer3.0'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('featureExtract.9', 'backbone.layer3.1'): v for k, v in checkpoint.items()}

        checkpoint = {k.replace('conv_r1', 'rpn_head.template_loc_conv'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('conv_r2', 'rpn_head.search_loc_conv'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('conv_cls1', 'rpn_head.template_cls_conv'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('conv_cls2', 'rpn_head.search_cls_conv'): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('regress_adjust', 'rpn_head.loc_adjust'): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader 加载数据集
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    # cfg.TRAIN.START_EPOCH = 0
    optimizer, lr_scheduler = build_opt_lr(dist_model.module,
                                           cfg.TRAIN.START_EPOCH)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
        #dist_model = DistModule(model)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()

    # content = torch.load('D:\SiamTrackers-master\SiamTrackers-master\SiamRPN\SiamRPN-pysot\models\pretrained_models\checkpoint_e41-16-up.pth')
    # print(content['state_dict'].keys())

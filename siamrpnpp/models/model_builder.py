# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from siamrpnpp.core.config import cfg
from siamrpnpp.models.loss import select_cross_entropy_loss, weight_l1_loss
from siamrpnpp.models.backbone import get_backbone
from siamrpnpp.models.head import get_rpn_head, get_mask_head, get_refine_head
from siamrpnpp.models.neck import get_neck
from siamrpnpp.core.xcorr import xcorr_fast

import cv2

import numpy as np
import matplotlib.pyplot as plt

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        # # hsi classfier
        # self.backbone2 = get_backbone(cfg.BACKBONE.TYPE,
        #                               **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)

        # zf_hsi = self.backbone2(z)
        # zf = 0.5 * zf + 0.5 * zf_hsi

        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track2(self, x, zf):
        xf = self.backbone(x)
        # 加权融合 0.9 0.1
        # xf_hsi = self.backbone2(x)
        # xf = 0.7 * xf + 0.3 * xf_hsi


        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        # response = xcorr_fast(xf, zf)
        # response_img = response.cpu().data.numpy()
        # response_img = np.squeeze(response_img, 0)
        # response_img = np.squeeze(response_img, 0)
        # response_img = np.transpose(response_img, [1, 2, 0])
        # response_img = response_img.astype(np.uint8)
        # plt.matshow(response_img)
        # plt.axis('off')
        # plt.show()



        cls, loc = self.rpn_head(zf, xf)

        heat = xf.cpu().data.numpy()  # 将tensor格式的feature map转为numpy格式
        heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除 [512,26,26]
        heat = heat[145 * 3:145 * 3 + 3, :]  # 切片获取某几个通道的特征图 435->438  [3,26,26]
        heatmap = np.maximum(heat, 0)  # heatmap与0比较
        heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值 [26,26]
        heatmap /= np.max(heatmap)  # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
        # plt.matshow(heatmap)  # 可以通过 plt.matshow 显示热力图
        # plt.show()
        # plt.matshow(heatmap, cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.show()
        # plt.show()

        heatmap_img = heatmap.reshape(26, 26, 1)
        x_np = x.cpu().data.numpy()
        x_np = np.squeeze(x_np, 0)
        # [C,H,W]->[H,W,C]
        x_np = np.transpose(x_np, [1, 2, 0])
        # x_np= x_np.reshape(287, 287, 3)
        x_img = x_np.astype(np.uint8)
        heat = cv2.resize(heatmap_img, (x_img.shape[1], x_img.shape[0]))  # 特征图的大小调整为与原始图像相同
        heat = np.uint8(255 * heat)  # 将特征图转换为uint8格式
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
        heat_img = cv2.addWeighted(x_img, 1, heat, 0.5, 0)
        # plt.imshow(x_img)
        # plt.axis('off')
        # plt.show()

        if cfg.MASK.MASK:
                mask, self.mask_corr_feature = self.mask_head(self.zf, xf)

        return {
            'cls': cls,
            'loc': loc,
            'mask': mask if cfg.MASK.MASK else None,
            'xf_heatmap': heatmap,
            'heat_img': heat_img
        }

    def track(self, x):
        xf = self.backbone(x)
        # 加权融合 0.9 0.1
        # xf_hsi = self.backbone2(x)
        # xf = 0.7 * xf + 0.3 * xf_hsi


        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)

        heat = xf.cpu().data.numpy()  # 将tensor格式的feature map转为numpy格式
        heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
        heat = heat[145 * 3:145 * 3 + 3, :]  # 切片获取某几个通道的特征图
        heatmap = np.maximum(heat, 0)  # heatmap与0比较
        heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值
        heatmap /= np.max(heatmap)  # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
        plt.matshow(heatmap)  # 可以通过 plt.matshow 显示热力图
        plt.show()

        heat = cv2.resize(heatmap, (x.shape[1], x.shape[0]))  # 特征图的大小调整为与原始图像相同
        heat = np.uint8(255 * heat)  # 将特征图转换为uint8格式
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
        heat_img = cv2.addWeighted(x, 1, heat, 0.5, 0)

        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None,
                'xf_heatmap': heatmap,
                'heat_img': heat_img
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        # zf_hsi = self.backbone2(template)
        # zf = 0.5 * zf + 0.5 * zf_hsi

        # 加权融合 0.5 0.5
        # xf_hsi = self.backbone2(search)
        # xf = 0.3 * xf + 0.7 * xf_hsi

        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs

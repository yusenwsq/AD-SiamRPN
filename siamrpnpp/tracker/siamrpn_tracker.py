# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from siamrpnpp.core.config import cfg
from siamrpnpp.utils.anchor import Anchors
from siamrpnpp.tracker.base_tracker import SiameseTracker
from updatenet.net_upd import UpdateResNet512, UpdateResNet256

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model, step=1):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        self.step = step

        if self.step == 1:
            # load UpdateNet network
            self.updatenet = UpdateResNet512()
            # self.updatenet = UpdateResNet256()
            update_model = torch.load('../models/HSupdatenet.pth.tar')['state_dict']

            update_model_fix = dict()
            for i in update_model.keys():
                if i.split('.')[0] == 'module':  # 多GPU模型去掉开头的'module'
                    update_model_fix['.'.join(i.split('.')[1:])] = update_model[i]
                else:
                    update_model_fix[i] = update_model[i]  # 单GPU模型直接赋值

            self.updatenet.load_state_dict(update_model_fix)

            # self.updatenet.load_state_dict(update_model)
            self.updatenet.eval().cuda()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        # bbox 的中心位置
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        # bbox 的大小
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        #
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        # 每个通道上 一个平面上求均值  axis=0 对列 axis=1 对行
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        z_crop_img = z_crop.cpu().data.numpy()
        z_crop_img = np.squeeze(z_crop_img, 0)
        z_crop_img = np.transpose(z_crop_img, [1, 2, 0])
        z_crop_img = z_crop_img.astype(np.uint8)
        plt.matshow(z_crop_img, cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.show()

        self.model.template(z_crop)
        # 累积模板 为 初始模板
        self.model.z_f_ = self.model.backbone(z_crop)
        self.model.zf_0 = self.model.backbone(z_crop)

    def track(self, img, step=1):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        x_crop_img = x_crop.cpu().data.numpy()
        x_crop_img = np.squeeze(x_crop_img, 0)
        x_crop_img = np.transpose(x_crop_img, [1, 2, 0])
        x_crop_img = x_crop_img.astype(np.uint8)
        # plt.matshow(x_crop_img, cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.show()

        if step > 0:
            # 检测模板
            z_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE, round(s_x), self.channel_average)
            z_f = self.model.backbone(z_crop)
            z_f_ = z_f
            if step == 2:
                zLR = 0.00102  # SiamFC[0.01, 0.05],  0.0102是siamfc初始化的方法
                z_f_ = (1 - zLR) * Variable(self.model.z_f_).cuda() + zLR * z_f  # 累积模板

            if step == 1:
                temp = torch.cat((Variable(self.model.zf_0).cuda(), Variable(self.model.z_f_).cuda(), z_f), 1)
                init_inp = Variable(self.model.zf_0).cuda()
                z_f_ = self.updatenet(temp, init_inp)

            self.model.z_f_ = z_f_
            outputs = self.model.track2(x_crop, z_f_)
        else:
            outputs = self.model.track(x_crop)

        # No_update template热力图
        # z_f_heat = z_f.cpu().data.numpy()
        # z_f_heat = np.squeeze(z_f_heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除 [512,26,26]
        # z_f_heat = z_f_heat[145 * 3:145 * 3 + 3, :]  # 切片获取某几个通道的特征图 435->438  [3,26,26]
        # z_f_heatmap = np.maximum(z_f_heat, 0)  # heatmap与0比较
        # z_f_heatmap = np.mean(z_f_heatmap, axis=0)  # 多通道时，取均值 [26,26]
        # z_f_heatmap /= np.max(z_f_heatmap)  # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
        # plt.matshow(z_f_heatmap)  # 可以通过 plt.matshow 显示热力图
        # plt.show()

        # update template/linear热力图
        # z_f_up_heat = z_f_.cpu().data.numpy()
        # z_f_up_heat = np.squeeze(z_f_up_heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除 [512,26,26]
        # z_f_up_heat = z_f_up_heat[145 * 3:145 * 3 + 3, :]  # 切片获取某几个通道的特征图 435->438  [3,26,26]
        # z_f_up_heatmap = np.maximum(z_f_up_heat, 0)  # heatmap与0比较
        # z_f_up_heatmap = np.mean(z_f_up_heatmap, axis=0)  # 多通道时，取均值 [26,26]
        # z_f_up_heatmap /= np.max(z_f_up_heatmap)  # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
        # plt.matshow(z_f_up_heatmap)  # 可以通过 plt.matshow 显示热力图
        # plt.show()

        # template [127,127,3]
        # z_crop_heat = z_crop.cpu().data.numpy()
        # z_crop_heat = np.squeeze(z_crop_heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除 [512,26,26]
        # z_crop_heat = np.transpose(z_crop_heat, [1, 2, 0]) # 切片获取某几个通道的特征图 435->438  [3,26,26]
        # z_crop_heatmap = np.maximum(z_crop_heat, 0)  # heatmap与0比较
        # z_crop_heatmap = np.mean(z_crop_heatmap, axis=0)  # 多通道时，取均值 [26,26]
        # z_crop_heatmap /= np.max(z_crop_heatmap)  # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
        # plt.matshow(z_crop_heatmap)  # 可以通过 plt.matshow 显示热力图
        # plt.show()
        # z_crop_heat = np.squeeze(z_crop_heat, 0)
        # z_crop_heat = np.transpose(z_crop_heat, [1, 2, 0])
        # z_crop_img = z_crop_heat.astype(np.uint8)
        # plt.matshow(z_crop_img, cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.show()


        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        xf_heatmap = outputs['xf_heatmap']
        heat_img = outputs['heat_img']
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score,
                'xf_heatmap': xf_heatmap,
                'heat_img': heat_img,
                # 'z_f_heatmap': z_f_heatmap,
                # 'z_f_up_heatmap': z_f_up_heatmap
               }

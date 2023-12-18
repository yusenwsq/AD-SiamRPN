# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import sys

sys.path.append(os.path.abspath('.'))
from siamrpnpp.core.config import cfg
from siamrpnpp.models.model_builder import ModelBuilder
from siamrpnpp.tracker.tracker_builder import build_tracker
from siamrpnpp.utils.bbox import get_axis_aligned_bbox
from siamrpnpp.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
# from toolkit.utils.region import vot_overlap, vot_float2str

import tqdm

'''
config = '../models/siamrpnpp_alexnet/config.yaml'         #SiamRPN   -AlexNet   180fps
# snapshot = '../models/siamrpnpp_alexnet/snapshot/checkpoint_e41-16-up.pth'
snapshot = './models/siamrpnpp_alexnet/snapshot/checkpoint_e16.pth'
tracker_name = 'SiamRPN1'
dataset = 'GOT-10k'

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', '-d', default=dataset, type=str, help='datasets')
parser.add_argument('--tracker_name', '-t', default=tracker_name, type=str,help='tracker  name')
parser.add_argument('--config', default=config, type=str, help='config file')
parser.add_argument('--snapshot', default=snapshot, type=str, help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--vis', action='store_true', help='whether visualzie result')

args = parser.parse_args()
'''
torch.set_num_threads(1)


def main(model_id):
    # load config
    cfg.merge_from_file(args.config)
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # cur_dir = os.path.dirname(os.path.realpath(__file__))

    # dataset_root = os.path.join( './datasets', args.dataset)
    # dataset_root = "E:/BaiduNetdiskDownload/HOT2022"
    #dataset_root = "D:/data/test"
    dataset_root = "G:/test"

    # create model
    model = ModelBuilder()

    # load model
    # model = load_pretrain(model, args.snapshot).cuda().eval()

    # content = torch.load(args.snapshot,
    #                      map_location=lambda storage, loc: storage.cpu())
    # model.load_state_dict(content['state_dict'])

    # content = torch.load(args.snapshot,
    #                      map_location=lambda storage, loc: storage.cpu())
    # model_dict = model.state_dict()  # 获取网络参数
    # model_dict.update(content['state_dict'])  # 更新网络参数
    # 载入 hsi 分类器
    # checkpoint = {k.replace('backbone', 'backbone2'): v for k, v in content['state_dict'].items()}
    # model_dict.update(model_dict)
    # model.load_state_dict(model_dict)

    # 加载Dasiamrpn 权重
    # checkpoint = torch.load(args.snapshot,
    #                         map_location=lambda storage, loc: storage.cpu())
    # model.load_state_dict(checkpoint['state_dict'])

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

    # alexnetlegacy 加载方式
    # checkpoint = torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu())
    # checkpoint = {k.replace('featureExtract', 'backbone.featureExtract'): v for k, v in checkpoint.items()}
    # checkpoint = {k.replace('conv_r1', 'rpn_head.template_loc_conv'): v for k, v in checkpoint.items()}
    # checkpoint = {k.replace('conv_r2', 'rpn_head.search_loc_conv'): v for k, v in checkpoint.items()}
    # checkpoint = {k.replace('conv_cls1', 'rpn_head.template_cls_conv'): v for k, v in checkpoint.items()}
    # checkpoint = {k.replace('conv_cls2', 'rpn_head.search_cls_conv'): v for k, v in checkpoint.items()}
    # checkpoint = {k.replace('regress_adjust', 'rpn_head.loc_adjust'): v for k, v in checkpoint.items()}
    # model.load_state_dict(checkpoint)

    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    # create dataset 
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    # model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            # for v_idx, video in tqdm(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]  # [topx,topy,w,h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                      True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, args.tracker_name,
                                      'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(args.tracker_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    # cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox["bbox"]))
                    # gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h] #[topx,topy,w,h]
                    gt_bbox_ = gt_bbox["bbox"]
                    tracker.init(img, gt_bbox_)
                    gt_bbox_ = [int(x) for x in gt_bbox_]
                    pred_bbox = gt_bbox_
                    # scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bbox = [int(x) for x in pred_bbox]
                    pred_bboxes.append(pred_bbox)
                    # scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()

            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, args.tracker_name,
                                          'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                                           '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, args.tracker_name, str(video.name))
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)

                result_path = os.path.join(video_path, '{}_00' + str(model_id) + '.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write('	'.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, args.tracker_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, str(video.name), toc, idx / toc))


if __name__ == '__main__':
    model_id = 0

    config = '../models/siamrpnpp_alexnet/config.yaml'  # SiamRPN   -AlexNet   180fps
    # snapshot = './models/siamrpnpp_alexnet/snapshot_up4/checkpoint_e' + str(index+1) + '.pth' # 24_3
    snapshot = '../models/SiamRPNBIG.model'
    # snapshot = './models/siamrpnpp_alexnet/snapshot7/checkpoint_e' + str(model_id) + '.pth'
    # tracker_name = 'SiamRPN_alex512_up4_'+str(index+1)
    tracker_name = 'SiamRPN_25hsi'
    dataset = 'GOT-10k'

    parser = argparse.ArgumentParser(description='siamrpn tracking')
    parser.add_argument('--dataset', '-d', default=dataset, type=str, help='datasets')
    parser.add_argument('--tracker_name', '-t', default=tracker_name, type=str, help='tracker  name')
    parser.add_argument('--config', default=config, type=str, help='config file')
    parser.add_argument('--snapshot', default=snapshot, type=str, help='snapshot of models to eval')
    parser.add_argument('--video', default='', type=str, help='eval one special video')
    parser.add_argument('--vis', action='store_true', help='whether visualzie result')

    args = parser.parse_args()

    main(model_id)

if __name__ == '__main__1':
    # model_id = 0

    # test 所有模型
    for i in range(19):
        i = i + 1
        config = '../models/siamrpnpp_alexnet/config.yaml'  # SiamRPN   -AlexNet   180fps
        # snapshot = '../models/siamrpnpp_alexnet/snapshot/checkpoint_e41-16-up.pth'
        snapshot = './models/siamrpnpp_alexnet/snapshot9/checkpoint_e' + str(i) + '.pth'
        tracker_name = 'SiamRPN9_2'
        dataset = 'GOT-10k'

        parser = argparse.ArgumentParser(description='siamrpn tracking')
        parser.add_argument('--dataset', '-d', default=dataset, type=str, help='datasets')
        parser.add_argument('--tracker_name', '-t', default=tracker_name, type=str, help='tracker  name')
        parser.add_argument('--config', default=config, type=str, help='config file')
        parser.add_argument('--snapshot', default=snapshot, type=str, help='snapshot of models to eval')
        parser.add_argument('--video', default='', type=str, help='eval one special video')
        parser.add_argument('--vis', action='store_true', help='whether visualzie result')

        args = parser.parse_args()

        main(i)
        print("model" + str(i) + " finished")

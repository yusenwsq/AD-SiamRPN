from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
import pandas as pd


from glob import glob
import sys
sys.path.append(os.path.abspath('.'))
from siamrpnpp.core.config import cfg
from siamrpnpp.models.model_builder import ModelBuilder
from siamrpnpp.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
        for img in images:
            frame2 = cv2.imread(img)
            img2 = img.replace("bestimg", "HSI-FalseColor")
            frame = cv2.imread(img2)
            # frame2为 HSI  frame 为RGB
            yield frame, frame2

def get_frames2(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.bm*'))
        images = sorted(images, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
        for img in images:
            frame2 = cv2.imread(img)
            # img2 = img.replace("bestimg", "HSI-FalseColor")
            # frame = cv2.imread(img2)
            # frame2为 HSI  frame 为RGB
            yield frame2

def select_model(arg):

    if arg == 1:
        config = '../models/siamrpnpp_alexnet/config.yaml'
        snapshot = '../models/siamrpnpp_alexnet/snapshot/checkpoint_e41-16-up.pth'
    elif arg == 2:
        config = '../models/siamrpnpp_alexnet/config.yaml'
        snapshot = '../models/ADSiamRPN.model'
    else:
        print('no model is selected')
        return 0

    param = 'model_'+str(arg)
    video2 = 'D:/data/test/worker/bestimg'
    return config, snapshot, video2, param

config, snapshot, video2, param=select_model(2)

def main():
    #load parameters
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file', default=config)
    parser.add_argument('--snapshot', type=str, help='model name', default=snapshot)
    parser.add_argument('--video_name', type=str, help='videos or image files', default=video2)
    args = parser.parse_args()

    # load config
    cfg.merge_from_file(args.config)

    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()
    # 加载 权重
    checkpoint = torch.load('../models/ADSiamRPN.model',
                         map_location=lambda storage, loc: storage.cpu())

    model.load_state_dict(checkpoint)

    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = "test1"
    else:
        video_name = 'webcam'

    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    directory = '../testing_dataset/result/'+param+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    ################################变量初始化###################################
    timer = 0
    num = 0
    # 获取初始框
    init = args.video_name.replace("bestimg", "HSI-FalseColor")
    init_rect_path = os.path.join(init, "groundtruth_rect.txt")
    gt = pd.read_table(init_rect_path, header=None, encoding="utf-8")
    gt = gt.dropna(axis=1, how='all').to_numpy()
    init_rect = gt[0]
    gt_id = -1

    # 保存结果
    pred_bboxes = []

    for (frame, frame2) in get_frames(args.video_name):
        start = cv2.getTickCount()
        num = num + 1
        if first_frame:
            gt_id += 1

            pred_bbox = init_rect

            tracker.init(frame2, init_rect)
            first_frame = False

            pred_bboxes.append(pred_bbox)
        else:
            gt_id += 1
            outputs = tracker.track(frame2)

            end = cv2.getTickCount()
            during = (end - start) / cv2.getTickFrequency()
            timer = timer+during

            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask, mask*255]).transpose(1, 2, 0)
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                c = sorted(contours, key=cv2.contourArea, reverse=True)[0] #面积最大的轮廓区域
                rect_new2 = cv2.boundingRect(c)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                cv2.rectangle(frame, (rect_new2[0], rect_new2[1]),
                               (rect_new2[0]+rect_new2[2], rect_new2[1]+rect_new2[3]),
                               (0, 0, 255), 2)
            else:
                gt_bbox = gt[gt_id]
                bbox = list(map(int, outputs['bbox']))
                pred_bbox = bbox
                pred_bboxes.append(pred_bbox)
                # 绿色 当前结果  ours
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 2)
                cv2.putText(frame, "", (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, 4, 1)
                # 真实框 红色  gt
                cv2.rectangle(frame, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                              (0, 0, 255), 2)
                cv2.putText(frame, "", (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, 4, 1)


                # 显示左上角第几张图片 道奇蓝    30,144,255
                cv2.putText(frame, "#", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 144, 255), 2, 8, 0)
                cv2.putText(frame, str(gt_id + 1), (37, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 144, 255), 2, 8, 0)



            cv2.imshow(video_name, frame)

            cv2.waitKey(30)

    fps = int(num/timer)
    print('FPS:%d'%(fps))

if __name__ == '__main__':
    main()

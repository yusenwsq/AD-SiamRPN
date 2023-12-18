
import json
import os

from tqdm import tqdm

from .dataset import Dataset
from .video import Video

class GOT10kVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(GOT10kVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

    # def load_tracker(self, path, tracker_names=None):
    #     """
    #     Args:
    #         path(str): path to result
    #         tracker_name(list): name of tracker
    #     """
    #     if not tracker_names:
    #         tracker_names = [x.split('/')[-1] for x in glob(path)
    #                 if os.path.isdir(x)]
    #     if isinstance(tracker_names, str):
    #         tracker_names = [tracker_names]
    #     # self.pred_trajs = {}
    #     for name in tracker_names:
    #         traj_file = os.path.join(path, name, self.name+'.txt')
    #         if os.path.exists(traj_file):
    #             with open(traj_file, 'r') as f :
    #                 self.pred_trajs[name] = [list(map(float, x.strip().split(',')))
    #                         for x in f.readlines()]
    #             if len(self.pred_trajs[name]) != len(self.gt_traj):
    #                 print(name, len(self.pred_trajs[name]), len(self.gt_traj), self.name)
    #         else:

    #     self.tracker_names = list(self.pred_trajs.keys())

class GOT10kDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(GOT10kDataset, self).__init__(name, dataset_root)
        # with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
        with open(os.path.join(dataset_root, 'got10k_hsi25.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(range(len(meta_data[0])), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = GOT10kVideo(video,
                                          dataset_root,
                                          meta_data[0][video]['base_path'], # 视频路径
                                          meta_data[0][video]['frame'][0]['bbox'], # 初始帧
                                          meta_data[0][video]['base_path'], # 图片名字
                                          meta_data[0][video]['frame'], # groundtruth
                                          None)


        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())

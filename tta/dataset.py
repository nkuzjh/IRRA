import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


import os
import random
from random import randint, shuffle
from random import random as rand
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image



class IRRA_tta_dataset(Dataset):
    def __init__(
        self,
        config, test_transforms,
        sims_topk_matrix_t2i, qids, gids, captions, imgs,
        recall_types,
        ss_idxs_list,
        uncertaintys_list,
        proba_top1_sim_list,
        proba_inversed_sim_list
    ):
        self.config = config
        self.transform = test_transforms
        self.sims_topk_matrix_t2i = sims_topk_matrix_t2i
        self.qids = qids
        self.gids = gids
        self.captions = captions
        self.imgs = imgs
        # self.recall_types = recall_types
        self.uncertaintys_list = uncertaintys_list
        # if config.get('uncertainty_temper_is_learnable', False) == True:
        if 1:
            self.proba_top1_sim_list = proba_top1_sim_list
            self.proba_inversed_sim_list = proba_inversed_sim_list

        if config.get('sample_selection', 'all') == 'top1':
            self.sims_topk_matrix_t2i = sims_topk_matrix_t2i[ss_idxs_list]
            self.qids = qids[ss_idxs_list]
            self.captions = captions[ss_idxs_list]
            # self.recall_types = [recall_types[i] for i in ss_idxs_list]
            self.uncertaintys_list = [uncertaintys_list[i] for i in ss_idxs_list]
            # if config.get('uncertainty_temper_is_learnable', False) == True:
            if 1:
                self.proba_top1_sim_list = [proba_top1_sim_list[i] for i in ss_idxs_list]
                self.proba_inversed_sim_list = [proba_inversed_sim_list[i] for i in ss_idxs_list]

    def __len__(self):
        return len(self.sims_topk_matrix_t2i)

    def __getitem__(self, index):
        topk_idx = self.sims_topk_matrix_t2i[index]
        gids_topk = self.gids[topk_idx] #([8])
        imgs_topk = self.imgs[topk_idx] #[8, 3, 384, 128])

        qids_repeatk = self.qids[index]#.repeat(self.config['k_tta']) #([8])

        captions_repeatk = self.captions[index]#.repeat(self.config['k_tta'], 1) #([8, 77])
        uncertainty = self.uncertaintys_list[index]
        proba_top1_sim = self.proba_top1_sim_list[index]
        proba_inversed_sim = self.proba_inversed_sim_list[index]

        return gids_topk, imgs_topk, qids_repeatk, captions_repeatk, uncertainty, proba_top1_sim, proba_inversed_sim


def create_tta_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, bs, n_worker, is_train, collate_fn in zip(datasets, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False

        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders



# class search_tta_img_aug_dataset(Dataset):
#     def __init__(self, config, transform):
#         ann_file = config['tta_file']
#         self.transform = transform
#         self.image_root = config.get('image_root_tta', config['image_root'])
#         self.max_words = config['max_words']

#         self.ann = read_json_to_list(ann_file)

#         self.be_pose_img = config.get('be_pose_img', False)
#         print('     tta img aug dataset -->    be_pose_img:', self.be_pose_img)

#         self.text = []
#         self.image = []
#         self.g_pids = []
#         self.q_pids = []
#         for img_id, ann in enumerate(self.ann):
#             self.g_pids.append(ann['image_id'])
#             self.image.append(ann['image'])
#             for i, caption in enumerate(ann['caption']):
#                 self.q_pids.append(ann['image_id'])
#                 self.text.append(pre_caption(caption, self.max_words))
#         pass

#     def __len__(self):
#         return len(self.image)

#     def __getitem__(self, index):
#         image_path = os.path.join(self.image_root, self.ann[index]['image'])
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)

#         if self.be_pose_img:
#             pose_path = os.path.join(self.image_root, 'pose/' + self.ann[index]['image'])
#             pose = Image.open(pose_path).convert('RGB')
#             pose = self.transform(pose)
#         else:
#             pose = {}

#         return image, pose, index


# def create_tta_img_aug_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
#     loaders = []
#     for dataset, bs, n_worker, is_train, collate_fn in zip(datasets, batch_size, num_workers, is_trains, collate_fns):
#         if is_train:
#             shuffle = True
#             drop_last = False
#         else:
#             shuffle = False
#             drop_last = False

#         loader = DataLoader(
#             dataset,
#             batch_size=bs,
#             num_workers=n_worker,
#             pin_memory=True,
#             shuffle=shuffle,
#             collate_fn=collate_fn,
#             drop_last=drop_last,
#         )
#         loaders.append(loader)

#     if len(loaders) <= 1:
#         print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

#     return loaders


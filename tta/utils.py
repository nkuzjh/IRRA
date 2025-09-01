
import os
import time
import datetime
import json

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F



def sample_selection_itc(sims_matrix_t2i, sims_matrix_i2t):
    # find_inter_top1_sample_selection
    ss_idxs_list = []
    for i, sims_t2i in enumerate(sims_matrix_t2i):
        top1_sim_t2i, top1_idx_t2i = sims_t2i.topk(k=1, dim=0) # 获取t2i top1的相似度(top1_sim_t2i)和索引(top1_idx_t2i)
        top1_sim_i2t, top1_idx_i2t  = sims_matrix_i2t[top1_idx_t2i][0].topk(k=1, dim=0) # 获取top1_idx_t2i对应的i2t相似度和索引
        if top1_idx_i2t == i: # 只保留t2i和i2t互为top1的样本
            ss_idxs_list.append(i)
        # else:
        #     print(f"t2i & i2t do not have inter top1 sample: i2t_idx={i}, t2i_idx={top1_idx_t2i}, top1_sim_i2t={top1_sim_i2t}, top1_sim_t2i={top1_sim_t2i}")

    return ss_idxs_list


def compute_uncertainty_itc(config, sims_matrix_t2i, sims_matrix_i2t):
    k_test = config['k_test']
    uncertainty_temper = config.get('uncertainty_temper', 1.0)
    uncertainty_t2i_temper = config.get('uncertainty_t2i_temper', 1.0)
    uncertainty_i2t_temper = config.get('uncertainty_i2t_temper', 1.0)

    uncertaintys_list = []
    proba_top1_sim_list = []
    proba_inversed_sim_list = []
    for i, sims_t2i in enumerate(sims_matrix_t2i):

        topk_sim_t2i, topk_idx_t2i = sims_t2i.topk(k=k_test, dim=0)
        proba_top1_sim_t2i = F.softmax(topk_sim_t2i * uncertainty_t2i_temper, dim=0)[0] # i2t和t2i的logtis rank/distribution呈现长尾or平均的现象

        proba_inversed_sim_i2t_top1_idx_t2i = torch.zeros([])
        topk_sim_i2t_top1_idx_t2i, topk_idx_i2t_top1_idx_t2i = sims_matrix_i2t[topk_idx_t2i[0]].topk(k=k_test, dim=0)
        if i in topk_idx_i2t_top1_idx_t2i:
            idx_i_in_topk_idx_i2t_top1_idx_t2i = torch.where(topk_idx_i2t_top1_idx_t2i == i)[0][0]
            proba_inversed_sim_i2t_top1_idx_t2i = F.softmax(topk_sim_i2t_top1_idx_t2i * uncertainty_i2t_temper, dim=0)[idx_i_in_topk_idx_i2t_top1_idx_t2i]

        uncertainty = torch.exp( (1 - (proba_top1_sim_t2i + proba_inversed_sim_i2t_top1_idx_t2i) / 2) * uncertainty_temper )

        uncertaintys_list.append(uncertainty)
        proba_top1_sim_list.append(proba_top1_sim_t2i)
        proba_inversed_sim_list.append(proba_inversed_sim_i2t_top1_idx_t2i)
    return uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list


def preprocess_tta_coefficients(config, sims_matrix_t2i):
    print(f"     preprocess_tta_coefficients  start")

    ## 获取metric标签用于可视化
    # k_test = config['k_test']
    # top128_sims, top128_idxs = sims_matrix_t2i.topk(k=k_test, dim=1)
    # (recall1, recall5, recall10), recall_types = calculate_recall(top128_idxs, labels)
    recall_types = [] #TODO

    print(f"     sample selection ...")
    if config.get('sample_selection', 'all') == 'top1':
        ss_idxs_list = sample_selection_itc(sims_matrix_t2i, sims_matrix_t2i.t()) # 找到i2t和t2i互为top1的样本索引
    else:
        ss_idxs_list = torch.arange(0, sims_matrix_t2i.size(0)) # 全部样本
    print("     number of sample after sample_selection: {}".format(len(ss_idxs_list)))

    print(f"     uncertainty ...")
    ## tta coeffis
    if config.get('uncertainty', None) == 'inversed_recall_proba':
        uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list = compute_uncertainty_itc(config, sims_matrix_t2i, sims_matrix_t2i.t())
    else:
        uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list = torch.ones(sims_matrix_t2i.size(0)) , torch.ones(sims_matrix_t2i.size(0)), torch.ones(sims_matrix_t2i.size(0))

    print(f"     preprocess_tta_coefficients  end")
    return recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list
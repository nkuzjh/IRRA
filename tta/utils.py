
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


def sample_selection_topk_itc(sims_matrix_t2i, sims_matrix_i2t, k_sample_selection):
    # find_inter_top1_sample_selection
    ss_idxs_list = []
    for i, sims_t2i in enumerate(sims_matrix_t2i):
        topk_sim_t2i, topk_idx_t2i = sims_t2i.topk(k=k_sample_selection, dim=-1) # 获取t2i topk的相似度(topk_sim_t2i)和索引(topk_idx_t2i)
        topk_sim_i2t, topk_idx_i2t  = sims_matrix_i2t[topk_idx_t2i].topk(k=k_sample_selection, dim=-1) # 获取top1_idx_t2i对应的i2t相似度和索引
        if i in topk_idx_i2t.reshape(-1).tolist(): # 只保留t2i和i2t互为topk的样本
            ss_idxs_list.append(i)
        # else:
        #     print(f"t2i & i2t do not have inter top1 sample: i2t_idx={i}, t2i_idx={top1_idx_t2i}, top1_sim_i2t={top1_sim_i2t}, top1_sim_t2i={top1_sim_t2i}")

    return ss_idxs_list


def compute_uncertainty_itc(config, sims_matrix_t2i, sims_matrix_i2t):
    k_test = config['k_test']
    uncertainty_temper = config.get('uncertainty_temper', 1.0)
    uncertainty_t2i_temper = config.get('uncertainty_t2i_temper', 1.0)
    uncertainty_i2t_temper = config.get('uncertainty_i2t_temper', 1.0)

    uncertaintys1_list = []
    uncertaintys2_list = []
    uncertaintys3_list = []
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

        uncertainty1 = torch.exp( (1 - (proba_top1_sim_t2i + proba_inversed_sim_i2t_top1_idx_t2i) / 2) * uncertainty_temper )
        uncertainty2 = torch.exp( torch.abs(proba_top1_sim_t2i - proba_inversed_sim_i2t_top1_idx_t2i) / ( (proba_top1_sim_t2i + proba_inversed_sim_i2t_top1_idx_t2i) / 2 ) * uncertainty_temper )
        uncertainty3 = torch.abs( torch.log(proba_top1_sim_t2i + 1e-2) - torch.log( proba_inversed_sim_i2t_top1_idx_t2i + 1e-2) )

        uncertaintys1_list.append(uncertainty1)
        uncertaintys2_list.append(uncertainty2)
        uncertaintys3_list.append(uncertainty3)
        proba_top1_sim_list.append(proba_top1_sim_t2i)
        proba_inversed_sim_list.append(proba_inversed_sim_i2t_top1_idx_t2i)
    return uncertaintys1_list, uncertaintys2_list, uncertaintys3_list, proba_top1_sim_list, proba_inversed_sim_list


def sample_neg_idxs(sims_matrix_i2t, k_tta, k_test, neg_sample_range=[32, 128]):
    """
    Sample negative indices from the similarity matrix.
    Args:
        sims_matrix_i2t: Similarity matrix of shape (num_images, num_texts).
        k_tta: Number of negative samples to sample.
        k_test: Number of top-k samples to consider.
    Returns:
        neg_sims: Negative similarities of shape (num_images, k_tta-1).
        neg_idxs: Indices of the negative samples.
    """
    num_images, num_texts = sims_matrix_i2t.shape
    neg_idxs = []
    neg_sims = []
    for i in range(num_images):
        topk_sim_i2t, topk_idx_i2t = sims_matrix_i2t[i].topk(k=k_test, dim=0)
        # random_idx = torch.randperm(k_test-2*k_tta)[:k_tta-1] + 2*k_tta # 随机采样k_tta-1个负样本索引
        random_idx = torch.randperm(neg_sample_range[1] - neg_sample_range[0])[:k_tta-1] + neg_sample_range[0]
        neg_sims.append(topk_sim_i2t[random_idx])
        neg_idxs.append(topk_idx_i2t[random_idx])
    return torch.stack(neg_sims, dim=0), torch.stack(neg_idxs, dim=0)


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
    if config.get('sample_selection', 'all') == 'topk':
        k_sample_selection = config.get('k_sample_selection', 5)
        ss_idxs_list = sample_selection_topk_itc(sims_matrix_t2i, sims_matrix_t2i.t(), k_sample_selection) # 找到i2t和t2i互为top1的样本索引
    else:
        ss_idxs_list = torch.arange(0, sims_matrix_t2i.size(0)) # 全部样本
    print("     number of sample after sample_selection: {}".format(len(ss_idxs_list)))

    print(f"     uncertainty ...")
    ## tta coeffis
    if config.get('uncertainty', None) is not None:
        uncertaintys1_list, uncertaintys2_list, uncertaintys3_list, proba_top1_sim_list, proba_inversed_sim_list = compute_uncertainty_itc(config, sims_matrix_t2i, sims_matrix_t2i.t())

    if config.get('uncertainty', None) == 'inversed_recall_proba':
        uncertaintys_list = uncertaintys1_list
    elif config.get('uncertainty', None) == 'diff_div_mean':
        uncertaintys_list = uncertaintys2_list
    elif config.get('uncertainty', None) == 'abs_diff_log':
        uncertaintys_list = uncertaintys3_list
    else:
        uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list = torch.ones(sims_matrix_t2i.size(0)) , torch.ones(sims_matrix_t2i.size(0)), torch.ones(sims_matrix_t2i.size(0))

    print(f"     pos/neg sampling ...")
    if config.get('neg_sample_range', None) is None:
        topk_sim, topk_idx = sims_matrix_t2i.topk(k=config['k_tta'], dim=1) #[k_tta]
    else:
        ## sampling stretegy
        ## 采样正样本
        # pos_sample_range =  tta_cfg.pos_sample_range if hasattr(tta_cfg, "pos_sample_range") else [0, 1] # 正样本采样范围
        top1_sims, top1_idxs = sims_matrix_t2i.topk(k=1, dim=1) #正样本直接使用top1 or 使用top5采样一个正样本；但是这里相似度top5不一定就是5个label, 根据训练top5分数分布，已确认无需使用top5采样正样本
        neg_sample_range = config['neg_sample_range'] #neg_sample_range = tta_cfg.neg_sample_range if hasattr(tta_cfg, "neg_sample_range") else [32, 128] # 负样本采样范围
        top1_sims, top1_idxs =top1_sims[:,0], top1_idxs[:,0]
        ## 采样困难负样本
        neg_sims, neg_idxs = sample_neg_idxs(sims_matrix_t2i, config['k_tta'], config['k_test'], neg_sample_range) # 负样本采样k_tta-1个, 根据score数值可视化差异确定采样范围
        ## 拼接正负样本
        sampled_sims_matrix_t2i = []
        sampled_sims_idx_t2i = []
        for i in range(sims_matrix_t2i.size(0)):
            sampled_sim = torch.cat((sims_matrix_t2i[i,top1_idxs[i]].reshape(-1), sims_matrix_t2i[i,neg_idxs[i]]))
            sampled_idx = torch.cat((top1_idxs[i].reshape(-1), neg_idxs[i]))
            sampled_sims_matrix_t2i.append(sampled_sim)
            sampled_sims_idx_t2i.append(sampled_idx)
        sampled_sims_matrix_t2i = torch.stack(sampled_sims_matrix_t2i) # shape=(5000, k_tta)
        sampled_sims_idx_t2i = torch.stack(sampled_sims_idx_t2i) # shape=(5000, k_tta)
        print("      number of sample after pos/neg sampling: {}".format(sampled_sims_matrix_t2i.size()))

        topk_idx = sampled_sims_idx_t2i

    print(f"     preprocess_tta_coefficients  end")
    return topk_idx, recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list
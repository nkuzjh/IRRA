import math

import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F

import utils



from tta.utils import preprocess_tta_coefficients
import time
import datetime



@torch.enable_grad()
def test_time_adapt_itm(model, optimizer, scaler, epoch, device, scheduler, config, dataloader):
    # model.eval()
    model.train()

    start_time = time.time()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('entropy', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('uncertainty', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = '      TTA Epoch: [{}]'.format(epoch)
    print_freq = 100

    for iter, (encoder_output, encoder_att, text_embeds, text_atts, uncertainty, proba_top1_sim, proba_inversed_sim) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):

        encoder_output = encoder_output.reshape(-1, encoder_output.size(-2), encoder_output.size(-1)).to(device)
        encoder_att = encoder_att.reshape(-1, encoder_att.size(-1)).to(device)
        text_embeds = text_embeds.reshape(-1, text_embeds.size(-2), text_embeds.size(-1)).to(device)
        text_atts = text_atts.reshape(-1, text_atts.size(-1)).to(device)
        uncertainty = uncertainty.to(device)
        if config.get('uncertainty', None) == 'inversed_recall_proba' and config.get('uncertainty_temper_is_learnable', False):
            proba_top1_sim =  proba_top1_sim.to(device)
            proba_inversed_sim = proba_inversed_sim.to(device)

        with torch.cuda.amp.autocast(enabled=True):#(device_type='cuda', dtype=torch.bfloat16):
            output = model.get_cross_embeds(
                encoder_output,#([24, 49, 1024])
                encoder_att,#([24, 49])
                text_embeds = text_embeds,#([24, 56, 768])
                text_atts = text_atts#([24, 56])
            )[:, 0, :] # (bs*k_tta, sequence, last_hidden_states)[:, 0, :] -> (bs*tta, last_hidden_states)
            ### 如果使用prompt learning增加一个随机初始化的token，这里能否取index=0的last_hidden_states作为itm结果？是否应该用index=1(即原本的cls token位置)替代？需要结合CoOp代码看一下是如何实现的，使用哪个token作为最终结果。
            ### 我在text_embeds之前加入随机初始化的embedding作为prompt learning的初始值，token数量从1-12进行exp，itm.output使用原cls token位置的feature作为结果logits输出
            logits = model.itm_head(output) # (bs*tta, 2)
            logits = logits.reshape(-1, config['k_tta'], 2) # (bs, tta, 2)
            score = logits[..., 1] # (bs, tta)
            entropy = -(F.softmax(score * config['score_temper'], dim=-1) * F.log_softmax(score * config['score_temper'], dim=-1)).sum(-1)
            if config.get('uncertainty', None) == 'inversed_recall_proba' and config.get('uncertainty_temper_is_learnable', False):
                uncertainty_temper = model.uncertainty_temper
                uncertainty = torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) * uncertainty_temper )
            if config.get('uncertainty', None) is not None:
                loss = entropy / uncertainty + uncertainty
            else:
                loss = entropy
            loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
        optimizer.zero_grad()

        metric_logger.update(entropy=entropy.mean().item())
        # metric_logger.update(uncertainty=uncertainty.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("     Averaged stats:", metric_logger.global_avg())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('     itm tta time {}'.format(total_time_str))
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


# @torch.enable_grad()
# def test_time_adapt_itm_itc(model, tokenizer, optimizer, scaler, epoch, device, scheduler, config, tta_img_aug_loader):#sims_matrix, image_embeds, text_embeds, text_atts):

#     ##### evaluate_itc #####
#     model.eval()
#     with torch.no_grad():
#         start_time = time.time()

#         print('     Computing text features for tta_itm')
#         texts = tta_img_aug_loader.dataset.text
#         num_text = len(texts)
#         text_bs = config['batch_size_test_text']
#         text_embeds = []
#         text_atts = []
#         text_feats = []
#         for i in range(0, num_text, text_bs):
#             text = texts[i: min(num_text, i + text_bs)]
#             text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
#             text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
#             text_feat = model.get_text_feat(text_embed)
#             text_feat = F.normalize(text_feat, dim=-1)

#             text_embeds.append(text_embed)
#             text_atts.append(text_input.attention_mask)
#             text_feats.append(text_feat)

#         text_embeds = torch.cat(text_embeds, dim=0)#[1978, 56, 768])
#         text_atts = torch.cat(text_atts, dim=0)#([1978, 56])
#         text_feats = torch.cat(text_feats, dim=0)#[1978, 2048])

#         print('     Computing image features for tta_itm')
#         image_embeds = []
#         image_feats = []
#         for image, pose, img_id in tta_img_aug_loader:
#             image = image.to(device)
#             image_embed, _ = model.get_vision_embeds(image)

#             if config.get('be_pose_img', False):
#                 pose = pose.to(device)
#                 if model.be_pose_conv:
#                     pose = model.pose_conv(pose)

#                 pose_embed, _ = model.get_vision_embeds(pose)
#                 image_embed = model.pose_block(image_embed, pose_embed)

#             image_feat = model.get_image_feat(image_embed)
#             image_feat = F.normalize(image_feat, dim=-1)
#             image_embeds.append(image_embed)
#             image_feats.append(image_feat)

#         image_embeds = torch.cat(image_embeds, dim=0)#[1978, 50, 1024])
#         image_feats = torch.cat(image_feats, dim=0)#[1978, 2048])

#         sims_matrix = image_feats @ text_feats.t()
#         sims_matrix_t2i = sims_matrix.t()

#         total_time = time.time() - start_time
#         total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#         print('     Computing features time {}'.format(total_time_str))

#     print(f'     Computing features: sims_matrix_t2i={sims_matrix_t2i.shape}, image_embeds={image_embeds.shape}, text_embeds={text_embeds.shape}, text_atts={text_atts.shape}')
#     # sims_matrix_t2i, image_embeds, text_embeds, text_atts
#     ##### evaluate_itc #####


#     ##### test_time_adapt_itm #####
#     model.train()

#     start_time = time.time()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('entropy', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     # metric_logger.add_meter('uncertainty', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     header = '      TTA Epoch: [{}]'.format(epoch)
#     print_freq = 100

#     print("     ### Compute ITC Uncertainty")
#     recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list = preprocess_tta_coefficients(config, sims_matrix_t2i)
#     if config.get('sample_selection', 'all') == 'top1':
#         sims_matrix_t2i = sims_matrix_t2i[ss_idxs_list]
#         text_embeds = text_embeds[ss_idxs_list]
#         text_atts = text_atts[ss_idxs_list]
#         # self.labels = [labels[i] for i in ss_idxs_list]
#         # self.recall_types = [recall_types[i] for i in ss_idxs_list]
#         uncertaintys_list = [uncertaintys_list[i] for i in ss_idxs_list]
#         if 1:# if config.get('uncertainty_temper_is_learnable', False) == True:
#             proba_top1_sim_list = [proba_top1_sim_list[i] for i in ss_idxs_list]
#             proba_inversed_sim_list = [proba_inversed_sim_list[i] for i in ss_idxs_list]

#     print(f"     samples after sample selection & uncertainty computing : {len(ss_idxs_list)}")

#     print(f"     batchsize: {config['batch_size_tta']}")
#     iterations = int(len(ss_idxs_list) // config['batch_size_tta'])
#     print(f"     iterations: { iterations }")
#     print("     ### Start ITM Test Time Adaptation")
#     for iter in metric_logger.log_every(range(iterations), print_freq, header):
#         topk_sim, topk_idx = sims_matrix_t2i[iter*config['batch_size_tta']: (iter+1)*config['batch_size_tta']].topk(k=config['k_tta'], dim=-1)
#         encoder_output = image_embeds[topk_idx]
#         encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long)
#         text_embed = text_embeds[iter*config['batch_size_tta']: (iter+1)*config['batch_size_tta']].repeat(config['k_tta'], 1, 1)
#         text_att = text_atts[iter*config['batch_size_tta']: (iter+1)*config['batch_size_tta']].repeat(config['k_tta'], 1)
#         uncertainty = torch.Tensor(uncertaintys_list[iter*config['batch_size_tta']: (iter+1)*config['batch_size_tta']])
#         proba_top1_sim = torch.Tensor(proba_top1_sim_list[iter*config['batch_size_tta']: (iter+1)*config['batch_size_tta']])
#         proba_inversed_sim = torch.Tensor(proba_inversed_sim_list[iter*config['batch_size_tta']: (iter+1)*config['batch_size_tta']])

#         encoder_output = encoder_output.reshape(-1, encoder_output.size(-2), encoder_output.size(-1)).to(device)
#         encoder_att = encoder_att.reshape(-1, encoder_att.size(-1)).to(device)
#         text_embed = text_embed.reshape(-1, text_embed.size(-2), text_embed.size(-1)).to(device)
#         text_att = text_att.reshape(-1, text_att.size(-1)).to(device)
#         uncertainty = uncertainty.to(device)
#         if config.get('uncertainty', None) == 'inversed_recall_proba' and config.get('uncertainty_temper_is_learnable', False):
#             proba_top1_sim =  proba_top1_sim.to(device)
#             proba_inversed_sim = proba_inversed_sim.to(device)

#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             # print(iter)
#             # print(encoder_output.shape, encoder_att.shape, text_embed.shape, text_att.shape)
#             output = model.get_cross_embeds(
#                 encoder_output,#([24, 50, 1024])
#                 encoder_att,#([24, 50])
#                 text_embed,#([24, 56, 768])
#                 text_att#([24, 56])
#             )[:, 0, :] # (bs*k_tta, sequence, last_hidden_states)[:, 0, :] -> (bs*tta, last_hidden_states)
#             ### 如果使用prompt learning增加一个随机初始化的token，这里能否取index=0的last_hidden_states作为itm结果？是否应该用index=1(即原本的cls token位置)替代？需要结合CoOp代码看一下是如何实现的，使用哪个token作为最终结果。
#             ### 我在text_embeds之前加入随机初始化的embedding作为prompt learning的初始值，token数量从1-12进行exp，itm.output使用原cls token位置的feature作为结果logits输出
#             logits = model.itm_head(output) # (bs*tta, 2)
#             logits = logits.reshape(-1, config['k_tta'], 2) # (bs, tta, 2)
#             score = logits[..., 1] # (bs, tta)
#             entropy = -(F.softmax(score * config['score_temper'], dim=-1) * F.log_softmax(score * config['score_temper'], dim=-1)).sum(-1)
#             if config.get('uncertainty', None) == 'inversed_recall_proba' and config.get('uncertainty_temper_is_learnable', False):
#                 uncertainty_temper = model.uncertainty_temper
#                 uncertainty = torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) * uncertainty_temper )
#             if config.get('uncertainty', None) is not None:
#                 loss = entropy / uncertainty + uncertainty
#             else:
#                 loss = entropy
#             loss = loss.mean()

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scale = scaler.get_scale()
#         scaler.update()
#         skip_lr_sched = (scale > scaler.get_scale())
#         if not skip_lr_sched:
#             scheduler.step()
#         optimizer.zero_grad()

#         metric_logger.update(entropy=entropy.mean().item())
#         # metric_logger.update(uncertainty=uncertainty.item())
#         metric_logger.update(loss=loss.item())
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("     Averaged stats:", metric_logger.global_avg())
#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('     itm tta time {}'.format(total_time_str))
#     ##### test_time_adapt_itm #####

#     return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


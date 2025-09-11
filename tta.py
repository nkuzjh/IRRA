from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs



import time
import datetime
import yaml
import logging
import random
import json
from easydict import EasyDict as edict

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from utils.iotools import save_train_configs
from datasets.bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset
from datasets.build import build_transforms, __factory

from tta.utils import preprocess_tta_coefficients
from tta.dataset import IRRA_tta_dataset, create_tta_loader
from tta.optim import configure_tta_model#, AttrDict, create_tta_optimizer, create_tta_scheduler
from solver import build_optimizer, build_lr_scheduler



@torch.enable_grad()
def do_tta(args, config, model, tta_loader, optimizer, scaler, epoch, device, scheduler):
    logger = logging.getLogger("IRRA.tta")
    logger.info("Enter tta...")
    model = model.train()

    start_time = time.time()

    loss_iter_periods = []
    entropy_iter_periods = []
    lr_iter_periods = []
    for iter, (gids_topk, imgs_topk, qids_repeatk, captions_repeatk, uncertainty, proba_top1_sim, proba_inversed_sim) in enumerate(tta_loader):
        imgs_topk = imgs_topk.reshape(-1, imgs_topk.size(-3), imgs_topk.size(-2), imgs_topk.size(-1)).to(device)# [16, 8, 3, 384, 128]) -> [128, 3, 384, 128])
        captions_repeatk = captions_repeatk.to(device)#16, 77 #.reshape(-1, captions_repeatk.size(-1)).#[16, 8, 77]) -> ([128, 77])
        uncertainty = uncertainty.to(device)#([16])
        if config.get('uncertainty', None) == 'inversed_recall_proba' and config.get('uncertainty_temper_is_learnable', False):
            proba_top1_sim =  proba_top1_sim.to(device)#([16])
            proba_inversed_sim = proba_inversed_sim.to(device)#([16])

        with torch.no_grad():
            gfeat = model.encode_image(imgs_topk) # image features
        with torch.cuda.amp.autocast(enabled=True):
            qfeat = model.encode_text(captions_repeatk) # text features
            # if config.get('compute_entropy_with_norm_cos_sim', False):
            gfeat = F.normalize(gfeat, p=2, dim=1)#[128, 512])
            qfeat = F.normalize(qfeat, p=2, dim=1)#[16, 512])

            gfeat = gfeat.reshape(-1,config['k_tta'], gfeat.size(-1))#16,512
            # qfeat = qfeat.reshape(-1,config['k_tta'], gfeat.size(-1))
            cos_sims = []
            for qfeat_, gfeat_ in zip(qfeat, gfeat):
                cos_sim = qfeat_ @ gfeat_.t()#[512]@[k_tta,512].t() = [k_tta]
                cos_sims.append(cos_sim)
            cos_sims  = torch.stack(cos_sims)#16, k_tta
            # cos_sims = qfeat @ gfeat.t()#[128, 128])
            # cos_sims = cos_sims.reshape(-1, config['k_tta'], 1)
            cos_sims_inter = cos_sims / args.temperature

            entropy = -(F.softmax(cos_sims_inter, dim=-1) * F.log_softmax(cos_sims_inter, dim=-1)).sum(-1)
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

        if (iter + 1) % args.log_period == 0:
            print(f"     Epoch[{epoch}] Iteration[{iter + 1}/{len(tta_loader)}], entropy: {entropy.mean().item():.4f}, loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.2e}")
            loss_iter_periods.append(loss.item())
            entropy_iter_periods.append(entropy.mean().item())
            lr_iter_periods.append(optimizer.param_groups[0]["lr"])

    print(f"     Averaged stats: entropy: {entropy.mean().item():.4f}, loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.2e}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('     itm tta time {}'.format(total_time_str))
    return {
        'entropy': np.mean(entropy_iter_periods),
        'loss': np.mean(loss_iter_periods),
        'lr': np.mean(lr_iter_periods),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA TTA")
    parser.add_argument("--config_file", default='tta_configs/ham_cuhk_tta/exp_debug.yaml')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        args = edict( yaml.load(f, Loader=yaml.FullLoader) )
    args.output_dir = os.path.join(args.output_dir, f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")[:-1]}')

    save_train_configs(args.output_dir, args)
    config = vars(args)


    # utils.init_distributed_mode(args)
    print('Not using distributed mode')
    args.distributed = False


    print("### Hyper-parameters:")
    print("     output_dir:", args.output_dir)

    seed = 42#args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    print("     seed:", seed)

    device = torch.device(args.device)
    print("     device:", device)

    tbs = ["epoch", "R1", "R5", "R10", "mAP", "mINP"]
    table = PrettyTable(tbs)
    for tb in tbs[1:]:
        table.custom_format[tb] = lambda f, v: f"{v:.3f}"

    print("     batch_size_tta:", config['batch_size_tta'])
    print("     epochs:", config['num_epoch'])
    print("     lr:", config['lr'])


    print("### Creating test dataset")
    dataset = __factory[args.dataset_name](root=args.root_dir)
    test_transforms = build_transforms(img_size=args.img_size, aug=False, is_train=False)
    ds = dataset.test
    test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], test_transforms)
    test_txt_set = TextDataset(ds['caption_pids'], ds['captions'], text_length=args.text_length)
    print(f"     test_txt_set: {len(test_txt_set)}    test_img_set: {len(test_img_set)}")#txt=2000 img=1000

    num_workers = args.num_workers
    test_img_loader = DataLoader(
        test_img_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_txt_loader = DataLoader(
        test_txt_set,
        batch_size=args.test_batch_size,#512
        shuffle=False,
        num_workers=num_workers
    )
    print(f"     test_txt_loader: {len(test_txt_loader)}    test_img_loader: {len(test_img_loader)}")


    print("### Creating model")
    num_classes = len(dataset.train_id_container)#train_id_container3701
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.ckpt_dir))
    model.to(device)
    print("     Total Params Sum: ", sum(p.numel() for p in model.parameters()))# if p.requires_grad))


    print("### Zero-Shot Score: ")
    test_result, recall1, similarity, qfeats, gfeats, qids, gids, captions, imgs = do_inference(model, test_img_loader, test_txt_loader)
    #qfeats=torch.Size([2000, 512]) gfeats=torch.Size([1000, 512]) #qids=2000 gids=1000 #len(captions[0:4])=512,77+512,77+512,77+464,77=2000,77 len(imgs)=2=(torch.Size([512, 3, 384, 128]), torch.Size([488, 3, 384, 128]))
    table.add_row([
        -999, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
    ])
    print(table)


    if args.tta:
        print("### TTA:")

        print("### Compute Cos Similarity Uncertainty")
        if config.get('compute_uncertainty_with_norm_cos_sim', False):
            sims_matrix_t2i = similarity.cpu()
        else:
            sims_matrix_t2i = qfeats.cpu() @ gfeats.t().cpu()
        # task_name = config['output_dir'].split('/')[1]
        # np.save(f'debug_embeddings/{task_name}/sims_matrix_t2i.npy', sims_matrix_t2i.detach().cpu().numpy())
        sims_topk_matrix_t2i, recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list  = preprocess_tta_coefficients(config, sims_matrix_t2i)
# task_name = config['output_dir'].split('/')[1]
# ss=np.array(gids)
# print(ss.shape)
# task_name = config['output_dir'].split('/')[1]
# np.save(f'debug_embeddings/{task_name}/gids.npy', ss)

        print("### Creating TTA dataset")
        is_img_aug = config.get('is_image_augmentation', False)
        test_transforms = build_transforms(img_size=args.img_size, aug=is_img_aug, is_train=is_img_aug)# (aug=F,is_train=T)->hflip; (aug=T,is_train=T)->train_aug; (is_train=F)->no_aug;
        tta_dataset = IRRA_tta_dataset(
            config,
            test_transforms,
            sims_topk_matrix_t2i.cpu(),
            qids.cpu(), gids.cpu(), torch.cat(captions, dim=0).cpu(), torch.cat(imgs, dim=0).cpu(),
            recall_types,
            ss_idxs_list,
            uncertaintys_list,
            proba_top1_sim_list,
            proba_inversed_sim_list,
        )
        print(f"     tta_dataset: {len(tta_dataset)}")
        # sample = next(iter(tta_dataset))
        # print(sample)

        print("### Creating tta dataloader")
        tta_loader = create_tta_loader(
            [tta_dataset],
            batch_size=[config['batch_size_tta']],
            num_workers=[4],
            is_trains=[True],
            collate_fns=[None]
        )[0]
        print(f"     tta_loader: {len(tta_loader)}")
        # batch = next(iter(tta_loader))
        # print(batch)


        print("### Configure adapted weights")
        # arg_tm = AttrDict(config['tta_model'])
        model = configure_tta_model(config, model)
        print("     TTA Dropout Modules: \r\n", [(n,m,m.training) for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True] )
        print("     TTA Require Gradient Params: \r\n", [(n, p.shape) for n,p in model.named_parameters() if p.requires_grad] )
        print("     TTA Dropout Modules Number: \r\n", sum([ 1 for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True ]) )
        print("     TTA Require Gradient Params Sum: \r\n", sum(p.numel() for p in model.parameters() if p.requires_grad) )

        # arg_opt = AttrDict(config['optimizer'])
        # optimizer = create_tta_optimizer(arg_opt, model)
        # arg_sche = AttrDict(config['schedular'])
        # arg_sche['step_per_epoch'] = math.ceil( len(tta_dataset) / config['batch_size_tta'] )
        # lr_scheduler = create_tta_scheduler(arg_sche, optimizer)
        optimizer = build_optimizer(args, model)
        scheduler = build_lr_scheduler(args, optimizer)

        scaler = GradScaler()  # bf16


        print("### Start Test Time Adaptation")
        start_time = time.time()
        best = 0
        best_epoch = 0
        best_logs = {}
        for epoch in range(args.num_epoch):
            train_stats = do_tta(args, config, model, tta_loader, optimizer, scaler, epoch, device, scheduler)

            test_result, recall1, similarity, qfeats, gfeats, qids, gids, captions, imgs = do_inference(model, test_img_loader, test_txt_loader)
            print("### TTA Eval Score: ")
            table.add_row([
                epoch, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
            ])
            print(table)

            logs = {'epo': epoch}
            for k, v in test_result.items():
                logs[k] = np.around(v, 3)
            for k, v in train_stats.items():
                logs[k] = float(v)
            print('     logs: ', logs)
            for k, v in logs.items():
                logs[k] = str(v)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(logs) + "\n")

            result = test_result['R1']
            if result > best:
                best = result
                best_epoch = epoch
                best_logs = logs

            torch.cuda.empty_cache()

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"best epoch {best_epoch} : {best_logs}")
        print(f"### best epoch {best_epoch} : {best_logs}")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('### Time {}'.format(total_time_str))




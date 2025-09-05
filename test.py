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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.ckpt_dir))#, 'best.pth'))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)



# ### python test.py --config_file 'configs/irra_cuhk/configs.yaml'
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 73.181 | 89.847 | 93.697 | 66.114 | 50.290 |
# +------+--------+--------+--------+--------+--------+
# ### python test.py --config_file 'configs/ham_cuhk/configs.yaml'
# best0.pth
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 70.598 | 86.891 | 91.780 | 63.391 | 47.063 |
# +------+--------+--------+--------+--------+--------+
# best1.pth
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 69.087 | 86.160 | 91.098 | 62.256 | 46.191 |
# +------+--------+--------+--------+--------+--------+
# best2.pth
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 68.925 | 86.501 | 91.163 | 62.359 | 46.155 |
# +------+--------+--------+--------+--------+--------+


# ### python test.py --config_file 'configs/irra_icfg/configs.yaml'
# +------+--------+--------+--------+--------+-------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP |
# +------+--------+--------+--------+--------+-------+
# | t2i  | 63.417 | 80.356 | 85.767 | 38.042 | 7.911 |
# +------+--------+--------+--------+--------+-------+
# ### python test.py --config_file 'configs/ham_icfg/configs.yaml'
# best0.pth
# +------+--------+--------+--------+--------+-------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP |
# +------+--------+--------+--------+--------+-------+
# | t2i  | 60.641 | 77.509 | 83.268 | 35.542 | 6.004 |
# +------+--------+--------+--------+--------+-------+
# best1.pth
# +------+--------+--------+--------+--------+-------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP |
# +------+--------+--------+--------+--------+-------+
# | t2i  | 60.273 | 76.728 | 82.633 | 35.228 | 5.902 |
# +------+--------+--------+--------+--------+-------+
# best2.pth
# +------+--------+--------+--------+--------+-------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP |
# +------+--------+--------+--------+--------+-------+
# | t2i  | 58.943 | 76.108 | 82.200 | 34.721 | 5.890 |
# +------+--------+--------+--------+--------+-------+


# ### python test.py --config_file 'configs/irra_rstp/configs.yaml'
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 60.250 | 81.650 | 88.150 | 47.198 | 25.334 |
# +------+--------+--------+--------+--------+--------+
# ### python test.py --config_file 'configs/ham_rstp/configs.yaml'
# best0.pth
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 59.400 | 80.000 | 87.000 | 44.121 | 20.555 |
# +------+--------+--------+--------+--------+--------+
# best1.pth
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 56.800 | 78.800 | 86.750 | 43.217 | 20.267 |
# +------+--------+--------+--------+--------+--------+
# best2.pth
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 58.700 | 79.550 | 87.150 | 44.098 | 20.910 |
# +------+--------+--------+--------+--------+--------+


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3,1,0,2'

from utils import *

import traceback
import numpy as np
from trainers import VAETrainer, WeakTrainer, TeacherTrainer
from config import *

logger = init_logger(log_name='echo')
seed_everything(1996)


cfg = WeakConfig()

for sources in ['books']:  # target
    for fold in [1]:
        cfg.fold = fold
        cfg.dataset = [sources]
        cfg.epoch = 10
        cfg.batch = 32
        cfg.lr = 1e-5
        cfg.max_len = 300
        cfg.use_checkpoints = False
        trainer = VAETrainer(cfg, use_teacher=True)
        trainer.train_weak(cfg)
        trainer.vis(cfg)
        # trainer = WeakTrainer(cfg)
        # trainer.train_weak_lora(cfg)
        # trainer.get_overconfidence_samples(cfg)
        # ptrainer = TeacherTrainer(cfg)
        # ptrainer.train_weak(cfg)
        # for target in ['books', 'dvd', 'electronics', 'kitchen']:
        #     if target != sources:
        #         cfg.batch = 32
        #         cfg.target = target
        #         cfg.max_len = 300
        #         trainer.test_CD_weak(cfg)
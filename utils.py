import tokenize

import torch
import torch.nn as nn
import numpy as np
import os
import random
from Dataloader import *
from itertools import combinations
from config import *
import logging

def init_logger(log_name: str = "echo", log_file='log', log_file_level=logging.NOTSET):
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    file_handler = logging.FileHandler(log_file, encoding="utf8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def seed_everything(seed=1996):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def to_multiCUDA(cfg, model):
    model = torch.nn.DataParallel(model, cfg.cudas)
    return model

def cal_Correlation(cfg):
    ''' 计算训练集的单词（组）与label的共现频率 共现次数/总句子数
    :param cfg:
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)
    train_loader = WeakDataset(cfg, tokenizer, 'train', False)
    Terms0, Terms1 = [], []
    Term2time0, Term2time1 = {}, {}
    for data in train_loader.datas:
        tokens = tokenizer.tokenize(data[0])
        tokens = [each for each in tokens if '#' not in each and len(each)>1]   # 去除子词
        if int(data[1]) == 1:
            Terms1.append(tokens)
        else:
            Terms0.append(tokens)
    for tokens in Terms0:
        for token in tokens:
            if token in Term2time0.keys():
                Term2time0[token] += 1
            else:
                Term2time0[token] = 1
    for tokens in Terms1:
        for token in tokens:
            if token in Term2time1.keys():
                Term2time1[token] += 1
            else:
                Term2time1[token] = 1
    # 接下来看数量不一致的单词
    Times0 = sorted(Term2time0.items(), key=lambda a:a[1], reverse=True)
    Times1 = sorted(Term2time1.items(), key=lambda a:a[1], reverse=True)
    Times0 = [each for each in Times0 if each[1]>5]
    Times1 = [each for each in Times1 if each[1]>5]
    print(Times0, Times1)

# cfg = WeakConfig()
# cal_Correlation(cfg)
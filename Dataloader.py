import copy
import math
import os
import json
import random
import traceback

import pandas as pd
from transformers import *

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
import numpy as np
from tqdm import tqdm
import pickle
from config import WeakConfig
from models import *



class BaseDataset():
    def __init__(self, cfg, tokenizer, train, target=False, fold=1):
        assert fold in [1,2,3,4,5]
        self.fold = fold
        super(BaseDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.target = target
        if target:
            self.dataset = [cfg.target]
        else:
            self.dataset = cfg.dataset
        self.train = train  # train / dev / test
        self.ipts = 0
        self.init()
        self.steps = self.get_steps()

    def init(self):
        self.load_data()
        self.data_iter = iter(self.datas)
        print('====init DataIter {} with Steps {}==='.format(' '.join(self.dataset), self.get_steps()))

    def __iter__(self):
        return self

    def get_steps(self):
        return len(self.datas) // self.cfg.batch

    def reset(self):
        self.data_iter = iter(self.datas)

    def load_data(self):
        self.datas = []
        if self.target:
            assert self.train == 'test'
            file = open('dataset/{}/test'.format(self.cfg.target), 'rb')
            datas = pickle.load(file)
            for i in range(len(datas[0])):
                text, label = datas[0][i], datas[1][i]
                self.datas.append([text.strip().lower().replace('\n', ' ').replace('\t', ' '),
                                   int(label), 'test_' + str(i)])  # index = dataset name + i
        else:
            for dataset in self.cfg.dataset:
                if self.train == 'train':
                    file = open('dataset/{}/fold-{}/train'.format(dataset, self.fold), 'rb')
                if self.train == 'dev':
                    file = open('dataset/{}/fold-{}/dev'.format(dataset, self.fold), 'rb')
                if self.train == 'test':
                    file = open('dataset/{}/test'.format(dataset, self.fold), 'rb')
                datas = pickle.load(file)
                for i in range(len(datas[0])):
                    text, label = datas[0][i], datas[1][i]
                    self.datas.append([text.strip().lower().replace('\n', ' ').replace('\t', ' '),
                                       int(label), dataset+'_'+str(i)])  # index = dataset name + i
        random.shuffle(self.datas)
        self.index = []
        for data in self.datas:
            index = int(data[-1].split('_')[1])
            self.index.append(index)

    def padding(self, max_token_len, texts, pad_token):
        resutls = []
        for text in texts:
            text = text[:self.cfg.max_len]
            if len(text) < max_token_len:
                text = text + [pad_token]*(max_token_len-len(text))
            else:
                text = text[:max_token_len]
            resutls.append(text)
        return resutls

    # the followings need to be rewrote
    def __next__(self):
        pass

    def get_batch(self):
        pass

class WeakDataset(BaseDataset):
    def __init__(self, cfg, tokenizer, teacher_tokenizer, train, target=False, fold=1):
        super(WeakDataset, self).__init__(cfg, tokenizer, train=train, target=target, fold=fold)
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.cfg = cfg
        self.train = train
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.ipts = 0
        self.steps = self.get_steps()

    def get_batch(self):
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.cfg.batch:
                break
        if len(batch_data) < 1:
            return None
        # base_texts is the teacher's id
        texts, labels, base_texts, masks = [], [], [], []
        if 'bert' in self.cfg.base_model:
            for data in batch_data:
                text, label = data[0], data[1]
                tokens = self.tokenizer.tokenize(text)[:self.cfg.max_len]
                base_tokens = self.teacher_tokenizer.tokenize(text)[:self.cfg.max_len]
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                base_tokens = [self.tokenizer.cls_token] + base_tokens + [self.tokenizer.sep_token]
                mask = [1]*len(base_tokens)
                base_tokens = self.tokenizer.convert_tokens_to_ids(base_tokens)
                tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                texts.append(tokens)
                base_texts.append(base_tokens)
                labels.append(label)
                masks.append(mask)
            max_token_len = max(len(each) for each in texts)
            texts = self.padding(max_token_len, texts, self.PAD_ID)
            base_texts = self.padding(max_token_len, base_texts, self.PAD_ID)
            masks = self.padding(max_token_len, masks, 0)
            return {
                'input_ids': torch.LongTensor(texts),
                'base_ids': torch.LongTensor(base_texts),
                'labels': torch.LongTensor(labels),
                'attention_mask': torch.LongTensor(masks)
            }
        else:   # 非bert系的模型，比如opt350m
            text = [data[0] for data in batch_data]
            labels = [data[1] for data in batch_data]
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.cfg.max_len)
            teacher_inputs = self.teacher_tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.cfg.max_len)
            return {
                'input_ids': inputs['input_ids'],
                'base_ids': teacher_inputs['input_ids'],
                'labels': torch.LongTensor(labels),
                'attention_mask': inputs['attention_mask']
            }


    # next operation for Python iterator, it's necessary for your own iter
    def __next__(self):
        if self.ipts is None:
            self.reset()
        self.ipts = self.get_batch()  # each iter get a batch
        if self.ipts is None:
            raise StopIteration
        else:
            return self.ipts

class PromptDataset(BaseDataset):
    def __init__(self, cfg, tokenizer, train, target=False, fold=1):
        super(PromptDataset, self).__init__(cfg, tokenizer, train=train, target=target, fold=fold)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.train = train
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.ipts = 0
        self.steps = self.get_steps()
        self.generate_prompt()

    def generate_prompt(self):
        data2index = {}
        for dataset in self.cfg.dataset:
            with open('confidence/{}-{}.json'.format(dataset, self.cfg.fold), 'r') as rf:
                index2con = json.load(rf)
            top_index = sorted(index2con.items(), key=lambda a:a[1], reverse=True)
            top_index = top_index[:int(len(top_index)*0.4)]   # 取40%
            top_index = [each[0] for each in top_index]
            data2index[dataset] = top_index
        new_datas = []
        for data in self.datas:
            index = data[-1].split('_')
            if index[1] in data2index[index[0]]:   # 选择置信度高的样本训练一个biased分类器
                if int(data[1]) == 1:
                    new_datas.append(data + ['It is good.'])
                else:
                    new_datas.append(data + ['It is bad.'])
        print('========new samples with high confidence:{}-{}=========='.format(len(new_datas), len(self.datas)))
        self.datas = new_datas
        self.reset()


    def get_batch(self):
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.cfg.batch:
                break
        if len(batch_data) < 1:
            return None
        # base_texts is the x' of IntegratedGrads Method, all pad_tokens
        texts, labels, base_texts, masks = [], [], [], []
        for data in batch_data:
            text, label = data[0], data[1]
            # tokens = self.tokenizer.tokenize(prompt+'.'+text)[:self.cfg.max_len]
            tokens = self.tokenizer.tokenize(text)[:self.cfg.max_len]
            # tokens = self.tokenizer.tokenize(text+'. '+prompt)
            # base_tokens = [self.tokenizer.pad_token] * len(tokens)
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            # base_tokens = [self.tokenizer.cls_token] + base_tokens + [self.tokenizer.sep_token]
            mask = [1]*len(tokens)
            # base_tokens = self.tokenizer.convert_tokens_to_ids(base_tokens)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            texts.append(tokens)
            # base_texts.append(base_tokens)
            labels.append(label)
            masks.append(mask)
        max_token_len = max(len(each) for each in texts)
        texts = self.padding(max_token_len, texts, self.PAD_ID)
        # base_texts = self.padding(max_token_len, base_texts, self.PAD_ID)
        masks = self.padding(max_token_len, masks, 0)
        return {
            'input_ids': torch.LongTensor(texts),
            'base_ids': torch.LongTensor(base_texts),
            'labels': torch.LongTensor(labels),
            'attention_mask': torch.LongTensor(masks)
        }

    # next operation for Python iterator, it's necessary for your own iter
    def __next__(self):
        if self.ipts is None:
            self.reset()
        self.ipts = self.get_batch()  # each iter get a batch
        if self.ipts is None:
            raise StopIteration
        else:
            return self.ipts

if __name__ == '__main__':
    cfg = WeakConfig()
    model, tokenizer = load_backbone(cfg)
    loader = WeakDataset(cfg=cfg, tokenizer=tokenizer, teacher_tokenizer=tokenizer, train='train', target=False)
    for ipt in loader:
        # print(ipt)
        pass

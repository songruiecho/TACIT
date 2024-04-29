import numpy as np

from config import WeakConfig
import torch.optim as optim
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from utils import *
import torch
import torch.nn as nn
# import seaborn as sns
# import matplotlib.pyplot as plt
from Dataloader import *
from models import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from accelerate import Accelerator
from data_parallel import BalancedDataParallel

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training, PeftModel
from transformers import Trainer

logger = init_logger(log_name='echo')

class WeakTrainer(nn.Module):
    '''
    Trainer for Bert-based Model
    '''
    def __init__(self, cfg):
        super(WeakTrainer, self).__init__()
        self.cfg = cfg
        self.model = WeakClassifier(cfg)
        if len(cfg.cudas) > 1:
            self.model = to_multiCUDA(cfg, self.model)
            self.tokenizer = self.model.module.tokenizer
        else:
            self.tokenizer = self.model.tokenizer
        # load teacher tokenizer
        if self.cfg.base_model == 'roberta':
            teacher_path = self.cfg.root + 'distilroberta_en_base/'
        else:
            teacher_path = self.cfg.root + 'distilbert_en_base/'
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        self.model.cuda()
        model_dir_name = '{}_on_{}_fold_{}'.format(self.cfg.base_model, self.cfg.dataset, self.cfg.fold)
        self.save_path = 'save_models/weakclassifier/' + model_dir_name
        self.train_loader = WeakDataset(cfg, self.tokenizer, self.teacher_tokenizer, 'train', False, fold=cfg.fold)
        self.test_loader = WeakDataset(cfg, self.tokenizer, self.teacher_tokenizer, 'dev', False, fold=cfg.fold)

    def train_weak(self, cfg):
        optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, eps=1e-8)
        # train the model
        best_metrics = 0.0
        for epoch in range(cfg.epoch):
            total_loss = 0.0
            probs = []
            self.model.train()
            for step, ipt in enumerate(self.train_loader):
                ipt = {k: v.cuda(non_blocking=True) for k, v in ipt.items()}
                out, loss = self.model(ipt)
                prob = out.logits.cpu().detach().numpy()
                if len(cfg.cudas) > 1:  # check-multi GPUs
                    loss = torch.sum(loss, dim=0)
                total_loss += loss.data
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if step % 50 == 0:
                    logger.info("epoch-{}, step-{}/{}, loss:{}".format(epoch, step, self.train_loader.steps, loss.data))
                if step == 0:
                    probs = prob
                else:
                    probs = np.concatenate([probs, prob], axis=0)
            # 模型测试
            targets, preds = self.eval_weak(cfg, self.model, self.test_loader)
            metric = accuracy_score(targets, preds)
            if metric > best_metrics:
                best_metrics = metric
                self.model.module.save(self.save_path)
            logger.info('====BEST results:{}, resuls:{}, loss:{}====='.format(best_metrics, metric, total_loss))

    def train_weak_lora(self, cfg):
        ''' 使用lora计算减少内存使用
        :return:
        '''
        # 配置 Lora 模型的参数
        self.model = prepare_model_for_int8_training(self.model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["self_attn.k_proj", "self_attn.v_proj"]   # lora分解作用的层
        )
        self.model = get_peft_model(self.model, lora_config)

        optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, eps=1e-8)
        # train the model
        best_metrics = 0.0
        for epoch in range(cfg.epoch):
            total_loss = 0.0
            probs = []
            self.model.train()
            for step, ipt in enumerate(self.train_loader):
                ipt = {k: v.cuda(non_blocking=True) for k, v in ipt.items()}
                out, loss = self.model(ipt)
                prob = out.logits.cpu().detach().numpy()
                if len(cfg.cudas) > 1:  # check-multi GPUs
                    loss = torch.sum(loss, dim=0)
                total_loss += loss.data
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if step % 50 == 0:
                    logger.info("epoch-{}, step-{}/{}, loss:{}".format(epoch, step, self.train_loader.steps, loss.data))
                if step == 0:
                    probs = prob
                else:
                    probs = np.concatenate([probs, prob], axis=0)
            # 模型测试
            targets, preds = self.eval_weak(cfg, self.model, self.test_loader)
            metric = accuracy_score(targets, preds)
            if metric > best_metrics:
                best_metrics = metric
                self.model.module.save(self.save_path)
            logger.info('====BEST results:{}, resuls:{}, loss:{}====='.format(best_metrics, metric, total_loss))

    def eval_weak(self, cfg, model, loader):
        model.eval()
        targets, preds = [], []
        for step, ipt in enumerate(loader):
            ipt = {k: v.cuda() for k, v in ipt.items()}
            out, loss = self.model(ipt)
            target = ipt['labels'].cpu().detach().numpy()
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            pred = torch.max(out.logits, dim=-1)[1].cpu().detach().numpy()
            # else:
            #     pred = out.squeeze().cpu().detach().numpy()
            targets.extend(list(target))
            preds.extend(list(pred))
        model.train()
        return targets, preds

    # CD: cross-domain
    def test_CD_weak(self, cfg):
        model = AutoModelForSequenceClassification.from_pretrained(self.save_path, num_labels=cfg.nclass).cuda()
        loader = WeakDataset(cfg, self.tokenizer, self.tokenizer, train='test', target=True)
        model.eval()
        targets, preds = [], []
        for step, ipt in enumerate(loader):
            input_ids = ipt['input_ids'].cuda()
            attention_mask = ipt['attention_mask'].cuda()
            labels = ipt['labels'].cuda()
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True,
                        output_hidden_states=True, return_dict=True).logits
            target = labels.cpu().detach().numpy()
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
            targets.extend(list(target))
            preds.extend(list(pred))
        acc = accuracy_score(targets, preds)
        logger.info('============acc on {}-{} is {}================='.format(cfg.dataset, cfg.target, acc))

    def test_CD_weak_lora(self):
        base_model = prepare_model_for_int8_training(self.model)
        lora_model = PeftModel.from_pretrained(
            base_model,
            self.save_path,
            device_map={"": "cuda:0"},
            torch_dtype=torch.float16,
        )
        loader = WeakDataset(cfg, self.tokenizer, self.tokenizer, train='test', target=True)
        model = lora_model.merge_and_unload()
        model.eval()
        targets, preds = [], []
        for step, ipt in enumerate(loader):
            input_ids = ipt['input_ids'].cuda()
            attention_mask = ipt['attention_mask'].cuda()
            labels = ipt['labels'].cuda()
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True,
                        output_hidden_states=True, return_dict=True).logits
            target = labels.cpu().detach().numpy()
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
            targets.extend(list(target))
            preds.extend(list(pred))
        acc = accuracy_score(targets, preds)
        logger.info('============acc on {}-{} is {}================='.format(cfg.dataset, cfg.target, acc))

    def get_overconfidence_samples(self, cfg):
        ''' 根据snapshots获取预测错误以及置信度过高的样本
        :return:
        '''
        index2con = {}    # 索引到置信度的映射
        cfg.use_checkpoints = True
        model = WeakClassifier(cfg).cuda()
        model.eval()
        logits, targets, confidences = [], [], []
        for step, ipt in enumerate(self.train_loader):
            ipt = {k: v.cuda() for k, v in ipt.items()}
            out = model(ipt)[0].logits
            target = ipt["labels"].cpu().detach().numpy()
            logit = torch.softmax(out, dim=-1).cpu().detach().numpy()
            targets.extend(list(target))
            if step == 0:
                logits = logit
            else:
                logits = np.concatenate([logits, logit], axis=0)
        for i in range(len(targets)):
            target, logit = targets[i], logits[i]
            confidence = logit[target]
            # if confidence < 0.6:
            #     print(confidence, self.train_loader.datas[i])
            confidences.append(confidence)
        for index, con in zip(self.train_loader.index, confidences):
            index2con[index] = float(con)
        # np.save('confidence/{}-{}'.format(self.cfg.dataset, self.cfg.fold), confidences)\
        with open('confidence/{}-{}.json'.format(self.cfg.dataset, self.cfg.fold), 'w') as wf:
            json.dump(index2con, wf)


class TeacherTrainer(WeakTrainer):
    '''
    Trainer for Bert-based Model
    '''
    def __init__(self, cfg):
        super(TeacherTrainer, self).__init__(cfg)
        self.cfg = cfg
        self.model = WeakClassifier(cfg)
        if len(cfg.cudas) > 1:
            self.model = to_multiCUDA(cfg, self.model)
        self.model.cuda()
        self.tokenizer = self.model.module.tokenizer
        dataset = '-'.join(sorted(self.cfg.dataset))
        model_dir_name = '{}_on_{}_fold_{}'.format(self.cfg.base_model, dataset, self.cfg.fold)
        self.save_path = 'save_models/promptclassifier/' + model_dir_name
        self.train_loader = PromptDataset(cfg, self.tokenizer, 'train', False, self.cfg.fold)
        self.test_loader = WeakDataset(cfg, self.tokenizer, self.tokenizer, 'dev', False, self.cfg.fold)

    def train_weak(self, cfg):
        optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, eps=1e-8)
        # train the model
        best_metrics = 0
        for epoch in range(cfg.epoch):
            total_loss = 0.0
            probs = []
            self.model.train()
            for step, ipt in enumerate(self.train_loader):
                ipt = {k: v.cuda() for k, v in ipt.items()}
                out, loss = self.model(ipt)
                prob = out.logits.cpu().detach().numpy()
                if len(cfg.cudas) > 1:  # check-multi GPUs
                    loss = torch.sum(loss, dim=0)
                # else:
                #     loss = MSE(out.squeeze(), ipt['labels'])
                total_loss += loss.data
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if step % 50 == 0:
                    logger.info("epoch-{}, step-{}/{}, loss:{}".format(epoch, step, self.train_loader.steps, loss.data))
                if step == 0:
                    probs = prob
                else:
                    probs = np.concatenate([probs, prob], axis=0)
            # 模型测试
            targets, preds = self.eval_weak(cfg, self.model, self.test_loader)
            metric = accuracy_score(targets, preds)
            if metric > best_metrics:
                best_metrics = metric
                self.model.module.save(self.save_path)
            logger.info('====BEST results:{}, resuls:{}, loss:{}====='.format(best_metrics, metric, total_loss))


class VAETrainer(WeakTrainer):
    '''
    Trainer for Bert-based Model
    '''
    def __init__(self, cfg, use_teacher=True):
        super(VAETrainer, self).__init__(cfg)
        self.cfg = cfg
        if use_teacher:
            self.model = DistilVAEClassifier(cfg)
        else:
            self.model = VAEClassifier(cfg)
        self.tokenizer = self.model.tokenizer
        if len(cfg.cudas) > 1:
            print('convert model to multi-CUDA')
            self.model = to_multiCUDA(cfg, self.model)
        self.model.cuda()
        dataset = '-'.join(sorted(self.cfg.dataset))
        model_dir_name = '{}_on_{}_fold_{}'.format(self.cfg.base_model, dataset, self.cfg.fold)
        if use_teacher:
            self.save_path = 'save_models/DistilVAEclassifier/' + model_dir_name
        else:
            self.save_path = 'save_models/VAEclassifier/' + model_dir_name
        if self.cfg.base_model == 'bert':
            teacher_path = self.cfg.root + 'distilbert_en_base/'
        else:
            teacher_path = self.cfg.root + 'distilroberta_en_base/'
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        self.train_loader = WeakDataset(cfg, self.tokenizer, self.teacher_tokenizer, 'train', False, fold=cfg.fold)
        self.test_loader = WeakDataset(cfg, self.tokenizer, self.teacher_tokenizer, 'dev', False, fold=cfg.fold)

    def train_weak(self, cfg):
        # optimizer = optim.AdamW([{'params': self.model.model.parameters(), 'lr':cfg.lr}],
        #                         lr=1e-5, eps=1e-8)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, eps=1e-8)
        # train the model
        best_metrics = 0.0
        for epoch in range(cfg.epoch):
            total_loss = 0.0
            self.model.train()
            for step, ipt in enumerate(self.train_loader):
                ipt = {k: v.cuda() for k, v in ipt.items()}
                out, loss = self.model(ipt)[:2]
                if len(cfg.cudas) > 1:  # check-multi GPUs
                    loss = torch.sum(loss, dim=0)
                total_loss += loss.data
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if step % 50 == 0:
                    logger.info("epoch-{}, step-{}/{}, loss:{}".format(epoch, step, self.train_loader.steps, loss.data))
            # 模型测试
            targets, preds = self.eval_weak(cfg, self.model, self.test_loader)
            metric = accuracy_score(targets, preds)
            if metric > best_metrics:
                # if metric > 0.95:  # 防止过拟合
                #     break
                best_metrics = metric
                torch.save(self.model.module, self.save_path)
                # for data in ['books', 'dvd', 'electronics', 'kitchen']:
                #     cfg.use_checkpoints = True
                #     cfg.batch = 32
                #     cfg.target = data
                #     self.test_CD_weak(cfg)
            logger.info('====BEST results:{}, resuls:{}, loss:{}====='.format(best_metrics, metric, total_loss))


    def eval_weak(self, cfg, model, loader):
        model.eval()
        targets, preds = [], []
        for step, ipt in enumerate(loader):
            ipt = {k: v.cuda() for k, v in ipt.items()}
            out, loss = self.model(ipt)[:2]
            target = ipt['labels'].cpu().detach().numpy()
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
            # else:
            #     pred = out.squeeze().cpu().detach().numpy()
            targets.extend(list(target))
            preds.extend(list(pred))
        model.train()
        return targets, preds

    # CD: cross-domain
    def test_CD_weak(self, cfg):
        # model = VAEClassifier(cfg).cuda()
        device = torch.device("cpu")
        model = torch.load(self.save_path, map_location=device)
        model = to_multiCUDA(cfg, model).cuda()
        loader = WeakDataset(cfg, self.tokenizer, self.tokenizer, train='test', target=True)
        model.eval()
        targets, preds = [], []
        for step, ipt in enumerate(loader):
            ipt = {k: v.cuda() for k, v in ipt.items()}
            out = model(ipt)[0]
            target = ipt["labels"].cpu().detach().numpy()
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
            targets.extend(list(target))
            preds.extend(list(pred))
        acc = accuracy_score(targets, preds)
        logger.info('============acc on {}-{} is {}================='.format(cfg.dataset, cfg.target, acc))

    def vis(self, cfg):
        device = torch.device("cpu")
        model = torch.load(self.save_path, map_location=device)
        model = to_multiCUDA(cfg, model).cuda()
        loader = WeakDataset(cfg, self.tokenizer, self.tokenizer, train='train', target=False)
        model.eval()
        means, vars = [], []
        for step, ipt in enumerate(loader):
            ipt = {k: v.cuda() for k, v in ipt.items()}
            out = model(ipt)
            mean, var = out[-1][0], out[-1][1]
            # mean, var = model.module.z_mean, model.module.z_log_var
            mean = mean.detach().cpu().numpy()
            var = var.detach().cpu().numpy()
            means.append(mean)
            vars.append(var)
        means = np.concatenate(means, axis=0)
        vars = np.concatenate(vars, axis=0)
        datas = np.concatenate([means, vars], axis=0)
        labels = [0]*means.shape[0] + [1]*means.shape[0]
        tsne = TSNE(n_components=2, random_state=42)
        data_tsne = tsne.fit_transform(datas)
        colors = ['purple', 'green']
        plt.tick_params(axis='x', labelsize=25)
        plt.tick_params(axis='y', labelsize=25)
        plt.scatter(data_tsne[:,0], data_tsne[:,1], s=10,
                    c=[colors[label] for label in labels])
        # plt.legend(labels=['μ', 'σ'])
        plt.show()
import numpy as np
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, GPT2Tokenizer, AutoTokenizer, AutoModel
import torch
from os.path import join, exists
import torch.functional as F
import os
import json

def load_backbone(cfg):
    model_dir_name = '{}_on_{}_fold_{}'.format(cfg.base_model, cfg.dataset, cfg.fold)
    save_path = 'save_models/weakclassifier/' + model_dir_name
    if cfg.use_checkpoints == False:
        print('load from official checkpoints')
        backbone = AutoModelForSequenceClassification.from_pretrained(cfg.bert_path, num_labels=cfg.nclass)
    else:
        print('load from {}'.format(save_path))
        if cfg.base_model == 'opt1.3b':
            backbone = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=cfg.nclass, load_in_8bit=True)
        else:
            backbone = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=cfg.nclass)
    if cfg.base_model in ['gpt2']:
        tokenizer = GPT2Tokenizer.from_pretrained(cfg.bert_path, do_lower_case=True)
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        # fix model padding token id
        backbone.config.pad_token_id = backbone.config.eos_token_id
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)
    return backbone, tokenizer

class Encoder(torch.nn.Module):
    def __init__(self, input_dim=768, latent_dim=64):
        super(Encoder, self).__init__()
        self.initial_dense = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=384),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=384, out_features=128),
            torch.nn.ReLU(inplace=True),
        )
        # 输出的均值和方差
        self.z_mean = torch.nn.Linear(in_features=128, out_features=latent_dim)
        self.z_log_var = torch.nn.Linear(in_features=128, out_features=latent_dim)

    def forward(self, x):
        # x = x.view(-1, 784)
        x = self.initial_dense(x)   # b*n*768
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

class Decoder(torch.nn.Module):
    def __init__(self,latent_dim=64,num_features=768):
        super(Decoder, self).__init__()
        self.initial_dense = torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_dim,out_features=128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=128,out_features=384),
            torch.nn.ReLU(inplace=True)
        )
        self.recover_text = torch.nn.Linear(in_features=384,out_features=num_features)

    def forward(self,x):
        x = self.initial_dense(x)
        text = self.recover_text(x)
        return text

class WeakClassifier(nn.Module):
    def __init__(self, cfg):
        super(WeakClassifier, self).__init__()
        self.cfg = cfg
        self.model, self.tokenizer = load_backbone(self.cfg)
        if 'bert' not in self.cfg.base_model:   # 非bert模型只调最后的分类头
            if 'opt' in self.cfg.base_model:    # 冻结opt的model层
                self.model.model.requires_grad = False

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ipt):
        input_ids = ipt['input_ids']
        attention_mask = ipt['attention_mask']
        labels = ipt['labels']
        # token_type_ids = ipt['token_type_ids']
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         # token_type_ids=token_type_ids,
                         output_attentions=True,
                         output_hidden_states=True, return_dict=True)
        loss = self.loss_fn(out.logits, labels)
        return out, loss

    def get_grad(self, input_ids, segment_ids=None, input_mask=None, label_ids=None,
                 tar_layer=None, one_batch_att=None, pred_label=None):
        '''
        :param input_ids:  input ids
        :param segment_ids:  token_type_ids
        :param input_mask:  attention masks
        :param label_ids:
        :param tar_layer: the layer of the attention head
        :param one_batch_att: the attention score of the target layer
        :param pred_label: the label predicted by baseline language model rather than the ground truth
        :return:
        '''
        _, pooled_output, att_score = self.model(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False,
            tar_layer=tar_layer, tmp_score=one_batch_att)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        prob = torch.softmax(logits, dim=-1)
        tar_prob = prob[:, label_ids[0]]
        if one_batch_att is None:
            return att_score[0], logits
        else:
            #gradient = torch.autograd.grad(torch.unbind(prob[:, labels[0]]), tmp_score)
            gradient = torch.autograd.grad(torch.unbind(prob[:, pred_label]), one_batch_att)
            return tar_prob, gradient[0]

    def js_div(self, p_output, q_output, get_softmax=True):
        ''' used for confidence
        :param p_output:
        :param q_output:
        :param get_softmax:
        :return:
        '''
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = torch.softmax(p_output, dim=-1)
            q_output = torch.softmax(q_output, dim=-1)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_pretrained(save_path)


class VAEClassifier(nn.Module):
    def __init__(self, cfg):
        super(VAEClassifier, self).__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(self.cfg.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.bert_path)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.class_head = nn.Linear(64, 2)
        self.mse_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, ipt):
        input_ids = ipt['input_ids']
        attention_mask = ipt['attention_mask']
        labels = ipt['labels']
        # token_type_ids = ipt['token_type_ids']
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)[1]
        z_mean, z_log_var = self.encoder(out)
        # sampling
        z = self.samples(z_mean, z_log_var)
        output = self.decoder(z)
        vae_loss = self.loss_fn(inputs=out, outputs=output, z_mean=z_mean,
                                 z_log_var=z_log_var, num_features=768)
        logits = torch.softmax(self.class_head(z_mean), dim=-1)
        ce_loss = self.ce_loss(logits, labels)
        return logits, self.cfg.lamda1*vae_loss+(1-self.cfg.lamda1)*ce_loss

    def samples(self, z_mean, z_log_var):
        """
        :param args: 编码器产生的均值和噪声
        :return:
        """
        eps = torch.nn.init.normal_(z_log_var, mean=0., std=1.0)
        z = z_mean + torch.exp(z_log_var / 2) * eps
        return z

    def loss_fn(self, inputs, outputs, z_mean, z_log_var, num_features=768):
        reconstruction_loss = self.mse_loss(outputs, inputs)
        reconstruction_loss = reconstruction_loss * num_features
        # 计算KL散度损失值
        kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
        kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)
        vae_loss = torch.mean(reconstruction_loss + kl_loss)
        return vae_loss

    def save(self, model, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model, save_path)

class DistilVAEClassifier(nn.Module):
    # 指定一个带有shortcut的teacher分类器，并利用方差特征逼近
    def __init__(self, cfg):
        super(DistilVAEClassifier, self).__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(self.cfg.bert_path)
        if 'bert' not in self.cfg.base_model:  # 非bert模型只调最后的分类头
            if 'opt' in self.cfg.base_model:  # 冻结opt的model层
                self.model.decoder.requires_grad = False
        if self.cfg.base_model == 'bert':
            teacher_path = '/home/songrui/code/ShortcutPrompt/save_models/promptclassifier/dbert_on_{}_fold_{}' \
                .format('-'.join(self.cfg.dataset), self.cfg.fold)
        else:
            teacher_path = '/home/songrui/code/ShortcutPrompt/save_models/promptclassifier/distilbert_on_{}_fold_{}' \
                .format('-'.join(self.cfg.dataset), self.cfg.fold)
        self.teacher = AutoModel.from_pretrained(teacher_path)
        # for name, parameter in self.teacher.named_parameters():  # 冻结教师参数
        #     parameter.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.bert_path)
        if 'opt' in self.cfg.base_model:   # 自回归模型向左padding
            self.tokenizer.padding_side = "left"
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.class_head = nn.Linear(64, self.cfg.nclass)
        self.var_head = nn.Linear(768, 64)  # 投影到teacher模型的特征空间
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.smoothL1Loss = torch.nn.SmoothL1Loss(beta=2.0)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, ipt):
        input_ids = ipt['input_ids']
        # 由于student 和 teacher的词表可能不一样，因此需要两套不同的ids
        teacher_ids = ipt['base_ids']
        attention_mask = ipt['attention_mask']
        labels = ipt['labels']
        # token_type_ids = ipt['token_type_ids']
        if 'deberta' in self.cfg.base_model:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            out = out[:, 0, :]
        else:
            if 'bert' in self.cfg.base_model:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)[1] # b*768
            else:   # for opt output
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
                batch_size = out.shape[0]
                out = out[torch.arange(batch_size, device=out.device), -1]  # 取最后一个作为pool output
                # 参考 https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L971
        # teacher_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)[1]
        if self.cfg.base_model == 'bert':
            teacher_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
            teacher_out = teacher_out[:, 0, :]
        else:
            teacher_out = self.teacher(input_ids=teacher_ids)
            teacher_out = teacher_out[1]
        teacher_out = self.var_head(teacher_out)
        self.z_mean, self.z_log_var = self.encoder(out)
        # sampling
        z = self.samples(self.z_mean, self.z_log_var)
        output = self.decoder(z)
        vae_loss = self.loss_fn(inputs=out, outputs=output, z_mean=self.z_mean,
                                 z_log_var=self.z_log_var, num_features=768)
        logits = torch.softmax(self.class_head(self.z_mean), dim=-1)
        ce_loss = self.ce_loss(logits, labels)
        distil_loss = self.distil_fn(teacher_out, self.z_log_var)
        loss = (1-self.cfg.lamda1-self.cfg.lamda2)*ce_loss + self.cfg.lamda1*vae_loss + self.cfg.lamda2*distil_loss
        return logits, loss, [self.z_mean, self.z_log_var]

    def samples(self, z_mean, z_log_var):
        """
        :param args: 编码器产生的均值和噪声
        :return:
        """
        eps = torch.nn.init.normal_(z_log_var, mean=0., std=1.0)
        z = z_mean + torch.exp(z_log_var / 2) * eps
        return z

    def loss_fn(self, inputs, outputs, z_mean, z_log_var, num_features=768):
        reconstruction_loss = self.mse_loss(outputs, inputs)
        reconstruction_loss = reconstruction_loss * num_features
        # 计算KL散度损失值
        kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
        kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)
        vae_loss = torch.mean(reconstruction_loss + kl_loss)
        return vae_loss

    def distil_fn(self, z1, z2):
        def _white(feature_map):
            # 1. 计算特征通道的均值和方差
            mean = torch.mean(feature_map, dim=1, keepdim=True)  # 计算均值
            var = torch.var(feature_map, dim=1, unbiased=False, keepdim=True)  # 计算方差
            # 2. 对特征张量进行中心化和缩放，得到标准化特征张量
            epsilon = 1e-8  # 用于稳定计算的小常数
            normalized_feature_map = (feature_map - mean) / torch.sqrt(var + epsilon)
            return normalized_feature_map
        z1, z2 = _white(z1), _white(z2)
        # vector1_normalized = torch.nn.functional.normalize(z1, dim=1)
        # vector2_normalized = torch.nn.functional.normalize(z2, dim=1)
        # SmoothL1Loss from "Contrastive Learning Rivals Masked Image Modeling
        # in Fine-tuning via Feature Distillation"
        # similarity = torch.nn.functional.cosine_similarity(vector1_normalized, vector2_normalized, dim=1)
        # return 1-torch.mean(similarity)
        loss = self.smoothL1Loss(z1, z2)
        return loss.mean()

    def save(self, model, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model, save_path)


class PoE(nn.Module):
    def __init__(self, cfg):
        super(PoE, self).__init__()
        self.cfg = cfg
        self.model, self.tokenizer = load_backbone(self.cfg)
        model_dir_name = '{}_on_{}'.format(self.cfg.base_model, self.cfg.dataset)
        self.teacher_path = 'save_models/promptclassifier/' + model_dir_name
        self.teacher = AutoModelForSequenceClassification.from_pretrained(self.teacher_path, num_labels=2)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ipt):
        input_ids = ipt['input_ids']
        attention_mask = ipt['attention_mask']
        labels = ipt['labels']
        s_out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True,
                         output_hidden_states=True, return_dict=True)
        t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True,
                           output_hidden_states=True, return_dict=True)
        loss = self.loss_fn(s_out.logits + t_out.logits, labels)
        return s_out, loss

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_pretrained(save_path)


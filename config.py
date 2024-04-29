import torch

class BaseConfig:
    def __init__(self):
        super(BaseConfig, self).__init__()
        self.get_attributes()

    def get_attributes(self):
        self.base_model = 'roberta'  # Bert, DistilBert, Alberta, RoBERTa
        self.root = '/home/songrui/data/'
        self.use_checkpoints = False  # 是否使用已经训练好的节点
        if self.base_model == 'bert':
            self.bert_path = self.root + 'bert-base-uncased/'
        if self.base_model == 'distilbert':
            self.bert_path = self.root + 'distilroberta_en_base/'
        if self.base_model == 'dbert':
            self.bert_path = self.root + 'distilbert_en_base/'
        if self.base_model == 'roberta':
            self.bert_path = self.root + 'roberta_en_base/'
        if self.base_model == 'deberta':
            self.bert_path = self.root + 'deberta-base/'
        if self.base_model == 'deberta_v3':
            self.bert_path = self.root + 'deberta-v3-base/'
        if self.base_model == 'opt350m':
            self.bert_path = self.root + 'opt-350m/'
        if self.base_model == 'opt1.3b':
            self.bert_path = self.root + 'opt-1.3b/'
        # self.dataset = ['dvd', 'electronics', 'kitchen']
        self.dataset = ['books']
        # self.dataset = ['books', 'dvd', 'kitchen']
        # self.dataset = ['books', 'dvd', 'electronics']
        self.target = 'dvd'
        self.max_len = 300
        self.nclass = 2
        self.lamda1 = 0.001   # 0.001 for roberta bert deberta, 1e-5 for opt
        self.lamda2 = 0.001
        self.cudas = [0,1,2]
        self.fold = 1
        assert self.fold in [1,2,3,4,5]


class WeakConfig(BaseConfig):
    '''
    weak model的配置
    '''
    def __init__(self):
        super(WeakConfig, self).__init__()
        self.lr = 1e-5
        self.epoch = 10
        self.batch = 64
        self.get_attributes()

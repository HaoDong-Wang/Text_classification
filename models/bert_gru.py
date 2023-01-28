import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert_gru'
        self.log_path = './logs/'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                 # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 24        # 128                                           # mini-batch大小
        self.pad_size = 48                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.hidden_size = 768
        self.bidirectional = True



class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.bidirectional = config.bidirectional
        self.gru = nn.GRU(config.hidden_size, config.hidden_size*2, 2, bidirectional = config.bidirectional, batch_first=True)
        self.fc = nn.Linear(config.hidden_size*2*2, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encode_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, hn = self.gru(encode_out)
        # print(out.size())
        # print(hn.size())
        h = hn[-(1 + int(self.bidirectional)):] # 用最后两个个hidden layer的结果
        x = torch.cat(h.split(1), dim=-1).squeeze(0) # 在上一步操作中，0维中只有一个元素，用squeeze把0维缩掉，变成两维( batch_size, hidden_out) 
        # print(x.size())
        # exit()
        out = self.fc(x)
        # print(out.size())
        # exit()
        return out


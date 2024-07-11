from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert import BertModel, BertTokenizer


def writetotxt(path,out):
    out = out.detach().cpu().numpy()
    with open(path, "w", encoding='utf-8') as file:
        print(out.shape)
        for j in out:
            tem_list = []
            for i in j:
                tem_list.append(i)
            # print(len(tem_list))
            tem_str = str(tem_list)
            file.write(tem_str[1:-1])
            file.write('\n')
    print('over')

class Config(object):

    def __init__(self, dataset):

        self.model_name = "SpliceSiteBert1DCNN"

        self.train_path = dataset + '/data/train.csv'

        self.dev_path = dataset + '/data/dev.csv'

        self.test_path = dataset + '/data/test.csv'

        self.datasetpkl = dataset + '/data/dataset.pkl'


        self.saved_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'

        self.log_path = dataset + '/logs/' + self.model_name

        self.result_path = dataset + '/results/' + self.model_name + '.csv'
        self.result_path_report = dataset + '/results/' + self.model_name + '_report.csv'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 2000  # 10

        self.num_classes = 1 # len(self.class_list)

        self.num_epochs = 200

        self.batch_size = 128

        self.pad_size = 40  # len(self.class_list) + 1

        self.learning_rate = 1e-4

        self.bert_path = 'bert_pretrain/new-4'

        self.tokenizer = BertTokenizer.from_pretrained((self.bert_path))

        self.hidden_size = 768

        self.filter_sizes = (
        3, 4, 5, 9, 10, 11)

        self.num_filters = 256

        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=config.hidden_size,
                                     out_channels=config.num_filters,
                                     kernel_size=k),
                           nn.ReLU(),
                           nn.MaxPool1d(kernel_size=config.pad_size - k + 1)
                           )
             for k in config.filter_sizes
             ])

        self.droptout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def forward(self, x,test,config):

        context = x[0]
        mask = x[2]
        encoder_out, pooled = self.bert(context, attention_mask=mask,
                                        output_all_encoded_layers=False)
        out = encoder_out.permute(0, 2, 1)
        out = torch.cat([conv(out) for conv in self.convs], 1)
        out = out.squeeze(2)

        if test:
            writetotxt(path=config.embsavepath,out=out)
        out = self.droptout(out)

        # out_np = out.detach().cpu().numpy()

        # np.savetxt('results/out.txt', out_np, fmt='%.6f', delimiter='\t')
        out = self.fc(out)
        out = out.squeeze()
        out = out.sigmoid()
        return out

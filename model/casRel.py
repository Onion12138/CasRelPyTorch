import torch.nn as nn
import torch
from transformers import BertModel
import time
from fastNLP import Callback
import os
from model.evaluate import metric


class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_name)
        self.sub_heads_linear = nn.Linear(self.config.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.config.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)
        self.obj_tails_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        # sub_head_mapping [batch, 1, seq] * encoded_text [batch, seq, dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        sub = (sub_head + sub_tail) / 2
        encoded_text = encoded_text + sub
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails

    def forward(self, **data):
        # def forward(self, token_ids, mask, sub_head, sub_tail):
        # print(data)
        token_ids = data['token_ids']
        mask = data['mask']
        encoded_text = self.get_encoded_text(token_ids, mask)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        sub_head_mapping = data['sub_head'].unsqueeze(1)
        # sub_head_mapping = sub_head.unsqueeze(1)
        sub_tail_mapping = data['sub_tail'].unsqueeze(1)
        # sub_tail_mapping = sub_tail.unsqueeze(1)
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text)

        return {
            "sub_heads": pred_sub_heads,
            "sub_tails": pred_sub_tails,
            "obj_heads": pred_obj_heads,
            "obj_tails": pre_obj_tails,
        }





class MyCallBack(Callback):
    def __init__(self, data_iter, rel_vocab, config):
        super().__init__()
        self.loss_sum = 0
        self.global_step = 0

        self.data_iter = data_iter
        self.rel_vocab = rel_vocab
        self.config = config

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.save_logs_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')


    def on_train_begin(self):
        self.best_f1_score = 0
        self.best_precision = 0
        self.best_recall = 0

        self.best_epoch = 0
        self.init_time = time.time()
        self.start_time = time.time()
        print("-" * 5 + "Initializing the model" + "-" * 5)

    def on_epoch_begin(self):
        self.eval_start_time = time.time()

    def on_epoch_end(self):
        precision, recall, f1_score = metric(self.data_iter, self.rel_vocab, self.config, self.model)
        self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.
                     format(self.epoch, time.time() - self.eval_start_time, f1_score, precision, recall))
        if f1_score > self.best_f1_score:
            self.best_f1_score = f1_score
            self.best_epoch = self.epoch
            self.best_precision = precision
            self.best_recall = recall
            self.logging("Saving the model, epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".
                         format(self.best_epoch, self.best_f1_score, precision, recall))
            path = os.path.join(self.config.save_weights_dir, self.config.weights_save_name)
            torch.save(self.model.state_dict(), path)

    def on_train_end(self):
        self.logging("-" * 5 + "Finish training" + "-" * 5)
        self.logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}, total time: {:5.2f}s".
                     format(self.best_epoch, self.best_f1_score, self.best_precision, self.best_recall,
                            time.time() - self.init_time))



from fastNLP import Callback
import os
from model.evaluate import metric
import torch


class MyCallBack(Callback):
    def __init__(self, data_iter, rel_vocab, config):
        super().__init__()
        self.best_epoch = 0
        self.best_recall = 0
        self.best_precision = 0
        self.best_f1_score = 0

        self.data_iter = data_iter
        self.rel_vocab = rel_vocab
        self.config = config

    def logging(self, s, print_=True, log_=False):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.save_logs_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def on_train_begin(self):
        self.logging("-" * 5 + "Begin Training" + "-" * 5)

    def on_epoch_end(self):
        precision, recall, f1_score = metric(self.data_iter, self.rel_vocab, self.config, self.model)
        self.logging('epoch {:3d}, f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'
                     .format(self.epoch, f1_score, precision, recall))

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
        self.logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}".
                     format(self.best_epoch, self.best_f1_score, self.best_precision, self.best_recall))


import json


class Config(object):
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.bert_name = args.bert_name
        self.bert_dim = args.bert_dim

        self.train_path = 'data/' + self.dataset + '/train.json'
        self.test_path = 'data/' + self.dataset + '/test.json'
        self.dev_path = 'data/' + self.dataset + '/dev.json'
        self.rel_path = 'data/' + self.dataset + '/rel.json'
        self.num_relations = len(json.load(open(self.rel_path, 'r')))

        self.save_weights_dir = 'saved_weights/' + self.dataset + '/'
        self.save_logs_dir = 'saved_logs/' + self.dataset + '/'
        self.result_dir = 'results/' + self.dataset + '/'

        self.period = 200
        self.test_epoch = 3
        self.weights_save_name = 'model.pt'
        self.log_save_name = 'model.out'
        self.result_save_name = 'result.json'

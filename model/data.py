import json
from random import choice
from fastNLP import TorchLoaderIter, DataSet
from fastNLP.io import JsonLoader
from fastNLP import Vocabulary, RandomSampler
import numpy as np
import torch
from transformers import BertTokenizer
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_rel = 18


def load_data(train_path, dev_path, test_path, rel_dict_path):
    paths = {'train': train_path, 'dev': dev_path, 'test': test_path}
    loader = JsonLoader({"text": "text", "spo_list": "spo_list"})
    data_bundle = loader.load(paths)
    id2rel = json.load(open(rel_dict_path))
    rel_vocab = Vocabulary(unknown=None, padding=None)
    rel_vocab.add_word_lst(list(id2rel.values()))
    return data_bundle, rel_vocab


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class MyDataset(DataSet):
    def __init__(self, config, dataset, rel_vocab, is_test):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.rel_vocab = rel_vocab
        self.is_test = is_test
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __getitem__(self, item):
        json_data = self.dataset[int(item)]
        text = json_data['text']
        tokenized = self.tokenizer(text, max_length=self.config.max_len, truncation=True)
        tokens = tokenized['input_ids']
        masks = tokenized['attention_mask']
        text_len = len(tokens)
        if not self.is_test:
            s2ro_map = defaultdict(list)
            for spo in json_data['spo_list']:
                triple = (self.tokenizer(spo['subject'], add_special_tokens=False)['input_ids'],
                          self.rel_vocab.to_index(spo['predicate']),
                          self.tokenizer(spo['object'], add_special_tokens=False)['input_ids'])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    s2ro_map[sub].append(
                        (obj_head_idx, obj_head_idx + len(triple[2]) - 1, triple[1]))

            if s2ro_map:
                token_ids = np.array(tokens)
                masks = np.array(masks)
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                obj_heads, obj_tails = np.zeros((text_len, self.config.num_relations)), \
                                       np.zeros((text_len, self.config.num_relations))
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, \
                       json_data['spo_list']
            else:
                return None
        else:
            token_ids = np.array(tokens)
            masks = np.array(masks)
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
            obj_heads, obj_tails = np.zeros((text_len, self.config.num_relations)), \
                                   np.zeros((text_len, self.config.num_relations))
            return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, \
                   json_data['spo_list']

    def __len__(self):
        return len(self.dataset)

def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)
    token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples = zip(*batch)
    # cur_batch = len(batch)
    # max_text_len = max(text_len)

    batch_token_ids = pad_sequence(token_ids, batch_first=True)
    batch_masks = pad_sequence(masks, batch_first=True)
    batch_sub_heads = pad_sequence(sub_heads, batch_first=True)
    batch_sub_tails = pad_sequence(sub_tails, batch_first=True)
    batch_sub_head = pad_sequence(sub_head, batch_first=True)
    batch_sub_tail = pad_sequence(sub_tail, batch_first=True)
    batch_obj_heads = pad_sequence(obj_heads, batch_first=True)
    batch_obj_tails = pad_sequence(obj_tails, batch_first=True)
    # batch_token_ids = torch.zeros((cur_batch, max_text_len), dtype=torch.long)
    # batch_masks = torch.zeros((cur_batch, max_text_len), dtype=torch.long)
    # batch_sub_heads = torch.zeros((cur_batch, max_text_len), dtype=torch.float)
    # batch_sub_tails = torch.zeros((cur_batch, max_text_len), dtype=torch.float)
    # batch_sub_head = torch.zeros((cur_batch, max_text_len), dtype=torch.float)
    # batch_sub_tail = torch.zeros((cur_batch, max_text_len), dtype=torch.float)
    # batch_obj_heads = torch.zeros((cur_batch, max_text_len, num_rel), dtype=torch.float)
    # batch_obj_tails = torch.zeros((cur_batch, max_text_len, num_rel), dtype=torch.float)
    #
    # for i in range(cur_batch):
    #     batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
    #     batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
    #     batch_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(sub_heads[i]))
    #     batch_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(sub_tails[i]))
    #     batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
    #     batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))
    #     batch_obj_heads[i, :text_len[i], :].copy_(torch.from_numpy(obj_heads[i]))
    #     batch_obj_tails[i, :text_len[i], :].copy_(torch.from_numpy(obj_tails[i]))

    return {
               "token_ids": batch_token_ids.to(device),
               "mask": batch_masks.to(device),
               "sub_head": batch_sub_head.to(device),
               "sub_tail": batch_sub_tail.to(device),
           }, {"sub_heads": batch_sub_heads.to(device),
               "sub_tails": batch_sub_tails.to(device),
               "obj_heads": batch_obj_heads.to(device),
               "obj_tails": batch_obj_tails.to(device),
               "triples": triples
               }


def get_data_iterator(config, dataset, rel_vocab, is_test=False, collate_fn=my_collate_fn):
    dataset = MyDataset(config, dataset, rel_vocab, is_test)
    return TorchLoaderIter(dataset=dataset,
                           collate_fn=collate_fn,
                           batch_size=config.batch_size if not is_test else 1,
                           sampler=RandomSampler())

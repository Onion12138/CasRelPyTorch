import torch
import os
import json
from transformers import BertTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tuple(triple_list):
    ret = []
    for triple in triple_list:
        ret.append((triple['subject'], triple['predicate'], triple['object']))
    return ret


def metric(data_iter, rel_vocab, config, model, output=True, h_bar=0.5, t_bar=0.5):

    orders = ['subject', 'relation', 'object']
    correct_num, predict_num, gold_num = 0, 0, 0
    tokenizer = BertTokenizer.from_pretrained(config.bert_name)

    for batch_x, batch_y in tqdm(data_iter):
        with torch.no_grad():
            token_ids = batch_x['token_ids']
            mask = batch_x['mask']
            encoded_text = model.get_encoded_text(token_ids, mask)
            pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
            sub_heads = torch.where(pred_sub_heads[0] > h_bar)[0]
            sub_tails = torch.where(pred_sub_tails[0] > t_bar)[0]
            subjects = []
            for sub_head in sub_heads:
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    sub_tail = sub_tail[0]
                    subject = ''.join(tokenizer.decode(token_ids[0][sub_head: sub_tail + 1]).split())
                    subjects.append((subject, sub_head, sub_tail))
            if subjects:
                triple_list = []
                repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                sub_head_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                               device=device)
                sub_tail_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                               device=device)
                for subject_idx, subject in enumerate(subjects):
                    sub_head_mapping[subject_idx][0][subject[1]] = 1
                    sub_tail_mapping[subject_idx][0][subject[2]] = 1
                pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                                 repeated_encoded_text)
                for subject_idx, subject in enumerate(subjects):
                    sub = subject[0]
                    obj_heads = torch.where(pred_obj_heads[subject_idx] > h_bar)
                    obj_tails = torch.where(pred_obj_tails[subject_idx] > t_bar)
                    for obj_head, rel_head in zip(*obj_heads):
                        for obj_tail, rel_tail in zip(*obj_tails):
                            if obj_head <= obj_tail and rel_head == rel_tail:
                                rel = rel_vocab.to_word(int(rel_head))
                                obj = ''.join(tokenizer.decode(token_ids[0][obj_head: obj_tail + 1]).split())
                                triple_list.append((sub, rel, obj))
                                break

                triple_set = set()
                for s, r, o in triple_list:
                    triple_set.add((s, r, o))
                pred_list = list(triple_set)

            else:
                pred_list = []

            pred_triples = set(pred_list)
            gold_triples = set(to_tuple(batch_y['triples'][0]))
            correct_num += len(pred_triples & gold_triples)
            predict_num += len(pred_triples)
            gold_num += len(gold_triples)

            if output:
                if not os.path.exists(config.result_dir):
                    os.mkdir(config.result_dir)
                path = os.path.join(config.result_dir, config.result_save_name)
                fw = open(path, 'w')
                result = json.dumps({
                    'triple_list_gold': [
                        dict(zip(orders, triple)) for triple in gold_triples
                    ],
                    'triple_list_pred': [
                        dict(zip(orders, triple)) for triple in pred_triples
                    ],
                    'new': [
                        dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                    ],
                    'lack': [
                        dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                    ]
                }, ensure_ascii=False)
                fw.write(result + '\n')

    print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    print('f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.format(f1_score, precision, recall))
    return precision, recall, f1_score

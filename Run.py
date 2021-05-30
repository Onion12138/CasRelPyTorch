import argparse
import torch
import torch.optim as optim
from model.casRel import CasRel
from model.data import load_data, get_data_iterator
from model.config import Config
from model.evaluate import metric
import time
import os
import torch.nn.functional as F
from fastNLP import Trainer

seed = 226
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--dataset', default='baidu', type=str, help='define your own dataset names')
parser.add_argument("--bert_name", default='bert-base-chinese', type=str, help='choose pretrained bert name')
parser.add_argument('--bert_dim', default=768, type=int)
args = parser.parse_args()
con = Config(args)


def logging(s, print_=True, log_=False):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(con.save_logs_dir, con.log_save_name), 'a+') as f_log:
            f_log.write(s + '\n')


def loss_fn(pred, gold, mask):
    pred = pred.squeeze(-1)
    loss = F.binary_cross_entropy(pred, gold, reduction='none')
    if loss.shape != mask.shape:
        mask = mask.unsqueeze(-1)
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


from model.casRel import MyCallBack
from fastNLP import LossBase


class MyLoss(LossBase):
    def __init__(self):
        super(MyLoss, self).__init__()

    def get_loss(self, predict, target):
        mask = target['mask']

        def loss_fn(pred, gold, mask):
            pred = pred.squeeze(-1)
            loss = F.binary_cross_entropy(pred, gold, reduction='none')
            if loss.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            loss = torch.sum(loss * mask) / torch.sum(mask)
            return loss

        return loss_fn(predict['sub_heads'], target['sub_heads'], mask) + \
               loss_fn(predict['sub_tails'], target['sub_tails'], mask) + \
               loss_fn(predict['obj_heads'], target['obj_heads'], mask) + \
               loss_fn(predict['obj_tails'], target['obj_tails'], mask)

    def __call__(self, pred_dict, target_dict, check=False):
        loss = self.get_loss(pred_dict, target_dict)
        if not (isinstance(loss, torch.Tensor) and len(loss.size()) == 0):
            if not isinstance(loss, torch.Tensor):
                raise TypeError(f"Loss excepted to be a torch.Tensor, got {type(loss)}")
            loss = torch.sum(loss) / (loss.view(-1)).size(0)
        return loss


if __name__ == '__main__':
    model = CasRel(con).to(device)
    data_bundle, rel_vocab = load_data(con.train_path, con.dev_path, con.test_path, con.rel_path)
    train_data = get_data_iterator(con, data_bundle.get_dataset('train'), rel_vocab)
    test_data = get_data_iterator(con, data_bundle.get_dataset('test'), rel_vocab, is_test=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=con.lr)
    trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer, batch_size=con.batch_size,
                      n_epochs=con.max_epoch, loss=MyLoss(), print_every=con.period, use_tqdm=True,
                      callbacks=MyCallBack(train_data, rel_vocab, con))
    trainer.train()
# def run():
#     print("-" * 5 + "Initializing the model" + "-" * 5)
#     model = CasRel(con).to(device)
#     data_bundle, rel_vocab = load_data(con.train_path, con.dev_path, con.test_path, con.rel_path)
#     train_data = get_data_iterator(con, data_bundle.get_dataset('train'), rel_vocab)
#     test_data = get_data_iterator(con, data_bundle.get_dataset('test'), rel_vocab, is_test=True)
#     model.train()
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=con.lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
#     global_step = 0
#     loss_sum = 0
#
#     best_f1_score = 0.0
#     best_precision = 0.0
#     best_recall = 0.0
#
#     best_epoch = 0
#     init_time = time.time()
#     start_time = time.time()
#
#     print("-" * 5 + "Start training" + "-" * 5)
#     for epoch in range(con.max_epoch):
#         for batch_x, batch_y in tqdm(train_data):
#             pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model(batch_x)
#             sub_heads_loss = loss_fn(pred_sub_heads, batch_y['sub_heads'], batch_x['mask'])
#             sub_tails_loss = loss_fn(pred_sub_tails, batch_y['sub_tails'], batch_x['mask'])
#             obj_heads_loss = loss_fn(pred_obj_heads, batch_y['obj_heads'], batch_x['mask'])
#             obj_tails_loss = loss_fn(pred_obj_tails, batch_y['obj_tails'], batch_x['mask'])
#             total_loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)
#
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#
#             global_step += 1
#             loss_sum += total_loss.item()
#
#             if global_step % con.period == 0:
#                 cur_loss = loss_sum / con.period
#                 elapsed = time.time() - start_time
#                 logging("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
#                         format(epoch, global_step, elapsed * 1000 / con.period, cur_loss))
#                 loss_sum = 0
#                 start_time = time.time()
#
#         scheduler.step()
#
#         if epoch % con.test_epoch == 0:
#             eval_start_time = time.time()
#             model.eval()
#             precision, recall, f1_score = metric(test_data, rel_vocab, con, model)
#             model.train()
#             logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.
#                     format(epoch, time.time() - eval_start_time, f1_score, precision, recall))
#
#             if f1_score > best_f1_score:
#                 best_f1_score = f1_score
#                 best_epoch = epoch
#                 best_precision = precision
#                 best_recall = recall
#                 logging("Saving the model, epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".
#                         format(best_epoch, best_f1_score, precision, recall))
#                 path = os.path.join(con.save_weights_dir, con.weights_save_name)
#                 torch.save(model.state_dict(), path)
#
#     logging("-" * 5 + "Finish training" + "-" * 5)
#     logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}, total time: {:5.2f}s".
#             format(best_epoch, best_f1_score, best_precision, best_recall, time.time() - init_time))
#
#
# if __name__ == '__main__':
#     run()

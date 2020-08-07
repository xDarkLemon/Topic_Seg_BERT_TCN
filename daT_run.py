import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from TCN.poly.model import TCN
from TCN.poly.utils import data_generator
import numpy as np
import pandas as pd
import random
import os
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score
from FocalLoss import FocalLoss
from torch.utils.data import Dataset, DataLoader
from bert_serving.client import BertClient
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_false', 
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=150,
                    help='upper epoch limit (default: 150)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='daT',
                    help='the dataset to run (default: daT)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
input_size = 768 # num of features at each step
output_size = 2
#X_train, X_valid, X_test = data_generator(args.data)
#train_data,test_data,dev_data,train_labels,test_labels,dev_labels = data_generator(args.data)

n_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout


model_name = "model_{0}.pt".format(args.data)
model = TCN(input_size, output_size, n_channels, kernel_size, dropout=args.dropout)
if os.path.exists(model_name):
    print('model found!')
    model = torch.load(model_name)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
test_score_lst = []

bc = BertClient()

class myDataset(torch.utils.data.Dataset):
    def __init__(self,dataset):
        super(myDataset,self).__init__()
        self.data = []
        self.labels = []
        self.embeds = []
        with open(dataset,'r',encoding='utf-8') as rfp:
            print('Encoding for sentences for {} ...'.format(dataset))
            for line in tqdm(rfp.readlines()):
                label,text = line.strip('\n').split('+++$+++')
                text = text.split('\t')
                label = [int(t) for t in label.split(',')]
                assert len(text)==len(label)
                self.data.append(text)
                self.labels.append(label)
                self.embeds.append(bc.encode(text))

    def __getitem__(self,index):
        text = self.data[index]
        #embed = bc.encode(text)
        embed = self.embeds[index]
        label = torch.tensor(self.labels[index])
        return (text,embed,label)

    def __len__(self):
        return len(self.labels)

import pickle
if os.path.exists('da_train.pkl'):
    train_dataset = pickle.load(open('da_train.pkl','rb'))
else:
    train_dataset = myDataset('da_train.txt')
    pickle.dump(train_dataset,open('da_train.pkl','wb'))

if os.path.exists('da_test.pkl'):
    test_dataset = pickle.load(open('da_test.pkl','rb'))
else:
    test_dataset = myDataset('da_test.txt')
    pickle.dump(test_dataset,open('da_test.pkl','wb'))

if os.path.exists('da_dev.pkl'):
    val_dataset = pickle.load(open('da_dev.pkl','rb'))
else:
    val_dataset = myDataset('da_dev.txt')
    pickle.dump(val_dataset,open('da_dev.pkl','wb'))


def evaluate(isDev=True,epoch_idx=1):
    logger = open('{}_record.log'.format(args.data),'a',encoding='utf-8')
    model.eval()
    total_loss = 0.0
    count = 0
    mydataset = val_dataset if isDev else test_dataset
    dataloader = DataLoader(mydataset,batch_size=3,shuffle=True)
    FL = FocalLoss(class_num=2,alpha=torch.tensor([[0.25],[0.75]]),gamma=2)
    real_label_lst = []
    pred_label_lst = []
    for iter_step,(text_batch,data_batch,label_batch) in enumerate(dataloader):
        if args.cuda:
            data_batch, label_batch = data_batch.cuda(), label_batch.cuda()
        output = model(data_batch)
        output = output.float()

        loss = 0
        for out,lb in zip(output,label_batch):
            loss += FL(out,lb)
        
        real_label_batch = label_batch.cpu()
        pred_label_batch = torch.argmax(output,dim=2).cpu()
        real_label_lst.append(real_label_batch)
        pred_label_lst.append(pred_label_batch)
        if isDev:
            logger.write('='*20+'\n')
            for txt,rl,pl in zip(list(zip(*text_batch)),real_label_batch,pred_label_batch):
                print('='*20)
                logger.write('='*20+'\n')
                for sent,r,p in zip(txt,rl,pl):
                    print('{} -> {} -> {}'.format(sent,r,p))
                    logger.write('{} \t {} \t {}\n'.format(sent,r,p))
                print('='*20)
                logger.write('='*20+'\n')
            logger.write(classification_report(y_true=real_label_batch,y_pred=pred_label_batch)+'\n')
        total_loss += loss.item()
        count += output.size(0)
    eval_loss = total_loss / count
    print("Validation/Test loss: {:.5f}".format(eval_loss))
    all_real_label_lst = torch.cat(real_label_lst).view(1,-1)[0]
    print('all_real_label_lst.shape:',all_real_label_lst.shape)
    all_pred_label_lst = torch.cat(pred_label_lst).view(1,-1)[0]
    print('all_pred_label_lst.shape:',all_pred_label_lst.shape)
    print('*'*20+'\nEpoch {} - Total classification report:'.format(epoch_idx))
    if isDev:
        logger.write('*'*20+'\nEpoch {} - Total classification report:\n'.format(epoch_idx))
    print(classification_report(y_true=all_real_label_lst,y_pred=all_pred_label_lst))
    if isDev:
        logger.write(classification_report(y_true=all_real_label_lst,y_pred=all_pred_label_lst))
    if not isDev:
        test_score_lst.append({'epoch':epoch_idx,'precision':precision_score(all_real_label_lst,all_pred_label_lst,pos_label=1),\
            'recall':recall_score(all_real_label_lst,all_pred_label_lst,pos_label=1),\
            'f1':f1_score(all_real_label_lst,all_pred_label_lst,pos_label=1)})
    logger.close()
    return eval_loss


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    FL = FocalLoss(class_num=2,alpha=torch.tensor([[0.25],[0.75]]),gamma=2)
    for iter_step,(text_batch,data_batch,label_batch) in enumerate(dataloader):
        #print('data.shape:',data_batch.shape)
        #print('label.shape:',label_batch.shape)
        #print('text_batch.shape:',text_batch)
        
        if args.cuda:
            data = data_batch.cuda()
            label = label_batch.cuda()
        else:
            data = data_batch
            label = label_batch


        optimizer.zero_grad()
        output = model(data)

        #print('output.shape:',output.shape)

        output = output.float()
        loss = 0
        for out,lb in zip(output,label):
            loss += FL(out,lb)
        #loss = FL(output,label)
        #print('loss:',loss)

        total_loss += loss.item()
        count += output.size(0)
        #print('cur_loss: ',loss.item())

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        #if idx > 0 and idx % args.log_interval == 0:
        if iter_step > 0 and (iter_step % args.log_interval == 0):
            cur_loss = total_loss / count
            print("Epoch {:2d} | iteration {:8d} | lr {:.5f} | loss {:.5f}".format(ep, iter_step, lr, cur_loss))
            total_loss = 0.0
            count = 0
        


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    for ep in range(1, args.epochs+1):
        train(ep)
        vloss = evaluate(isDev=False,epoch_idx=ep)
        tloss = evaluate(isDev=True,epoch_idx=ep)
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Epoch {}: Saved model!\n".format(ep))
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-3:]):
         
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        vloss_list.append(vloss)

    print('-' * 89)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(test_data,test_labels,isDev=False,epoch_idx=args.epochs)
    test_score_df = pd.DataFrame(test_score_lst)
    print('test_score_df:')
    print(test_score_df)
    test_score_df.to_csv(open('{}_test_score.tsv'.format(args.data),'a'),sep='\t',index=None,header=None)


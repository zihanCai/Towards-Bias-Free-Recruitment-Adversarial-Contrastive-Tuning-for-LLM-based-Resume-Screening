import torch
# from transformers import (
#     Trainer, 
#     TrainingArguments, 
#     AutoModelForSequenceClassification, 
#     AutoTokenizer, 
#     EvalPrediction, 
#     DataCollatorWithPadding, 
#     TrainerCallback
# )

import torch.nn.functional as F
import torch.nn as nn
from bert_with_adversary import BertWithAdversary
from tqdm import tqdm
from transformers import BertTokenizer,DataCollatorForLanguageModeling
from torch.optim.lr_scheduler import StepLR
from CustomDataCollator import CustomDataCollatorForLanguageModeling
import dataset_utils
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
# from trainer import AdversarialTrainer
from torch.utils.data import DataLoader
import time
import os
import csv
from transformers import set_seed
from sklearn.metrics import accuracy_score, classification_report,recall_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
set_seed(42)



def train(model,config,data_loader,):
    ttt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    epochs = config['epochs']
    best_models = []
    top_k = 3
    for epoch in range(epochs):
        total_loss = 0.0
        total_cls_loss =0.0
        total_bias_loss=0.0
        loop = tqdm((data_loader), total = len(data_loader),leave =False)
        step = 0
        best_loss = 10000000
        best_bias_loss = 0
        csv_file = config['out_dir']+'model-lr-{}-lam-{}-epoch-{}-log.csv'.format(config['learning_rate'],config['lambda'],epoch)
        # train_bias =False
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Total Loss', 'MLM Loss', 'Bias Loss','Adv Acc','Cls Acc'])
        for batch in loop:
            input_ids = batch['input_ids']
            input_ids = input_ids.to(device)
            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.to(device)
            gender = batch['gender']
            gender = gender.to(device)
            labels = batch['labels']
            labels = labels.to(device)
            profession = batch['profession']
            profession = profession.to(device)
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask, 
                        labels=labels, gender=gender,lamb=config['lambda'],profession = profession)
            loss = out['loss']
            cls_loss = out['cls_loss']
            cls_logits = out['cls_logits']
            bias_loss = out['bias_loss']
            bias_logits = out["bias_logits"]
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_bias_loss += bias_loss.item()



            gender_list = gender.cpu().tolist()
            adv_preds = torch.argmax(bias_logits, dim=1)
            adv_pred_list = adv_preds.cpu().tolist()
            adv_acc = accuracy_score(gender_list,adv_pred_list)

            preds = torch.argmax(cls_logits,dim=1)
            preds_list = preds.cpu().tolist()
            profession_list = profession.cpu().tolist()
            cls_acc = accuracy_score(profession_list,preds_list)
            loss.backward()
            optimizer.step()
            
            model.zero_grad()
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            
            if step % 5 == 0:
                loop.set_postfix(loss = total_loss/(step+1),
                                 cls_loss = total_cls_loss/(step+1),
                                 bias_loss=total_bias_loss/(step+1),
                                 adv_acc=adv_acc,cls_acc=cls_acc,
                                 lr=scheduler.get_last_lr()[0])
                with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, step, loss.item(), cls_loss.item(), bias_loss.item(),adv_acc,cls_acc])
            step+=1

        model_path = config['out_dir']+'model-lr-{}-lam-{}-epoch-{}'.format(config['learning_rate'],config['lambda'],epoch)
        print('save model to {}!!!'.format(model_path))
        torch.save(model.state_dict(),model_path )
        scheduler.step()

        

            

if __name__ == '__main__':
    config = {
        'learning_rate':0.001,
        'batch_size':128,
        'epochs':15,
        'out_dir':'D:/wfy/code/LLM-fairness/save_model/models/adv/',
        'lambda':5,
        'adv':True
    } 


    model_name = 'D:/wfy/code/model/bert_model'
    num_bias_labels = 2  
    model = BertWithAdversary(model_name,num_bias_labels=num_bias_labels,num_profession_label=28,adv=config['adv'])
    model.to(device) 
    tokenizer = BertTokenizer.from_pretrained('D:/wfy/code/model/bert_model')
    

   
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=True)
    train_dataset = dataset_utils.get_bios_dataset(train=True)
    test_dataset = dataset_utils.get_bios_dataset(train=False)
    test_dataset = test_dataset.select(range(10000))
    train_dataloader = DataLoader(dataset=train_dataset,collate_fn=data_collator,batch_size=config['batch_size'])

    train(model=model,config=config,data_loader=train_dataloader)
    pass

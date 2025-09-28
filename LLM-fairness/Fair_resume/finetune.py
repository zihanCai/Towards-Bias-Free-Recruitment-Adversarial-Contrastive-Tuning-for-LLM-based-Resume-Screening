import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import AdamW
import torch.nn.functional as F
import torch.nn as nn
from bert_with_adversary import BertWithAdversary
from tqdm import tqdm
from transformers import BertTokenizer,DataCollatorForLanguageModeling,BigBirdTokenizer,DataCollatorWithPadding
from torch.optim.lr_scheduler import StepLR
# from CustomDataCollator import CustomDataCollatorForLanguageModeling
from dataset_utils import get_resume_dataset_with_gender,get_resume_dataset_with_race
from datasets import concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
# from trainer import AdversarialTrainer
from torch.utils.data import DataLoader
import time
import os
import csv
from transformers import set_seed
from sklearn.metrics import accuracy_score, classification_report,recall_score
from train_utils import *
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
set_seed(42)
def load_model(path = 'all-MiniLM-L6-v2'):

    model = SentenceTransformer(path).to(device)

    
    return model

def custom_collate_fn(batch):
    resumes = [example.texts[0] for example in batch] 
    descriptions = [example.texts[1] for example in batch]  
    labels = [example.label for example in batch] 
    return {
        "resumes": resumes,
        "descriptions": descriptions,
        "labels": torch.tensor(labels, dtype=torch.float32)  
    }
def get_dataloader(data,descriptions,config):
    train_samples = []
    for item in data:
        description = descriptions.loc[descriptions['job'] == item['Category'], 'description'].values[0]
        train_samples.append(
            InputExample(
                texts=[item["Resume_str"], description],  
                label=item["gender_encoded"] 
                # label=item["race_encoded"] 
            )
        )

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config['batch_size'],collate_fn=custom_collate_fn)
    return train_dataloader
def sample_race(dataset):


    race_counts = Counter(dataset['race_encoded'])
    num_black = race_counts.get(0, 0)  
    num_white = race_counts.get(1, 0)  

    target_black = int((num_white * 3) / 7)  

    black_data = dataset.filter(lambda x: x['race_encoded'] == 0)
    white_data = dataset.filter(lambda x: x['race_encoded'] == 1)

    black_data = black_data.shuffle(seed=42).select(range(target_black))

    balanced_dataset = concatenate_datasets([black_data, white_data]).shuffle(seed=42)

    # balanced_counts = Counter(balanced_dataset['race_encoded'])

    return balanced_dataset
def sample_gender(dataset):

    gender_counts = Counter(dataset['gender_encoded'])
    num_female = gender_counts.get(0, 0)  
    num_male = gender_counts.get(1, 0)  

    target_female = int((num_male * 3) / 7)  

    female_data = dataset.filter(lambda x: x['gender_encoded'] == 0)
    male_data = dataset.filter(lambda x: x['gender_encoded'] == 1)

    female_data = female_data.shuffle(seed=42).select(range(target_female))

    balanced_dataset = concatenate_datasets([female_data, male_data]).shuffle(seed=42)

    #balanced_counts = Counter(balanced_dataset['gender_encoded'])
    #print(balanced_counts)
    return balanced_dataset
def train(model,config,data_loader,):

    return 0

if __name__ == '__main__':
    config = {
        'learning_rate':0.001,
        'batch_size':128,
        'epochs':15,
        'out_dir':'D:/wfy/code/LLM-fairness/Fair_resume/save_model/base/',
        'lambda':2.5,
        'adv':False
    } 

    model = load_model(path = 'D:/wfy/model/MiniLM-L6-v2')
    dataset,job_description = get_resume_dataset_with_gender()
    # dataset,job_description = get_resume_dataset_with_race()
    #dataset = sample_race(dataset)
    dataset = sample_gender(dataset)
    dataloader = get_dataloader(data=dataset,descriptions=job_description,config=config)

    main_optimizer = AdamW(model.parameters(),lr = 2e-5, eps = 1e-8)

    main_loss_fn = losses.MultipleNegativesRankingLoss(model)



    model = pretrain_main_task(model=model,
                       optimizer_main=main_optimizer,
                       train_loader=dataloader,
                       loss_criterion=main_loss_fn,
                       epochs=20)
    
    model.save('D:/wfy/code/LLM-fairness/Fair_resume/save_model/models/gender/base/finetune_bert_20ep')
    print("save model!")
    pass

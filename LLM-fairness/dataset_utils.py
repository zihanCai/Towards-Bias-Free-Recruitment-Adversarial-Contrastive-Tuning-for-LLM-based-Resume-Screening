# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertForMaskedLM, BertConfig, BertTokenizer,DataCollatorForLanguageModeling
import datasets

def get_bios_dataset(train = True):
    if train :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Bios_classification/train_dataset')

    else :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Bios_classification/test_dataset')
    dataset = dataset.remove_columns('hard_text')
    return dataset

def get_resume_dataset(train=True):
    if train :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Resume/resume/train_dataset')

    else :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Resume/resume/test_dataset')
    dataset = dataset.remove_columns('hard_text')
    return dataset

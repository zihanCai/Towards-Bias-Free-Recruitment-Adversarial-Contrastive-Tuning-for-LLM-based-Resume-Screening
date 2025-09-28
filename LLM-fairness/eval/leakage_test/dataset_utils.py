# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertForMaskedLM, BertConfig, BertTokenizer,DataCollatorForLanguageModeling
import datasets

def get_bios_dataset(train = True):
    if train :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Bios_sentence/train_dataset')
    else :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Bios_sentence/test_dataset')

    return dataset
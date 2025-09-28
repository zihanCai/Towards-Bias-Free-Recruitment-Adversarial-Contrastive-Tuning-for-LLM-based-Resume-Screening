# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertForMaskedLM, BertConfig, BertTokenizer,DataCollatorForLanguageModeling
import datasets
import pandas as pd
def get_bios_dataset(train = True):
    if train :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Bios_classification/train_dataset')

    else :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Bios_classification/test_dataset')
    dataset = dataset.remove_columns('hard_text')
    return dataset

def get_resume_datasetAndDescription(train=True):
    if train :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Resume/resume/train_dataset_all_encoded_MiniLM')

    else :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Resume/resume/test_dataset_all_encoded_MiniLM')

    job_description = pd.read_csv('D:/wfy/datasets/Resume/job_describe/job_descriptions.csv')
    # dataset = dataset.remove_columns('Resume_str')
    # dataset = dataset.remove_columns('Category')
    dataset = dataset.remove_columns('race')
    dataset = dataset.remove_columns('gender')
    # dataset = dataset.remove_columns('input_ids')
    # dataset = dataset.remove_columns('attention_mask')
    return dataset,job_description
def get_resume_dataset_with_race():
    dataset = datasets.load_from_disk('D:/wfy/datasets/Resume/resume/resume_dataset_with_race_encode')
    job_description = pd.read_csv('D:/wfy/datasets/Resume/job_describe/job_descriptions.csv')
    return dataset,job_description

def get_resume_dataset_with_gender():
    
     dataset = datasets.load_from_disk('D:/wfy/datasets/Resume/resume/resume_dataset_with_gender_encode')
     job_description = pd.read_csv('D:/wfy/datasets/Resume/job_describe/job_descriptions.csv')

     return dataset,job_description

def get_resume_dataset(train=True):
    if train :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Resume/resume/train_dataset_all_encoded_MiniLM')

    else :
        dataset = datasets.load_from_disk('D:/wfy/datasets/Resume/resume/test_dataset_all_encoded_MiniLM')

    job_description = pd.read_csv('D:/wfy/datasets/Resume/job_describe/job_descriptions.csv')
    dataset = dataset.remove_columns('Resume_str')
    dataset = dataset.remove_columns('Category')
    dataset = dataset.remove_columns('race')
    dataset = dataset.remove_columns('gender')
    # dataset = dataset.remove_columns('input_ids')
    # dataset = dataset.remove_columns('attention_mask')
    return dataset
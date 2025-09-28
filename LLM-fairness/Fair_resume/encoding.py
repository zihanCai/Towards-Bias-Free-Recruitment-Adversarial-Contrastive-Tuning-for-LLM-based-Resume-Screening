from transformers import DataCollatorWithPadding,BertModel,BertTokenizer,AutoTokenizer,AutoModel
import torch
from torch.nn.functional import cosine_similarity
from dataset_utils import get_resume_dataset_with_race,get_resume_dataset_with_gender
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_collate_fn(batch):
    resumes = [example.texts[0] for example in batch]  
    labels = [example.label for example in batch]  

    return {
        "resumes": resumes,
        "gender": torch.tensor(labels, dtype=torch.float32),  
    }
    # return {
    #         "resumes": resumes,
    #         "race": torch.tensor(labels, dtype=torch.float32),  
    #     }


def load_model(path='D:/wfy/model/bigbird-roberta-base'):

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    return tokenizer,model


def embedding(model,test_loader,tokenizer):

    model.eval()
    correct = 0
    total = 0
    
    loop = tqdm((test_loader), total = len(test_loader),leave =False)
    embeddings,genders,races,categories = [],[],[],[]
    with torch.no_grad():
        for batch in loop:
            resumes = batch['Resume_str']

            gender = batch["gender_encoded"]  
            # race = batch["race_encoded"] 
            category = batch['Category']

            inputs = tokenizer(
                resumes,
                max_length=512,      
                truncation=True,      
                padding="max_length",  
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(device)  
            attention_mask = inputs["attention_mask"].to(device)

            out = model(input_ids=input_ids,
                        attention_mask=attention_mask)
            
            cls_embedding = out.last_hidden_state[:, 0, :]


            embeddings.extend(cls_embedding.cpu().tolist())
            # races.extend(race.tolist())
            genders.extend(gender.tolist())
            categories.extend(category)

    df = pd.DataFrame({
    'Gender': genders,
    # 'Race': races,
    'Category':categories,
    'Embeddings':embeddings
    })
    
    return df

def get_sentence_embedding_from_text(text, tokenizer, model):
   
    inputs = tokenizer(
        text,
        max_length=512,       
        truncation=True,       
        padding="max_length"   
    )
    input_ids = torch.tensor([inputs["input_ids"]]).to(device)  
    attention_mask = torch.tensor([inputs["attention_mask"]]).to(device)
  
    with torch.no_grad():  
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
    return cls_embedding

def get_sentence_embedding_from_ids(input_ids, attention_mask, model):
   
    input_ids = torch.tensor(input_ids).unsqueeze(0)  
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
    
    return cls_embedding

def calculate_similarity(embedding1, embedding2):
    # similarity = cosine_similarity(embedding1, embedding2)
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    num = embedding1.dot(embedding2.T)
    denom = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return num / denom
    return similarity.item()  



if __name__ == '__main__':

    # test_dataset,_ = get_resume_dataset_with_race()
    test_dataset,_ = get_resume_dataset_with_gender()
    # test_dataset = test_dataset.select(range(100))
    sentence_bert = 'D:/wfy/model/MiniLM-L6-v2'
    sentence_bert_finetune = 'D:/wfy/model/MiniLM-L6-v2-finetuned'
    finetune_20ep = 'D:/wfy/code/LLM-fairness/Fair_resume/save_model/models/gender/base/finetune_bert_20ep'
    adv_bert = 'D:/wfy/code/LLM-fairness/Fair_resume/save_model/models/gender/adv/lr-0.0001-lam-2.5/model_iter-19'
    method_list = [sentence_bert,sentence_bert_finetune,finetune_20ep,adv_bert]

    # collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

    #for path in method_list:
        # text ='Given a job description, retrieve resumes that satisfy the requirements. {}'.format(item['description'])
    tokenizer,model = load_model(sentence_bert)
    model.to(device)
    df = embedding(model=model,test_loader=test_loader,tokenizer=tokenizer)
    path = 'D:/wfy/code/LLM-fairness/Fair_resume/results/gender/sentence_bert/embedding.csv'.format()
    df.to_csv(path)



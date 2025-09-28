import torch
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    EvalPrediction, 
    DataCollatorWithPadding, 
    TrainerCallback
)
import torch.nn.functional as F
import torch.nn as nn
from bert_with_adversary import BertWithAdversaryForMLM
from tqdm import tqdm
from transformers import get_scheduler,BertTokenizer,DataCollatorForLanguageModeling
from CustomDataCollator import CustomDataCollatorForLanguageModeling
import dataset_utils
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from trainer import AdversarialTrainer
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
 
class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"Custom Log: {logs}")

def custom_optimizer_and_scheduler(trainer):
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    return optimizer, scheduler
 
def preprocess_logits_for_metrics(logits, labels):
    return torch.nn.functional.softmax(logits, dim=-1)


def preprocess_data(example):
    encoding = tokenizer(
        example['hard_text'], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'gender': example['gender'],      
    }


model_name = 'D:/wfy/code/model/bert_model'
num_bias_labels = 2  
model = BertWithAdversaryForMLM(model_name,num_bias_labels=num_bias_labels)
model.to(device) 
tokenizer = BertTokenizer.from_pretrained('D:/wfy/code/model/bert_model')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=True)
train_dataset = dataset_utils.get_bios_dataset(train=True)
test_dataset = dataset_utils.get_bios_dataset(train=False)
test_dataset = test_dataset.select(range(10000))
training_args = TrainingArguments(
    output_dir="D:/wfy/code/LLM-fairness/save_model/models",             
    evaluation_strategy="epoch",        
    save_strategy="epoch",              
    learning_rate=5e-5,                 
    per_device_train_batch_size=128,      
    per_device_eval_batch_size=128,       
    num_train_epochs=10,                 
    weight_decay=0.01,                 
    logging_dir="D:/wfy/code/LLM-fairness/save_model/logs",              
    logging_steps=50,                        
    save_total_limit=4,                 
    seed=42,                           
    disable_tqdm=False,                 
    save_safetensors=False            
)

trainer = AdversarialTrainer(
    model=model,                                 
    args=training_args,                       
    train_dataset=train_dataset,              
    eval_dataset=test_dataset,               
    tokenizer=tokenizer,                      
    data_collator=data_collator,              
)

trainer.train()

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup,DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,recall_score
import dataset_utils
from tqdm import tqdm
from fairlearn.metrics import MetricFrame, true_positive_rate,demographic_parity_difference
from bert_classifier import BertClassifier
import numpy as np
def train_epoch(model, data_loader, optimizer, scheduler, device,criterion):
    model.train()
    total_loss = 0
   
    progress_bar = tqdm(data_loader, desc="Training", leave=True)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['profession'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs,labels)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step() 
 
        progress_bar.set_postfix({"Batch_Loss": f"{loss.item():.4f}"})
    
    return total_loss / len(data_loader)
def evaluate(model, data_loader, device):
    model.eval()
    predictions, true_labels,genders = [], [], []
    progress_bar = tqdm(data_loader, desc="test", leave=True)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['profession'].to(device)
            gender = batch['gender'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
            genders.extend(gender.cpu().tolist())

    acc = accuracy_score(true_labels, predictions)
    dp = demographic_parity_difference(y_true=true_labels,y_pred=predictions,sensitive_features=genders)

    genders = np.array(genders)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    y_pred_group_0 = predictions[genders == 0]
    y_true_group_0 = true_labels[genders == 0]
    y_pred_group_1 = predictions[genders == 1]
    y_true_group_1 = true_labels[genders == 1]
    recall_score_0 = recall_score(y_true=y_true_group_0,y_pred=y_pred_group_0,average='macro')
    recall_score_1 = recall_score(y_true=y_true_group_1,y_pred=y_pred_group_1,average='macro')
    tpr_d = abs(recall_score_0-recall_score_1)
    val_report = classification_report(true_labels, predictions, zero_division=0)
    print("tpr_d by sensitive group:")
    print(tpr_d)
    print(f"Demographic Parity: {dp:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print("Validation Report:")
    print(val_report)


def train(epochs,model,train_loader,test_loader,optimizer,scheduler,device,model_save_path,criterion):

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device,criterion)
        
        evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1} Completed")
        print(f"Train Loss: {train_loss:.4f}")


        save_path = model_save_path+"{}".format(epoch+1)
        torch.save(model.state_dict(),save_path )
        print(f"Model saved at {save_path}")
    # tokenizer.save_pretrained("bert_bios_finetuned")
    print("Model saved successfully!")


if __name__ == '__main__':
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_name = 'model-epoch(0)'
    # model_name = 'model-lr-5e-05-lam-5-epoch-8'
    model_name = 'bert_model'
    my_model_path = 'D:/wfy/code/LLM-fairness/save_model/models/model-lr-5e-05-lam-5-epoch-8'

    bert_model_path = 'D:/wfy/code/model/bert_model'
    test_model_path = "D:/wfy/code/LLM-fairness/train_downstream_classification/save_model/model-lr-5e-05-lam-5-epoch-8-downstream-epoch10"
    model = BertClassifier(path=bert_model_path,num_labels=28)
    # model.load_state_dict(torch.load(test_model_path))
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model.to(device)
    model_save_path = f"D:/wfy/code/LLM-fairness/train_downstream_classification/save_model/{model_name}-downstream-epoch"
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = dataset_utils.get_bios_dataset(train=True)
    train_loader = DataLoader(dataset=train_dataset,batch_size=128,collate_fn=data_collator)
    test_dataset = dataset_utils.get_bios_dataset(train=False)
    test_dataset = test_dataset.select(range(1000))
    test_loader = DataLoader(dataset=test_dataset,batch_size=128,collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    epochs = 10
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    train(epochs=epochs,
          model=model,
          train_loader=train_loader,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device,
          model_save_path=model_save_path,
          test_loader=test_loader,
          criterion = criterion)
    # evaluate(model=model,data_loader=test_loader,device=device)

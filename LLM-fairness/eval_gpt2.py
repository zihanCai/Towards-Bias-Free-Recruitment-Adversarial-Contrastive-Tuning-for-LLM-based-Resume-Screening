import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from gpt2_with_adversary import GPT2WithAdversary
from gpt2_data_collator import GPT2DataCollator
from dataset_utils_extended import get_bios_dataset_with_text
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict
import numpy as np
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_gpt2_model(model, data_loader, device):
    model.eval()
    
    all_cls_predictions = []
    all_cls_labels = []
    all_bias_predictions = []
    all_bias_labels = []
    all_cls_logits = []
    all_bias_logits = []
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bias_loss = 0.0
    
    with torch.no_grad():
        loop = tqdm(data_loader, desc="Evaluating")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gender = batch['gender'].to(device)
            profession = batch['profession'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                gender=gender,
                profession=profession,
                lamb=1.0  
            )
            
            cls_preds = torch.argmax(outputs['cls_logits'], dim=1)
            bias_preds = torch.argmax(outputs['bias_logits'], dim=1)
            
            all_cls_predictions.extend(cls_preds.cpu().tolist())
            all_cls_labels.extend(profession.cpu().tolist())
            all_bias_predictions.extend(bias_preds.cpu().tolist())
            all_bias_labels.extend(gender.cpu().tolist())
            
            all_cls_logits.extend(outputs['cls_logits'].cpu().numpy())
            all_bias_logits.extend(outputs['bias_logits'].cpu().numpy())
            
            total_loss += outputs['loss'].item()
            total_cls_loss += outputs['cls_loss'].item()
            total_bias_loss += outputs['bias_loss'].item()
    
    cls_accuracy = accuracy_score(all_cls_labels, all_cls_predictions)
    bias_accuracy = accuracy_score(all_bias_labels, all_bias_predictions)
    
    avg_loss = total_loss / len(data_loader)
    avg_cls_loss = total_cls_loss / len(data_loader)
    avg_bias_loss = total_bias_loss / len(data_loader)
    
    return {
        'cls_accuracy': cls_accuracy,
        'bias_accuracy': bias_accuracy,
        'avg_loss': avg_loss,
        'avg_cls_loss': avg_cls_loss,
        'avg_bias_loss': avg_bias_loss,
        'cls_predictions': all_cls_predictions,
        'cls_labels': all_cls_labels,
        'bias_predictions': all_bias_predictions,
        'bias_labels': all_bias_labels,
        'cls_logits': np.array(all_cls_logits),
        'bias_logits': np.array(all_bias_logits)
    }

def analyze_fairness_metrics(results):
    cls_labels = np.array(results['cls_labels'])
    cls_predictions = np.array(results['cls_predictions'])
    bias_labels = np.array(results['bias_labels'])  # 0: male, 1: female
    
    male_mask = (bias_labels == 0)
    female_mask = (bias_labels == 1)
    
    male_accuracy = accuracy_score(cls_labels[male_mask], cls_predictions[male_mask])
    female_accuracy = accuracy_score(cls_labels[female_mask], cls_predictions[female_mask])
    
    print(f"Male group accuracy: {male_accuracy:.4f}")
    print(f"Female group accuracy: {female_accuracy:.4f}")
    print(f"Accuracy difference: {abs(male_accuracy - female_accuracy):.4f}")
    
    profession_gender_stats = defaultdict(lambda: {'male': 0, 'female': 0})
    
    for profession, gender in zip(cls_predictions, bias_labels):
        if gender == 0:
            profession_gender_stats[profession]['male'] += 1
        else:
            profession_gender_stats[profession]['female'] += 1
    
    return {
        'male_accuracy': male_accuracy,
        'female_accuracy': female_accuracy,
        'accuracy_difference': abs(male_accuracy - female_accuracy),
        'profession_gender_stats': dict(profession_gender_stats)
    }

def load_and_evaluate_gpt2_model(model_path, test_dataset, config):
    model = GPT2WithAdversary(
        model_name='gpt2',
        num_bias_labels=2,
        num_profession_label=28,
        adv=True
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = GPT2DataCollator(tokenizer=tokenizer, max_length=512)
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=data_collator,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    results = evaluate_gpt2_model(model, test_dataloader, device)
    fairness_metrics = analyze_fairness_metrics(results)
    
    return results, fairness_metrics

if __name__ == '__main__':

    config = {
        'batch_size': 64,
        'model_dir': 'D:/wfy/code/LLM-fairness/save_model/models/gpt2_adv/',
    }
    
     test_dataset = get_bios_dataset_with_text(train=False)
    test_dataset = test_dataset.select(range(5000)) 
    
    print("开始评估GPT-2模型...")
    
    for epoch in range(5):  # 假设有5个epoch的模型
        model_path = os.path.join(config['model_dir'], f'gpt2-model-lr-0.001-lam-5-epoch-{epoch}')
        
        if os.path.exists(model_path):
            print(f"\n评估第{epoch}个epoch的模型:")
            try:
                results, fairness_metrics = load_and_evaluate_gpt2_model(model_path, test_dataset, config)
                
                print(f"职业分类准确率: {results['cls_accuracy']:.4f}")
                print(f"性别预测准确率: {results['bias_accuracy']:.4f}")
                print(f"平均总损失: {results['avg_loss']:.4f}")
                print(f"男性组准确率: {fairness_metrics['male_accuracy']:.4f}")
                print(f"女性组准确率: {fairness_metrics['female_accuracy']:.4f}")
                print(f"准确率差异: {fairness_metrics['accuracy_difference']:.4f}")
                
            except Exception as e:
                print(f"评估模型失败: {e}")
        else:
            print(f"模型文件不存在: {model_path}")
    
    print("GPT-2模型评估完成！") 
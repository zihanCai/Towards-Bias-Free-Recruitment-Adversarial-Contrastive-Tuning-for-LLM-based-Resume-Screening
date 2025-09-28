import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForMaskedLM, BertConfig, BertTokenizer,DataCollatorWithPadding
import torch.nn.functional as F
from bert_with_adversary import BertWithAdversary
from dataset_utils import get_bios_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,recall_score
from adapters import BertAdapterModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import pandas as pd
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import numpy as np

import json
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

from itertools import combinations
from tqdm import tqdm

from collections import defaultdict 

def confusion_matrix_based_scores(cnf):
    """Calculate confusion matrix based scores.

    Implementation from https://stackoverflow.com/a/43331484
    See https://en.wikipedia.org/wiki/Confusion_matrix for different scores

    Args:
        cnf (np.array): a confusion matrix.

    Returns:
        dict: a set of metrics for each class, indexed by the metric name.
    """
    FP = cnf.sum(axis=0) - np.diag(cnf) + 1e-5
    FN = cnf.sum(axis=1) - np.diag(cnf) + 1e-5
    TP = np.diag(cnf) + 1e-5
    TN = cnf.sum() - (FP + FN + TP) + 1e-5

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    # Positive Prediction Rates
    PPR = (TP+FP)/(TP+FP+FN+TN)

    return {
        "TPR":TPR,
        "TNR":TNR,
        "PPV":PPV,
        "NPV":NPV,
        "FPR":FPR,
        "FNR":FNR,
        "FDR":FDR,
        "ACC":ACC,
        "PPR":PPR,
    }

def power_mean(series, p, axis=0):
    """calculate the generalized mean of a given list.

    Args:
        series (list): a list of numbers.
        p (int): power of the generalized mean aggregation
        axis (int, optional): aggregation along which dim of the input. Defaults to 0.

    Returns:
        np.array: aggregated scores.
    """
    if p>50:
        return np.max(series, axis=axis)
    elif p<-50:
        return np.min(series, axis=axis)
    else:
        total = np.mean(np.power(series, p), axis=axis)
        return np.power(total, 1 / p)


def Aggregation_GAP(distinct_groups, all_scores, metric="TPR", group_agg_power = None, class_agg_power=28):
    """Aggregate fairness metrics at the group level and class level.

    Args:
        distinct_groups (list): a list of distinc labels of protected groups.
        all_scores (dict): confusion matrix based scores for each protected group and all.
        metric (str, optional): fairness metric. Defaults to "TPR".
        group_agg_power (int, optional): generalized mean aggregation power at the group level. Use absolute value aggregation if None. Defaults to None.
        class_agg_power (int, optional): generalized mean aggregation power at the class level. Defaults to 2.

    Returns:
        np.array: aggregated fairness score.
    """
    group_scores = []
    for gid in distinct_groups:
        group_scores.append(all_scores[gid][metric]) 
    Scores = np.stack(group_scores, axis = 1)
    score_gaps = Scores - all_scores["overall"][metric].reshape(-1,1)
    if group_agg_power is None:
        score_gaps = np.sum(abs(score_gaps),axis=1)
    else:
        score_gaps =power_mean(score_gaps,p=group_agg_power,axis=1)
    score_gaps = power_mean(score_gaps, class_agg_power)

    return score_gaps


def Aggregation_Ratio(distinct_groups, all_scores, metric="TPR", group_agg_power = None, class_agg_power=28):
    """Aggregate fairness metric ratios at the group level and class level.

    Args:
        distinct_groups (list): a list of distinc labels of protected groups.
        all_scores (dict): confusion matrix based scores for each protected group and all.
        metric (str, optional): fairness metric. Defaults to "TPR".
        group_agg_power (int, optional): generalized mean aggregation power at the group level. Use absolute value aggregation if None. Defaults to None.
        class_agg_power (int, optional): generalized mean aggregation power at the class level. Defaults to 2.

    Returns:
        np.array: aggregated fairness score.
    """
    group_scores = []
    for gid in distinct_groups:
        group_scores.append(all_scores[gid][metric]) 
    Scores = np.stack(group_scores, axis = 1)
    score_ratios = Scores / all_scores["overall"][metric].reshape(-1,1)
    if group_agg_power is None:
        score_ratios = np.sum(abs(score_ratios),axis=1)
    else:
        score_ratios =power_mean(score_ratios,p=group_agg_power,axis=1)
    score_ratios = power_mean(score_ratios, class_agg_power)

    return score_ratios

def gap_eval_scores(y_pred, y_true, protected_attribute, metrics=["TPR","FPR","PPR"], args = None):
    """fairness evaluation

    Args:
        y_pred (np.array): model predictions.
        y_true (np.array): target labels.
        protected_attribute (np.array): protected labels.
        metrics (list, optional): a list of metric names that will be considered for fairness evaluation. Defaults to ["TPR","FPR","PPR"].

    Returns:
        tuple: (fairness evaluation results, confusion matrices)
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    protected_attribute = np.array(protected_attribute)

    if (args is not None) and args.regression:
        eval_scores = {
            "mean_absolute_error" : mean_absolute_error(y_true, y_pred),
            "mean_squared_error" : mean_squared_error(y_true, y_pred),
            "r2_score" : r2_score(y_true, y_pred),
        }
        y_true = pd.cut(np.squeeze(y_true), bins=args.regression_bins, labels=False, duplicates = "drop")
        y_pred = pd.cut(np.squeeze(y_pred), bins=args.regression_bins, labels=False, duplicates = "drop")
        y_true = np.nan_to_num(y_true, nan=0)
        y_pred = np.nan_to_num(y_pred, nan=0)

    else:
        
        eval_scores = {
            "accuracy" : accuracy_score(y_true, y_pred),
            "macro_fscore" : f1_score(y_true, y_pred, average="macro"),
            "micro_fscore" : f1_score(y_true, y_pred, average="micro"),
        }

    all_scores = {}
    confusion_matrices = {}

    distinct_labels = [i for i in range(len(set(y_true)))]
    overall_confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=distinct_labels)
    confusion_matrices["overall"] = overall_confusion_matrix
    all_scores["overall"] = confusion_matrix_based_scores(overall_confusion_matrix)

    distinct_groups = [i for i in range(len(set(protected_attribute)))]
    for gid in distinct_groups:
        group_identifier = (protected_attribute ==gid)
        group_confusion_matrix = confusion_matrix(y_true=y_true[group_identifier], y_pred=y_pred[group_identifier], labels=distinct_labels)
        confusion_matrices[gid] = group_confusion_matrix
        all_scores[gid] = confusion_matrix_based_scores(group_confusion_matrix)

    for _metric in metrics:
        eval_scores["{}_GAP".format(_metric)] = Aggregation_GAP(distinct_groups=distinct_groups, all_scores=all_scores, metric=_metric)

    return eval_scores, confusion_matrices


    


def load_model(model_name,num_bias_labels,num_profession_labels,adv):
    bert_model_name = 'D:/wfy/code/model/bert_model'
    num_profession_labels = 28 
    num_bias_labels = 2  
    model = BertWithAdversary(bert_model_name,num_bias_labels=num_bias_labels,
                                                   num_profession_label=num_profession_labels,adv=adv)
    
    model.load_state_dict(torch.load(model_name))
    model.to(device) 
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    return model,tokenizer
def mlm_eval(model,tokenizer):
    occ_list = ['engineer','construction planner','salesperson','primary school teacher','designer','financial officer']
    masked_text = "[MASK] is a "+occ_list[1]+"."
    encoded_input = tokenizer(masked_text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input.to(device))
    logits = output.logits
    mask_token_index = torch.where(encoded_input['input_ids'] == tokenizer.mask_token_id)[1]
    mask_token_logits = logits[0, mask_token_index, :]
    mask_token_probs = F.softmax(mask_token_logits, dim=-1)
    top_5_tokens = torch.topk(mask_token_probs, 5, dim=1)
    print("text:",masked_text)
    for token, prob in zip(top_5_tokens.indices[0].tolist(), top_5_tokens.values[0].tolist()):
        print(f"Predicted token: {tokenizer.decode([token])}, Probability: {prob:.4f}")


def eval(model,test_loader):

    model.eval()
    correct = 0
    total = 0
    loop = tqdm((test_loader), total = len(test_loader),leave =False)
    adv_predictions, genders,professions, predications= [], [], [], []

    with torch.no_grad():

        for batch in loop:
            input_ids = batch['input_ids']
            input_ids = input_ids.to(device)
            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.to(device)
            gender = batch['gender']
            gender = gender.to(device)
            # labels = batch['labels']
            # labels = labels.to(device)
            profession = batch['profession']
            profession = profession.to(device)
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask, 
                        labels=None, gender=gender,lamb=1.5,profession = profession)
            
            loss = out['loss']
            cls_loss = out['cls_loss']
            cls_logits = out['cls_logits']
            bias_loss = out['bias_loss']
            bias_logits = out["bias_logits"]

            # total_loss += loss.item()
            # total_cls_loss += cls_loss.item()
            # total_bias_loss += bias_loss.item()
            
 
            adv_preds = torch.argmax(bias_logits, dim=1)
            adv_predictions.extend(adv_preds.cpu().tolist())
            genders.extend(gender.cpu().tolist())
            preds = torch.argmax(cls_logits,dim=1)
            predications.extend(preds.cpu().tolist())
            professions.extend(profession.cpu().tolist())

        #adv_acc = accuracy_score(genders,adv_predictions)
        #cls_acc = accuracy_score(professions,predications)
        #calculate_metrics_by_group(sensitive_attributes=genders,predictions=predications,true_labels=professions)
    #print('Adv Accuracy:{},   Cls Accuracy:{}'.format(adv_acc,cls_acc))
    eval_scores, confusion_matrices = gap_eval_scores(y_pred=predications,y_true=professions,protected_attribute=genders,
                                                        metrics=["TPR","FPR","PPR"], args = None)
    print(eval_scores,confusion_matrices)

if __name__ == '__main__':

    # test_dataset = torch.load("D:/wfy/datasets/Bios_preprocessed/test_dataset.pt")
    
    
    model,tokenizer = load_model(model_name='D:/wfy/code/LLM-fairness/save_model/models/adv/model-lr-0.001-lam-2-epoch-10',
                                 num_bias_labels=2,num_profession_labels=28,adv=False)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_dataset = get_bios_dataset(train=False)
    test_dataset = test_dataset.select(range(30000))
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True,collate_fn=data_collator)
    eval(model=model,test_loader=test_loader)

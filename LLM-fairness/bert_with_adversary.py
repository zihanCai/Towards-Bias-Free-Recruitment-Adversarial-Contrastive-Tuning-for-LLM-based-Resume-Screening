import torch
import torch.nn as nn
from transformers import BertForMaskedLM,BertModel,BertForSequenceClassification
from peft import PeftConfig,LoraConfig
from adapters import AutoAdapterModel
import torch.nn.functional as F
class Classifier(nn.Module):
  def __init__(self, num_profession_label, bert_hidden_size, hidden_size = 128):
    super().__init__()
    self.linear1 = nn.Linear(bert_hidden_size, hidden_size)
    # self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, num_profession_label)

  def forward(self, clf_last_state):
    output1 = self.linear1(clf_last_state)
    output = self.linear2(output1)
    return output  

class Adversary(nn.Module):
  def __init__(self, num_protected_label, bert_hidden_size, hidden_size = 128):
    super().__init__()

    self.linear1 = nn.Linear(bert_hidden_size, hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, num_protected_label)

  def forward(self, clf_last_state):
    reversed_state = GradientReversalFunction.apply(clf_last_state)
    output1 = self.relu(self.linear1(reversed_state))
    output = self.linear2(output1)
    return output
  
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output
    
class GradientReversalLayer(torch.nn.Module):
    def forward(self, input):
        return GradientReversalFunction.apply(input)


class BertWithAdversary(nn.Module):
    def __init__(self, model_name, num_bias_labels,num_profession_label,adv):
        
        super(BertWithAdversary, self).__init__()
        # self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)
        self.adv = adv
        #self.bert = AutoAdapterModel.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.bert.add_adapter(adapter_config=lora_config,adapter_name='debias')


        self.bias_classifier = Adversary(num_protected_label=num_bias_labels,
                                         bert_hidden_size=self.bert.config.hidden_size)
        self.classifier = Classifier(num_profession_label=num_profession_label,
                                     bert_hidden_size=self.bert.config.hidden_size)
        self.CE = nn.CrossEntropyLoss()
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        
    def save_model(self, path):
        self.bert.save_pretrained(path)

    # @staticmethod
    def forward(self, input_ids, attention_mask, labels, gender,profession,lamb):
        #  train_bias = False
        # _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        hidden_states = outputs[0]
        cls_logits = self.classifier(hidden_states[:, 0, :])
        cls_loss = self.CE(cls_logits,profession)
        # reversed_hidden_state = self.grl(hidden_state)

        bias_logits = self.bias_classifier(hidden_states[:, 0, :])

        # bias_loss = None
        # if gender is not None:
        bias_loss = self.CE(bias_logits, gender)

        #if bias_loss is not None:
        # if train_bias:
            #total_loss = mlm_loss - bias_loss*0.1
        # else:
        if self.adv:
            total_loss = cls_loss + bias_loss*lamb
        else:
            total_loss = cls_loss
        

        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "bias_loss": bias_loss,
            "cls_logits": cls_logits,
            "bias_logits": bias_logits,
        }




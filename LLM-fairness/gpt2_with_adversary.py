import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2TokenizerFast, GPT2Config
from peft import LoraConfig
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, num_profession_label, gpt2_hidden_size, hidden_size=128):
        super().__init__()
        self.linear1 = nn.Linear(gpt2_hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_profession_label)

    def forward(self, clf_last_state):
        output1 = self.linear1(clf_last_state)
        output = self.linear2(output1)
        return output  

class Adversary(nn.Module):
    def __init__(self, num_protected_label, gpt2_hidden_size, hidden_size=128):
        super().__init__()
        self.linear1 = nn.Linear(gpt2_hidden_size, hidden_size)
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

class GPT2WithAdversary(nn.Module):
    def __init__(self, model_name, num_bias_labels, num_profession_label, adv):
        super(GPT2WithAdversary, self).__init__()
        
        self.adv = adv
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        hidden_size = self.gpt2.config.hidden_size
        
        self.bias_classifier = Adversary(
            num_protected_label=num_bias_labels,
            gpt2_hidden_size=hidden_size
        )
        self.classifier = Classifier(
            num_profession_label=num_profession_label,
            gpt2_hidden_size=hidden_size
        )
        self.CE = nn.CrossEntropyLoss()

    def save_model(self, path):
        self.gpt2.save_pretrained(path)

    def forward(self, input_ids, attention_mask, labels, gender, profession, lamb):

        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        hidden_states = outputs[0]
        
        batch_size = input_ids.shape[0]
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1  
        
        last_hidden_states = hidden_states[torch.arange(batch_size), sequence_lengths]
        
        cls_logits = self.classifier(last_hidden_states)
        cls_loss = self.CE(cls_logits, profession)
        
        bias_logits = self.bias_classifier(last_hidden_states)
        bias_loss = self.CE(bias_logits, gender)
        
        if self.adv:
            total_loss = cls_loss + bias_loss * lamb
        else:
            total_loss = cls_loss
        
        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "bias_loss": bias_loss,
            "cls_logits": cls_logits,
            "bias_logits": bias_logits,
        } 
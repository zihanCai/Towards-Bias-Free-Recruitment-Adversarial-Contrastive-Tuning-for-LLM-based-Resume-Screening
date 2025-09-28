import torch
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses,models
from transformers import BertForMaskedLM,BertModel,BertForSequenceClassification,AutoModel,AdamW
from peft import PeftConfig,LoraConfig
from adapters import AutoAdapterModel
import torch.nn.functional as F

class Adversary(nn.Module):
    def __init__(self, embedding_dim,bias_labels = 2):
        super(Adversary, self).__init__()

        self.a1 = nn.Linear(embedding_dim,120)
        self.a2 = nn.Linear(120, bias_labels)

        nn.init.xavier_normal_(self.a1.weight)

    def forward(self, input_ids):


        #Adversary
        adversary = F.relu(self.a1(input_ids))
        adversary_output = self.a2(adversary)

        return adversary_output

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output,None


class AdversarialNetwork(nn.Module):
    def __init__(self, embedding_dim, num_protected_label=2,hidden_dim=128,):
        super(AdversarialNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_protected_label),  # 二分类任务
        )
    
    def forward(self, x, alpha=1.0):
        # x = GradientReversalLayer.apply(x, alpha)
        return self.net(x)

class SentenceWithAdversary(nn.Module):
    def __init__(self, model_name, num_bias_labels,adv):
        
        super(SentenceWithAdversary, self).__init__()
        self.adv = adv
        self.base_model = AutoModel.from_pretrained(model_name)
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.base_model.add_adapter(adapter_config=lora_config,adapter_name='debias')


        self.bias_classifier = AdversarialNetwork(num_protected_label=num_bias_labels,
                                         embedding_dim=self.base_model.config.hidden_size)

        self.CE = nn.CrossEntropyLoss()

        
    def save_model(self, path):
        self.base_model.save_pretrained(path)

    # @staticmethod
    def forward(self, input_ids, attention_mask, gender,lamb):
        # _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        hidden_states = outputs[0]
        


        cls_loss = None
        bias_logits = self.bias_classifier(hidden_states[:, 0, :])

        bias_loss = self.CE(bias_logits, gender)
        if self.adv:
            total_loss = cls_loss + bias_loss*lamb
        else:
            total_loss = cls_loss
        

        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "bias_loss": bias_loss,
            "bias_logits": bias_logits,
        }
    

if __name__ == '__main__':
    model = SentenceWithAdversary(model_name='D:/wfy/model/MiniLM-L6-v2-finetuned',adv=True,num_bias_labels=2)
    loss = losses.MultipleNegativesRankingLoss(model)
    pass

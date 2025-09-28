from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, path, num_labels,hidden_size1 = 384, hidden_size2 = 182):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        # self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(768, hidden_size1)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(hidden_size2, num_labels)  
        for param in self.bert.parameters():
            param.requires_grad = False
        # for param in self.bert.encoder.layer[-1].parameters():
        #     param.requires_grad = True
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        # dropout_output = self.dropout(pooled_output)
        x = self.fc1(pooled_output)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


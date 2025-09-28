import torch
import torch.nn as nn
from transformers import BertModel

class DualEncoderModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", hidden_dim=768):
        super(DualEncoderModel, self).__init__()
        self.resume_encoder = BertModel.from_pretrained(bert_model_name)
        self.job_encoder = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim

    def forward(self, resume_input, job_input):

        resume_emb = self.resume_encoder(**resume_input).pooler_output  # [batch_size, hidden_dim]

        job_emb = self.job_encoder(**job_input).pooler_output  # [batch_size, hidden_dim]
        return resume_emb, job_emb

    def compute_similarity(self, resume_emb, job_emb):

        return torch.matmul(resume_emb, job_emb.T)

def nll_loss(similarity_matrix, positive_indices):

    positive_scores = similarity_matrix[range(similarity_matrix.size(0)), positive_indices]
    log_prob = positive_scores - torch.logsumexp(similarity_matrix, dim=1)
    return -log_prob.mean()
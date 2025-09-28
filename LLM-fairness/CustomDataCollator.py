from transformers import DataCollatorForLanguageModeling
import torch

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, features):
    
        batch = super().__call__(features)
        if "gender" in features[0]:
            batch["gender"] = torch.tensor([f["gender"] for f in features], dtype=torch.long)
        
        return batch

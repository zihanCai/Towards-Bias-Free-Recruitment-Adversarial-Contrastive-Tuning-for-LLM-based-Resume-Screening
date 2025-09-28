from transformers import Trainer
import torch

class AdversarialTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        gender = inputs.pop("gender")
        input_ids = inputs.pop('input_ids')
        attention_mask = inputs.pop("attention_mask")
        outputs = model(input_ids=input_ids,attention_mask=attention_mask, labels=labels, gender=gender)
        loss = outputs["loss"]

        if return_outputs:
            return loss, outputs
        return loss
    def training_step(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()

        return loss.detach()
    

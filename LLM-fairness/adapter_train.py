from transformers import AutoTokenizer, AutoConfig,BertTokenizer,BertConfig
 
from adapters import AutoAdapterModel,BertAdapterModel
from adapters import BnConfig
from transformers import TrainingArguments,DataCollatorWithPadding
import torch
from adapters import AdapterTrainer
import dataset_utils
 
 
model_path = "D:/wfy/code/model/bert_model"
 
tokenizer = BertTokenizer.from_pretrained(model_path)
 
config = BertConfig.from_pretrained(model_path, num_labels=3)
 
model = BertAdapterModel.from_pretrained(model_path, config=config)

 
 
adapter_name = "trouble_shooting"
 

 
config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
 
model.add_adapter(adapter_name, config=config)
 
 

 
model.add_classification_head(adapter_name,num_labels=28, activation_function="relu")
 
 
 
model.train_adapter(adapter_name)

 
 
training_args = TrainingArguments(
 
    num_train_epochs=5,
 
    per_device_train_batch_size = 16,
 
    logging_steps=2,
 
    save_steps = 10,
 
    gradient_accumulation_steps = 4,
 
    output_dir="D:/wfy/code/LLM-fairness/save_model/models/base",
    save_total_limit=10,
 
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataset = dataset_utils.get_bios_dataset(train=True)
train_dataset = train_dataset.rename_column("profession", "labels")
test_dataset = dataset_utils.get_bios_dataset(train=False)
test_dataset = test_dataset.rename_column("profession", "labels")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
trainer = AdapterTrainer (model=model, 
                        tokenizer=tokenizer,
                        args=training_args,
                        train_dataset=train_dataset,
                        optimizers=(optimizer, None),
                        data_collator=data_collator,
                        )
 
trainer.train() 
 
trainer.save_model() 


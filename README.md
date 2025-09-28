## Introduction  
This repository provides code for mitigating biases (e.g., gender, race) in Large Language Models (LLMs) during resume screening tasks. It leverages adversarial training (via `bert_with_adversary.py`, `gpt2_with_adversary.py`) and adapter fine-tuning (`adapter_train.py`) to reduce bias while preserving performance on downstream tasks like job category classification. The project includes modules for model training, embedding generation, and multi-dimensional fairness evaluation (e.g., TPR-GAP, rRD, WEAT).  All the datasets can be downloaded from the google drive link:https://drive.google.com/drive/folders/1DVl1Bec3ZoejqadDnu-YNb7djtqKoYUd?usp=drive_link


## Contributions  
1. **Comprehensive Fairness Evaluation**: Integrates classification fairness metrics (TPR-GAP, FPR-GAP via `eval.py`), ranking fairness (rRD via `Fair_resume/measure/`), and semantic bias detection (WEAT via `eval/seat/weat.py`) to holistically assess model bias.  
2. **Hybrid Bias Mitigation**: Combines adversarial debiasing (for LLMs like BERT and GPT-2) and adapter fine-tuning to balance bias reduction and task performance.  
3. **Modular Design**: Decouples data processing (`dataset_utils.py`), training (`train.py`, `finetune.py`), and evaluation components, enabling easy adaptation to new models or datasets.

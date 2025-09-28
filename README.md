## Introduction  
This project focuses on improving the fairness of Large Language Models (LLMs) in resume screening tasks. It reduces model bias towards protected attributes such as gender and race through techniques like adversarial training and adapter fine-tuning, while maintaining performance in downstream classification tasks (e.g., job category prediction).  

The project includes not only bias mitigation model training modules but also a comprehensive fairness evaluation framework:  
- Fairness metrics based on confusion matrices (e.g., GAP in TPR, FPR, PPR across groups)  
- Ranking fairness metrics (e.g., rRD, rank-based Relative Difference)  
- Word Embedding Association Test (WEAT) to detect potential semantic biases in models  


## Contributions  
1. **Multi-dimensional fairness evaluation mechanism**: Integrates classification task fairness (TPR-GAP, FPR-GAP), ranking fairness (rRD), and semantic bias detection (WEAT) to comprehensively measure model bias levels.  
2. **Efficient bias mitigation strategies**: Combines adversarial training (`bert_with_adversary.py`, `gpt2_with_adversary.py`) and adapter fine-tuning (`adapter_train.py`) to reduce bias while minimizing the impact on task performance.  
3. **Modular and extensible design**: Decouples data processing (`dataset_utils.py`), model training, embedding generation (`encoding.py`), and evaluation (`eval.py`, `Fair_resume/measure/`) modules, supporting quick adaptation to different models (BERT, GPT-2, etc.) and datasets.  


## Usage Methods  

### 1. Environment Setup  
Ensure required packages (including `scikit-learn`, `scipy` for fairness evaluation) are installed:  
```bash
pip install -r requirements_2.txt
```


### 2. Model Training (see the "Model Training" section above)  
- Base model training: `python train_base.py`  
- Adapter training: `python adapter_train.py`  
- Adversarial training: `cd Fair_resume && python finetune.py`  


### 3. Fairness and Performance Evaluation  

#### (1) Classification Task Fairness Evaluation (Metrics like TPR/FPR)  
Use `eval.py` to calculate fairness metrics on model predictions (e.g., TPR differences across protected groups):  
```python
# Example: Evaluate on test set
from eval import gap_eval_scores
import numpy as np

# Assume y_pred is model predictions, y_true is true labels, protected_attribute is protected attributes (e.g., gender)
y_pred = np.array([0, 1, 0, 1, 0])
y_true = np.array([0, 1, 1, 1, 0])
protected_attribute = np.array([0, 0, 1, 1, 0])  # 0/1 represent different groups

# Calculate fairness metrics (TPR-GAP, FPR-GAP, etc.)
eval_scores, _ = gap_eval_scores(y_pred, y_true, protected_attribute)
print("Fairness evaluation results:", eval_scores)
```


#### (2) Ranking Fairness Evaluation (rRD Metric)  
Use `Fair_resume/measure/test.py` to calculate rRD (rank-based Relative Difference) for ranking results:  
```bash
cd Fair_resume/measure
python test.py
```  
- The script loads ranking result CSV files (e.g., `result_{profession}.csv`) and calculates rRD values for different job categories, reflecting the representational differences of protected groups in rankings.  


#### (3) Semantic Bias Detection (WEAT Test)  
Run the WEAT test using `eval/seat/seat.py` to detect potential semantic biases in model embeddings:  
```bash
cd eval/seat
python seat.py --data_dir [test_data_directory] --bert_version [model_version] --results_path [results_save_path]
```  
- Test results include effect size and p-value, measuring the strength of association between target concepts (e.g., occupations) and attributes (e.g., gender).  


## Expected Results  
1. **Model Performance**: Fine-tuned models maintain high accuracy (typically ≥80%) and F1-score (macro-average ≥0.75) in job classification tasks.  
2. **Fairness Improvement**:  
   - TPR-GAP and FPR-GAP of adversarial training and adapter models are reduced by over 30% compared to base models;  
   - rRD values (ranking fairness) are reduced by 20%-40%, indicating more balanced distribution of protected groups in rankings;  
3. **Semantic Bias Mitigation**: WEAT tests show that the effect size of association between occupations and attributes like gender is significantly reduced (absolute value <0.3) with p-value >0.05 (no statistical significance).  

Results can be viewed through CSV files or logs output by `eval.py` and `test.py`, supporting horizontal comparison of different models (base model, adversarial training model, adapter model).

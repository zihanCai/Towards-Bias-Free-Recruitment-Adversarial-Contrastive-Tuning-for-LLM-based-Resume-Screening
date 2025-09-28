class Config:
    def __init__(self):
        pass
    
    def mlm_config(
        self, 
        mlm_probability=0.15, 
        special_tokens_mask=None,
        prob_replace_mask=0.8,
        prob_replace_rand=0.1,
        prob_keep_ori=0.1,
    ):
       
        assert sum([prob_replace_mask, prob_replace_rand, prob_keep_ori]) == 1,                 ValueError("Sum of the probs must equal to 1.")
        self.mlm_probability = mlm_probability
        self.special_tokens_mask = special_tokens_mask
        self.prob_replace_mask = prob_replace_mask
        self.prob_replace_rand = prob_replace_rand
        self.prob_keep_ori = prob_keep_ori
        
    def training_config(
        self,
        batch_size,
        epochs,
        learning_rate,
        weight_decay,
        device,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        
    def io_config(
        self,
        from_path,
        save_path,
    ):
        self.from_path = from_path
        self.save_path = save_path

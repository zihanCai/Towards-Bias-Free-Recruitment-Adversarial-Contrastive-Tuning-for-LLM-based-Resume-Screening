from measures import *
import numpy as np
import pandas as pd
KL_DIVERGENCE = "rKL"
ND_DIFFERENCE = "rND"
RD_DIFFERENCE = "rRD"
def compute_rRD(df, rank_cutoffs, protected_column='Gender', protected_value=-1):

    total_items = len(df)
    
    S_plus_count = df[df[protected_column] == protected_value].shape[0]
    
    S_minus_count = total_items - S_plus_count

    global_ratio = S_plus_count / S_minus_count
    

    weighted_sum = 0
    ideal = 0

    for cutoff in rank_cutoffs:

        top_k = df.iloc[:cutoff]

        S_plus_top_k = top_k[top_k[protected_column] == protected_value].shape[0]

        S_minus_top_k = cutoff - S_plus_top_k

        if S_minus_top_k == 0:
            continue
        
        top_k_ratio = S_plus_top_k / S_minus_top_k

        abs_diff = abs(top_k_ratio - global_ratio)
        
        weight = 1 / np.log2(cutoff)
        
        weighted_sum += weight * abs_diff

    rRD = weighted_sum
    return rRD


def calculate_fairness(df):
    # Specify the input population with size of user and protected group
    user_N = len(df)
    pro_N = (df['Gender'] == -1).sum()
    # normalized fairness follow here  
    # if this input population has been computed, then get from recorded maximum (stored in normalizer.txt)      
    # else compute the normalizer of input population

    # max_rKL = getNormalizer(user_N,pro_N,KL_DIVERGENCE)  
    # max_rND = getNormalizer(user_N,pro_N,ND_DIFFERENCE)
    max_rRD = getNormalizer(user_N,int(pro_N),RD_DIFFERENCE)

    cut_point = 10
    _ranking = df['index'].tolist()
    _protected_group = df[df['Gender'] == -1]['index'].tolist()
    # fair_rKL = calculateNDFairness(test_ranking,pro_index,cut_point,KL_DIVERGENCE,max_rKL)
    # fair_rND = calculateNDFairness(test_ranking,pro_index,cut_point,ND_DIFFERENCE,max_rND)
    fair_rRD = calculateNDFairness(_ranking,_protected_group,cut_point,RD_DIFFERENCE,max_rRD)
    #fair_rRD = compute_rRD(df=df,rank_cutoffs=)
    # print ("rKL of test ranking is ".format(fair_rKL))
    # print ("rND of test ranking is ".format(fair_rND))
    #print ("rKL of test ranking is {}".format(fair_rRD))
    return fair_rRD
import numpy as np

def calculate_exposure_with_discount(df):

    N = len(df)
    bias_score = 0
    for i in range(N):
        g = result_sorted.iloc[i]["Gender"]  
        bias_score += (1 / np.log2(1 + i+1)) * g

    # bias_score /= N  
    return abs(bias_score)
    


def calculate_performance(df,profession):
    N = len(df)
    accuracy = (df['Category'] == profession).sum()
    accuracy = accuracy/N
    return accuracy


if __name__ == '__main__':
    professions = ['HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE', 
                    'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 
                    'BPO', 'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 
                    'CHEF', 'FINANCE', 'APPAREL', 'ENGINEERING', 'ACCOUNTANT', 
                    'CONSTRUCTION', 'PUBLIC-RELATIONS', 'BANKING', 'ARTS', 'AVIATION']
    rank_cutoffs = [10,20,30,40,50]

    rRD_list = []
    gED_list = []
    acc_list = []
    meanRank_list = []
    our_method = 'lr-0.0001-lam-2.5-model_iter-19'
    sentence_bert = 'sentence_bert'
    sentence_bert_finetune = 'sentence_bert_finetune'
    finetune_performence = 'finetune_bert_20ep'
    
    for profession in professions:

        path = 'D:/wfy/code/LLM-fairness/Fair_resume/results/{}/result_{}.csv'.format(finetune_performence,profession)
        result = pd.read_csv(path)
        result["Gender"] = result["Gender"].apply(lambda x: 1 if x == 1 else -1)
        # result["Race"] = result["Race"].apply(lambda x: 1 if x == 1 else -1)
        result_sorted = result.sort_values(by='Similarity',ascending=False, inplace=False).reset_index()
        
        result_sorted = result_sorted[:50]
        pro_N = (result_sorted['Gender'] == -1).sum()
        # max_rRD = getNormalizer(len(result_sorted),int(pro_N),RD_DIFFERENCE)
        # rRD = compute_rRD(df=result_sorted,rank_cutoffs=rank_cutoffs)

        rRD = calculate_fairness(result_sorted)
        gED = calculate_exposure_with_discount(result_sorted)
        # gED = GED(result_sorted=result_sorted)
        # meanRank_0, meanRank_1= mean_rank (result_sorted=test)
        acc = calculate_performance(result_sorted,profession)
        acc_list.append(acc)
        rRD_list.append(rRD)
        gED_list.append(gED)
        # meanRank_list.append((meanRank_0, meanRank_1))
    print('mehotd-{}:{}'.format(finetune_performence,acc_list))

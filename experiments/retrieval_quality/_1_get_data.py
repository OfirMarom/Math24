import glb
import os
import numpy as np
from processing.similar_cases import SimilarCases
import math

top_ks = [1,2,3,4,5]

def is_relevant(test_case, solved_case, label_method):
    if label_method == glb.LabelMethods.any_cat:
        test_cats = [glb.classes[i][0] for i in test_case['classes']]
        solved_cats = [glb.classes[i][0] for i in solved_case['classes']]
        return any(x in test_cats for x in solved_cats)
    elif label_method == glb.LabelMethods.any_cat_and_decomp:
        return any(x in test_case['classes'] for x in solved_case['classes']) 
        
def get_precision_at_k(test_case, k, sim_method, label_method):
    similar_cases = SimilarCases(test_case)
    sim_cases = similar_cases.get(k, sim_method)
    relevant_list = [i for i, _case in enumerate(sim_cases) if is_relevant(test_case, _case, label_method)]
    n_relevant = len(relevant_list)
    precision = n_relevant / k
    return precision  

def get_recall_at_k(test_case, k, sim_method, label_method):
    similar_cases = SimilarCases(test_case)
    solved_cases = similar_cases.solved_cases
    sim_cases = similar_cases.get(k, sim_method)
    relevant_list = [i for i, _case in enumerate(sim_cases) if is_relevant(test_case, _case, label_method)]
    n_relevant = len(relevant_list)
    total_relevant_list = [i for i, _case in enumerate(solved_cases) if is_relevant(test_case, _case, label_method)]
    total_relevant = len(total_relevant_list)
    if total_relevant == 0:
        recall=0
    else:
        recall = n_relevant / total_relevant
    return recall

def get_mean_reciprocal_rank(test_case, k, sim_method, label_method):
    similar_cases = SimilarCases(test_case)
    sim_cases = similar_cases.get(k, sim_method)
    relevant_list = [i for i, _case in enumerate(sim_cases) if is_relevant(test_case, _case, label_method)]
    if not relevant_list:
        return 0
    else:
        return 1 / (min(relevant_list) + 1)
    
def get_normalized_discounted_cumulative_gain(test_case, k, sim_method, label_method):
    similar_cases = SimilarCases(test_case)
    sim_cases = similar_cases.get(k, sim_method)
    relevant_list = [i for i, _case in enumerate(sim_cases) if is_relevant(test_case, _case, label_method)]
    dcg = 0
    idcg = 0
    for i in range(k):
        rank = i + 1
        rel_score = 1 if i in relevant_list else 0
        dcg += rel_score /  math.log2(rank+1) 
        idcg += 1 / math.log2(rank+1) 
    ndcg = dcg / idcg    
    return ndcg     
        

def get_metrics(test_cases, k, sim_method, label_method):
    metrics = { 'precision': [], 
                'recall': [],
                'f1_score': [],
                'mrr': [],
                'ndcg': []
               }
    for test_case in test_cases:
       precision = get_precision_at_k(test_case, k, sim_method, label_method)
       recall = get_recall_at_k(test_case, k, sim_method, label_method)
       f1_score = 0 if precision + recall == 0 else  2 * (precision * recall) / (precision + recall)
       mrr = get_mean_reciprocal_rank(test_case, k, sim_method, label_method)
       ndcg = get_normalized_discounted_cumulative_gain(test_case, k, sim_method, label_method)
       metrics['precision'].append(precision)
       metrics['recall'].append(recall)
       metrics['f1_score'].append(f1_score)
       metrics['mrr'].append(mrr)
       metrics['ndcg'].append(ndcg)
    for key, val in metrics.items():
         metrics[key] = np.mean(val)
         
    return metrics       
    
def save_run(run_index):
    path = os.path.join(glb.results_dir, 'retrieval_data.json')
    if os.path.exists(path):
        results = glb.open_json(path)
    else:        
        results = [None] * len(glb.test_case_seeds)
    cases = glb.open_json(os.path.join(glb.data_dir, 'cases.json'))    
    test_cases = [_case for _case in cases if _case['is_test']==True]  
    result = {}
    for label_method in glb.LabelMethods:
        label_method_value = label_method.value
        if label_method not in result:
            result[label_method_value] = {}
        for k in top_ks:
            if k not in result[label_method_value]:
                result[label_method_value][k] = {}
            latent_metrics = get_metrics(test_cases, k, 'latent', label_method)
            feature_metrics = get_metrics(test_cases, k, 'features_scaled', label_method)
            result[label_method_value][k] = { 'feature': feature_metrics, 'latent': latent_metrics }
    results[run_index] = result 
    glb.save_json(results, path)
        
     

    
  
            

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Lvye

"""
import pandas as pd
import numpy as np
import os

def read_pred_resultsFile(file_pred):
    groups_pred = []
    with open(file_pred) as f:
         groups_pred = [eval(line.strip()) for line in f]
        
    pred_numSum = 0
    for i in range(len(groups_pred)):
        pred_numSum = pred_numSum + len(groups_pred[i])    
        
    return groups_pred, pred_numSum

def get_true_groups(file, k):
    groups_truth = []
    true_numSum = 0
    for i in range(k):
        file_sel = file.loc[file['cluster_id'] == i]
        group_i = np.unique(file_sel['anon_slr_id']).tolist()
        groups_truth.append(group_i)
        true_numSum =  true_numSum + len(group_i)
    return groups_truth, true_numSum

def evaluating_groups_results(groups_pred, groups_truth):
    res = np.zeros((len(groups_pred), len(groups_truth)))
    for i in range(len(groups_pred)):
        pred_group_i = groups_pred[i]
        if len(pred_group_i) > 0 :
            for j in range(len(groups_truth)):
                truth_group_j = groups_truth[j]
                intersect_ij = list(set(pred_group_i).intersection(set(truth_group_j)))
                # intersect_acc = len(intersect_ij) / len(truth_group_j) # recall of each true Group
                intersect_acc = len(intersect_ij)/len(pred_group_i) # precision of this predicted group
                intersect_acc = round(intersect_acc, 4)
                res[i,j] = intersect_acc
    return res

def evaluating_groups_results_recall(groups_pred, groups_truth):
    res = np.zeros((len(groups_pred), len(groups_truth)))
    for i in range(len(groups_pred)):
        pred_group_i = groups_pred[i]
        if len(pred_group_i) > 0 :
            for j in range(len(groups_truth)):
                truth_group_j = groups_truth[j]
                intersect_ij = list(set(pred_group_i).intersection(set(truth_group_j)))
                intersect_acc = len(intersect_ij) / len(truth_group_j) # recall of each true Group
                # intersect_acc = len(intersect_ij)/len(pred_group_i) # precision of this predicted group
                intersect_acc = round(intersect_acc, 4)
                res[i,j] = intersect_acc
    return res

# if __name__ == '__main__':


#     fold_lambda = ['Item_Uniform', 'Item_lambda_vs_34_vb_38']
    
#     for fold in fold_lambda:
#         file_name = None
#         if fold == 'Item_lambda_vs_34_vb_38':
#             file_name = 'Sellers_SynDataAll_45033_10_60_4_Categorical_Categorical'
#         if fold == 'Item_Uniform':
#             file_name = 'Sellers_SynDataAll_45018_10_60_4_Uniform_Uniform'
#         file_root ='/home/dell/Desktop/codes_synthesizing_data/'+ fold +'/then_processed/' + file_name +'.csv'
        
#         # pars #
#         pattern_iter = 10
#         classes = 4 
#         k = 3
#         ############ groudtruth of clustering of sellers ############
#         file = pd.read_csv(file_root, header = 0)
#         seller_set = np.unique(file['anon_slr_id']).tolist()
#         seller_num = len(seller_set)  
#         print('seller number: {}'.format(seller_num), end = '\n')  
#         groups_truth, true_numSum = get_true_groups(file, k)
#         assert(true_numSum == seller_num)  
        
#         save_data_tag ='./SynthesizedData/'+ fold
#         save_data_tag2 = save_data_tag + '/' + file_name    
        
#         path_regrouping = save_data_tag + '/Clustering_Results'
#         if not os.path.exists(path_regrouping):
#             os.mkdir(path_regrouping)
        
  
#         methods_res = ['Dual_Classes_'+ str(classes) + '_results',
#                       'Single_Classes_'+ str(classes) + '_y_results',
#                       'Ours_Classes_'+ str(classes) + '_alpha_0.6_results']  
               
#         for res in methods_res:
#             path_res = save_data_tag2 +'/' + res            
#             for it in range(pattern_iter + 1):
#                 print('iteration: {} '.format(it), end='\n' )        
#                 file_pred = path_res +'/iter_'+ str(it) + '_sellers.txt'  
#                 if os.path.exists(file_pred):
#                     # print(file_pred)
#                     groups_pred, pred_numSum = read_pred_resultsFile(file_pred) 
#                     assert(pred_numSum == seller_num)
                  
#                     #################### evaluating groups_pred and groups_truth  ####################       
#                     res_table = evaluating_groups_results(groups_pred, groups_truth) 
#                     res_table_recall = evaluating_groups_results_recall(groups_pred, groups_truth) 
#                     pd.DataFrame(res_table).to_csv(path_regrouping + '/'+ res[:6] +'_result_iter_'+ str(it) + '_precision.csv' , header = False, index = False)              
#                     pd.DataFrame(res_table_recall).to_csv(path_regrouping + '/'+ res[:6] +'_result_iter_'+ str(it) + '_recall.csv' , header = False, index = False)

#                 else:
#                     print('NO SUCH RESULT FILE: ', end = '\n')
#                     print(file_pred)

            
    



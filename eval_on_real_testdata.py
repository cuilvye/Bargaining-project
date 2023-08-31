#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Evaluate different inference models using the complete real testing data.
@author: Lvye

"""

import numpy as np
import pandas as pd
import os, pickle, argparse

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ""

from tqdm import tqdm
from keras.models import model_from_json
from utils_regrouping import Seller_VsPerformance_on_RealDataset_Ours
from utils_regrouping import Seller_VsPerformance_RealDataset_DualLearning
from utils_regrouping import Seller_VsPerformance_RealDataset_SingleLearning

from model import build_rnn_GRUModel

def load_trained_model(path_model, g, classes):
    model_name = path_model + 'model_iter_0_g_' + str(g) + '.json'

    if os.path.exists(model_name):
        model_weights = path_model + 'model_iter_0_g_' + str(g) + '.h5'
        json_file = open(model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model  = model_from_json(loaded_model_json) 
        model.load_weights(model_weights)
        return model
    else:
        model_weights = path_model + 'model_weights_iter_0_g_' + str(g) + '.h5'
        model = build_rnn_GRUModel(classes) 
        model.load_weights(model_weights)
        return model

def Loaded_preTrained_Models(path_model, model_type, g):
    model_name = path_model + model_type + '_model_iter_0_g_' + str(g) + '.json'
    model_weights = path_model + model_type + '_model_iter_0_g_' + str(g) + '.h5'
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model  = model_from_json(loaded_model_json) 
    model.load_weights(model_weights)
    
    return model    

def predict_SellerVs_on_TestData(file_prior, g_sellers, path_model, alpha, classes, syn_tag, vs_disType, dis_pars):
    
    VsAcc, numSet = [], []
    sellers = np.unique(file_prior['anon_slr_id']).tolist()
    for i in tqdm(range(len(sellers))):
        seller_i = sellers[i]
        dataframePrior_sel = file_prior.loc[file_prior['anon_slr_id'] == seller_i] 
        dataframePrior_sel = dataframePrior_sel.reset_index(drop = True)#
        for g in range(len(g_sellers)):
            if seller_i in g_sellers[g]:
                model = load_trained_model(path_model, g, classes)
                break 
        vsacc, num = Seller_VsPerformance_on_RealDataset_Ours(dataframePrior_sel, alpha, classes, syn_tag, vs_disType, dis_pars, model)
        VsAcc.append(vsacc)
        numSet.append(num)
    
    assert(len(VsAcc) == len(sellers))    
    
    VsAcc_mean = np.sum(np.array(VsAcc) * np.array(numSet)) / np.sum(numSet)
    
    return VsAcc_mean

def Dualpredict_SellerVs_on_TestData(file_prior, g_sellers, path_model, classes, dis_pars):
    
    VsAcc, numSet = [], []
    sellers = np.unique(file_prior['anon_slr_id']).tolist()
    for i in tqdm(range(len(sellers))):
        seller_i = sellers[i]
        dataframePrior_sel = file_prior.loc[file_prior['anon_slr_id'] == seller_i] 
        dataframePrior_sel = dataframePrior_sel.reset_index(drop = True)#
        for g in range(len(g_sellers)):
            if seller_i in g_sellers[g]:
                model = Loaded_preTrained_Models(path_model, 'vs', g)
                break 
        vsacc, num = Seller_VsPerformance_RealDataset_DualLearning(dataframePrior_sel, classes, model, dis_pars)
        VsAcc.append(vsacc)
        numSet.append(num)
    
    assert(len(VsAcc) == len(sellers))    
    
    VsAcc_mean = np.sum(np.array(VsAcc) * np.array(numSet)) / np.sum(numSet)
    
    return VsAcc_mean

def Singlepredict_SellerVs_on_TestData(file_prior, g_sellers, path_model, classes, dis_pars):
    
    VsAcc, numSet = [], []
    sellers = np.unique(file_prior['anon_slr_id']).tolist()
    for i in tqdm(range(len(sellers))):
        seller_i = sellers[i]
        dataframePrior_sel = file_prior.loc[file_prior['anon_slr_id'] == seller_i] 
        dataframePrior_sel = dataframePrior_sel.reset_index(drop = True)#
        for g in range(len(g_sellers)):
            if seller_i in g_sellers[g]:
                model = Loaded_preTrained_Models(path_model, 'classify', g)
                break 
        vsacc, num = Seller_VsPerformance_RealDataset_SingleLearning(dataframePrior_sel, classes, model, dis_pars)
        VsAcc.append(vsacc)
        numSet.append(num)
    
    assert(len(VsAcc) == len(sellers))    
    
    VsAcc_mean = np.sum(np.array(VsAcc) * np.array(numSet)) / np.sum(numSet)
    
    return VsAcc_mean


args_parser = argparse.ArgumentParser()

args_parser.add_argument('--file_path',  default ='./Datasets/RealDataset/Corr/', help = 'the root path of dataset', type = str)
args_parser.add_argument('--file_name', default ='Corr_vs_inference_eBay_Selection_Final_Prior_FS_sel_differ_0_30_VsNum_45', help = 'the dataset name', type = str)
args_parser.add_argument('--classes', default = 5, help = 'the number of action types', type = int)
args_parser.add_argument('--k', default = 3, help = 'the number of clusters', type = int)
args_parser.add_argument('--iter_num', default = 7, help = 'iteration times', type = int)
args_parser.add_argument('--split_idx', default = 1, help = 'different data split:train/valid/test', type = int)
args_parser.add_argument('--model_root', default = './RetrainingModels/RealDataset', help = 'the root path of trained models', type = str)

args = args_parser.parse_args()

file_path = args.file_path
file_name = args.file_name
iter_num = args.iter_num
k = args.k
classes = args.classes
model_root = args.model_root 
split_idx = args.split_idx

syn_tag = False
dis_pars = {'vs_num': 45}

# file_train_prior, file_valid_prior, file_test_prior = None, None, None
# if split_idx == 1:
#     print('Train_' + file_name +'.csv')
#     file_train_prior = pd.read_csv(file_path +'/Train_' + file_name +'.csv', header = 0)
#     file_valid_prior = pd.read_csv(file_path +'/Valid_' + file_name +'.csv', header = 0)
#     file_test_prior = pd.read_csv(file_path +'/Test_' + file_name +'.csv', header = 0)
# else:
    
print('Train' + str(split_idx) + '_' + file_name +'.csv')
file_train_prior = pd.read_csv(file_path +'/Train' +str(split_idx) +'_' + file_name +'.csv', header = 0)
file_valid_prior = pd.read_csv(file_path +'/Valid'+str(split_idx) +'_'+ file_name +'.csv', header = 0)
file_test_prior = pd.read_csv(file_path +'/Test'+str(split_idx) +'_' + file_name +'.csv', header = 0)
 
print('...........Reloading the grouped result...........')
path_group = model_root + '/Grouping_Exps/' + file_name + '_k_' + str(k) + '/Classes_'+ str(classes)
# if k == 3 and 'v2' in file_name: 
#     path_group = './RealDataset/Grouping_Exps/' + file_name  + '/Classes_'+ str(classes)
# if k == 3 and 'v2' not in file_name:
#     path_group = './RealDataset/Grouping_Exps/Classes_'+ str(classes)

with open(path_group +'/IterPred_'+ str(iter_num) +'_Clustered_Sellers_dict.pkl', 'rb') as f:
    clusterDict = pickle.load(f) 

clusters_key = list(clusterDict.keys())
g_sellers = []
for g in range(len(clusters_key)):
    g_sellers.append([])
for g_key in clusters_key:
    g_sellers[int(g_key)] = clusterDict[g_key]


save_data_tag = model_root + '/Experiments_Results/' +  file_name +'_k_' + str(k) + '_iter_' + str(iter_num)

save_path = './Eval/RealData/' + file_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

resultsAcc_table = {'data-type':[],
                    'Ours-NoClustering': [],   'Ours-Clustering': [], 
                    'Dual-NoClustering': [],   'Dual-Clustering': [],
                    'Single-NoClustering': [], 'Single-Clustering': []}

save_model_base = model_root + '/Experiments_Results/'+ file_name
save_model_C = save_data_tag

################################################# data pars  in our algorithm ###########################################################

alpha = 0.6
vs_disType = 'Uniform'

################# the performance with base model(NoClustering model) #################
train_vsAcc_base, valid_vsAcc_base, test_vsAcc_base = 0, 0, 0
# if k == 2: 
path_Basemodel = save_model_base + '/RNN_Split_'+ str(split_idx) + '_NoClustering_Ours_Classes_'+ str(classes) + '_alpha_' + str(alpha) +'_models/'
Model_Base = load_trained_model(path_Basemodel, 0, classes)
train_vsAcc_base, _ = Seller_VsPerformance_on_RealDataset_Ours(file_train_prior, alpha, classes, syn_tag, vs_disType, dis_pars, Model_Base)
valid_vsAcc_base, _ = Seller_VsPerformance_on_RealDataset_Ours(file_valid_prior, alpha, classes, syn_tag, vs_disType, dis_pars, Model_Base)
test_vsAcc_base, _ = Seller_VsPerformance_on_RealDataset_Ours(file_test_prior, alpha, classes, syn_tag, vs_disType, dis_pars, Model_Base)
resultsAcc_table['Ours-NoClustering'].append(round(train_vsAcc_base, 4))
resultsAcc_table['Ours-NoClustering'].append(round(valid_vsAcc_base, 4))
resultsAcc_table['Ours-NoClustering'].append(round(test_vsAcc_base, 4))
print('==================================== Ours-NoClustering=================================================')
print('train_vsacc_base: {}, valid_vsacc_base: {}, test_vsacc_base: {}; '.format(train_vsAcc_base,
                                                                                  valid_vsAcc_base,
                                                                                  test_vsAcc_base), end ='\n')
################# the performance with clustering model #################
train_vsAcc, valid_vsAcc, test_vsAcc = 0, 0, 0
path_model = save_model_C + '/RNN_Split_'+ str(split_idx) +'_Ours_Classes_'+ str(classes) + '_alpha_' + str(alpha) +'_models/'
train_vsAcc = predict_SellerVs_on_TestData(file_train_prior, g_sellers, path_model, alpha, classes, syn_tag, vs_disType, dis_pars)
valid_vsAcc = predict_SellerVs_on_TestData(file_valid_prior, g_sellers, path_model, alpha, classes, syn_tag, vs_disType, dis_pars)
test_vsAcc = predict_SellerVs_on_TestData(file_test_prior, g_sellers, path_model, alpha, classes, syn_tag, vs_disType, dis_pars)
resultsAcc_table['Ours-Clustering'].append(round(train_vsAcc, 4))
resultsAcc_table['Ours-Clustering'].append(round(valid_vsAcc, 4))
resultsAcc_table['Ours-Clustering'].append(round(test_vsAcc, 4))
print('==================================== Ours-Clustering=================================================')
print('train_vsacc: {}, valid_vsacc: {}, test_vsacc: {}; '.format(train_vsAcc,
                                                                  valid_vsAcc,
                                                                  test_vsAcc), end ='\n')


################# the performance with base model(NoClustering model) #################
# if k == 2:
path_model_base = save_model_base +'/Split_'+ str(split_idx) + '_NoClustering_Dual_Classes_'+ str(classes) +'_models/'
Model_vs_Base = Loaded_preTrained_Models(path_model_base, 'vs', 0)
# Model_y_Base = Loaded_preTrained_Models(path_model, 'classify', 0)
train_vsAcc_base, _ = Seller_VsPerformance_RealDataset_DualLearning(file_train_prior, classes,  Model_vs_Base, dis_pars)
valid_vsAcc_base, _ = Seller_VsPerformance_RealDataset_DualLearning(file_valid_prior, classes, Model_vs_Base, dis_pars)
test_vsAcc_base, _ = Seller_VsPerformance_RealDataset_DualLearning(file_test_prior, classes, Model_vs_Base, dis_pars)
resultsAcc_table['Dual-NoClustering'].append(round(train_vsAcc_base, 4))
resultsAcc_table['Dual-NoClustering'].append(round(valid_vsAcc_base, 4))
resultsAcc_table['Dual-NoClustering'].append(round(test_vsAcc_base, 4))
print('==================================== Dual-NoClustering=================================================')
print('train_vsacc_base: {}, valid_vsacc_base: {}, test_vsacc_base: {}; '.format(train_vsAcc_base,
                                                                                  valid_vsAcc_base,
                                                                                  test_vsAcc_base), end ='\n')
# ################# the performance with clustering model #################
path_model = save_model_C + '/Split_'+ str(split_idx) + '_Dual_Classes_'+ str(classes) +'_models/'
train_vsAcc = Dualpredict_SellerVs_on_TestData(file_train_prior, g_sellers, path_model, classes,  dis_pars)
valid_vsAcc = Dualpredict_SellerVs_on_TestData(file_valid_prior, g_sellers, path_model,  classes,  dis_pars)
test_vsAcc = Dualpredict_SellerVs_on_TestData(file_test_prior, g_sellers, path_model,  classes,  dis_pars)
resultsAcc_table['Dual-Clustering'].append(round(train_vsAcc, 4))
resultsAcc_table['Dual-Clustering'].append(round(valid_vsAcc, 4))
resultsAcc_table['Dual-Clustering'].append(round(test_vsAcc, 4))
print('==================================== Dual-Clustering=================================================')
print('train_vsacc: {}, valid_vsacc: {}, test_vsacc: {}; '.format(train_vsAcc,
                                                                  valid_vsAcc,
                                                                  test_vsAcc), end ='\n')
################# the performance with base model(NoClustering model) #################
# if k == 2:
path_model_base = save_model_base +'/Split_'+ str(split_idx) + '_NoClustering_Single_Classes_'+ str(classes) +'_vs_models/'
Model_vs_Base = Loaded_preTrained_Models(path_model_base , 'classify', 0)
# Model_y_Base = Loaded_preTrained_Models(path_model, 'classify', 0)
train_vsAcc_base, _ = Seller_VsPerformance_RealDataset_SingleLearning(file_train_prior, classes,  Model_vs_Base, dis_pars)
valid_vsAcc_base, _ = Seller_VsPerformance_RealDataset_SingleLearning(file_valid_prior, classes, Model_vs_Base, dis_pars)
test_vsAcc_base, _ = Seller_VsPerformance_RealDataset_SingleLearning(file_test_prior, classes, Model_vs_Base, dis_pars)
resultsAcc_table['Single-NoClustering'].append(round(train_vsAcc_base, 4))
resultsAcc_table['Single-NoClustering'].append(round(valid_vsAcc_base, 4))
resultsAcc_table['Single-NoClustering'].append(round(test_vsAcc_base, 4))
print('==================================== Single-NoClustering=================================================')
print('train_vsacc_base: {}, valid_vsacc_base: {}, test_vsacc_base: {}; '.format(train_vsAcc_base,
                                                                                  valid_vsAcc_base,
                                                                                  test_vsAcc_base), end ='\n')
################# the performance with clustering model #################
path_model = save_model_C + '/Split_'+ str(split_idx) +'_Single_Classes_'+ str(classes) + '_vs_models/' 
train_vsAcc = Singlepredict_SellerVs_on_TestData(file_train_prior, g_sellers, path_model, classes,   dis_pars)
valid_vsAcc = Singlepredict_SellerVs_on_TestData(file_valid_prior, g_sellers, path_model,  classes,   dis_pars)
test_vsAcc = Singlepredict_SellerVs_on_TestData(file_test_prior, g_sellers, path_model,  classes,  dis_pars)
resultsAcc_table['Single-Clustering'].append(round(train_vsAcc, 4))
resultsAcc_table['Single-Clustering'].append(round(valid_vsAcc, 4))
resultsAcc_table['Single-Clustering'].append(round(test_vsAcc, 4))
print('==================================== Single-Clustering=================================================')
print('train_vsacc: {}, valid_vsacc: {}, test_vsacc: {}; '.format(train_vsAcc,
                                                                  valid_vsAcc,
                                                                  test_vsAcc), end ='\n')
############# save the result file #############
resultsAcc_table['data-type'].append('train')
resultsAcc_table['data-type'].append('valid')
resultsAcc_table['data-type'].append('test')

pd.DataFrame(resultsAcc_table).to_csv(save_path +'/Classes_' + str(classes) + '_Split_' + str(split_idx) + '_ResultVsAcc.csv', index = False)













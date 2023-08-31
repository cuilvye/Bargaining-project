#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Evaluate different inference models with complete synthetic testing data.
@author: Lvye

"""

import numpy as np
import pandas as pd
import os, pickle
import argparse

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ""

from tqdm import tqdm
from keras.models import model_from_json
from utils_regrouping import Seller_VsPerformance_on_SynDataset_Ours
from utils_regrouping import Seller_VsPerformance_SynDataset_DualLearning
from utils_regrouping import Seller_VsPerformance_SynDataset_SingleLearning
from utils_dataProcessing import Compute_prior_range_data
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
    
    mseSet, VsAcc, numSet = [], [], []
    sellers = np.unique(file_prior['anon_slr_id']).tolist()
    for i in tqdm(range(len(sellers))):
        seller_i = sellers[i]
        dataframePrior_sel = file_prior.loc[file_prior['anon_slr_id'] == seller_i] 
        dataframePrior_sel = dataframePrior_sel.reset_index(drop = True)#
        for g in range(len(g_sellers)):
            if seller_i in g_sellers[g]:
                model = load_trained_model(path_model, g, classes)
                break 
        # print('seller_i id: {}; dataframe shape: {}-{}'.format(seller_i, dataframePrior_sel.shape[0], dataframePrior_sel.shape[1]))
        mse, vsacc, num = Seller_VsPerformance_on_SynDataset_Ours(dataframePrior_sel, alpha, classes, syn_tag, vs_disType, dis_pars, model)
        mseSet.append(mse)
        VsAcc.append(vsacc)
        numSet.append(num)
    
    assert(len(mseSet) == len(sellers))
    assert(len(VsAcc) == len(sellers))    
    
    mse_mean = np.sum(np.array(mseSet) * np.array(numSet)) / np.sum(numSet)
    VsAcc_mean = np.sum(np.array(VsAcc) * np.array(numSet)) / np.sum(numSet)
    
    return mse_mean, VsAcc_mean

def Dualpredict_SellerVs_on_TestData(file_prior, g_sellers, path_model, classes, dis_pars):
    
    mseSet, VsAcc, numSet = [], [], []
    sellers = np.unique(file_prior['anon_slr_id']).tolist()
    for i in tqdm(range(len(sellers))):
        seller_i = sellers[i]
        dataframePrior_sel = file_prior.loc[file_prior['anon_slr_id'] == seller_i] 
        dataframePrior_sel = dataframePrior_sel.reset_index(drop = True)#
        for g in range(len(g_sellers)):
            if seller_i in g_sellers[g]:
                model = Loaded_preTrained_Models(path_model, 'vs', g)
                break 
        mse, vsacc, num = Seller_VsPerformance_SynDataset_DualLearning(dataframePrior_sel, classes, model, dis_pars)
        mseSet.append(mse)
        VsAcc.append(vsacc)
        numSet.append(num)
    
    assert(len(mseSet) == len(sellers))
    assert(len(VsAcc) == len(sellers))    
    
    mse_mean = np.sum(np.array(mseSet) * np.array(numSet)) / np.sum(numSet)
    VsAcc_mean = np.sum(np.array(VsAcc) * np.array(numSet)) / np.sum(numSet)
    
    return mse_mean, VsAcc_mean

def Singlepredict_SellerVs_on_TestData(file_prior, g_sellers, path_model, classes, dis_pars):
    
    mseSet, VsAcc, numSet = [], [], []
    sellers = np.unique(file_prior['anon_slr_id']).tolist()
    for i in tqdm(range(len(sellers))):
        seller_i = sellers[i]
        dataframePrior_sel = file_prior.loc[file_prior['anon_slr_id'] == seller_i] 
        dataframePrior_sel = dataframePrior_sel.reset_index(drop = True)#
        for g in range(len(g_sellers)):
            if seller_i in g_sellers[g]:
                model = Loaded_preTrained_Models(path_model, 'classify', g)
                break 
        mse, vsacc, num = Seller_VsPerformance_SynDataset_SingleLearning(dataframePrior_sel, classes, model, dis_pars)
        mseSet.append(mse)
        VsAcc.append(vsacc)
        numSet.append(num)
    
    assert(len(mseSet) == len(sellers))
    assert(len(VsAcc) == len(sellers))    
    
    mse_mean = np.sum(np.array(mseSet) * np.array(numSet)) / np.sum(numSet)
    VsAcc_mean = np.sum(np.array(VsAcc) * np.array(numSet)) / np.sum(numSet)
    
    return mse_mean, VsAcc_mean


args_parser = argparse.ArgumentParser()
args_parser.add_argument('--file_root',  default ='./Datasets/SynthesizedData/', help = 'the root path of dataset', type = str)
args_parser.add_argument('--fold_lambda', default = 'SynData_Uniform', help = 'the dataset name', type = str)
args_parser.add_argument('--classes', default = 6, help = 'action classes number', type = int)
args_parser.add_argument('--split_idx', default = 1, help = 'different data split:train/valid/test', type = int)
args_parser.add_argument('--model_root', default = './RetrainingModels/SynthesizedData', help = 'the root path of trained models', type = str)
args = args_parser.parse_args()

fold_lambda = args.fold_lambda
file_path = args.file_root
classes = args.classes
split_idx = args.split_idx
model_root = args.model_root

file_name = None
if fold_lambda == 'MINUS_SynData_lambda_vs_54_vb_58':
    # file_name = 'Sellers_length15_SynDataAll_60022_10_100_4_Categorical_Categorical'
    file_name = 'Sellers_length15_SynDataAll_30006_10_100_4_Categorical_Categorical'
elif fold_lambda == 'SynData_Uniform':
    # file_name = 'Sellers_length15_SynDataAll_90012_10_100_4_Uniform_Uniform'
    # file_name = 'V1_Sellers_length15_SynDataAll_45030_10_100_4_Uniform_Uniform'
    file_name = 'Sellers_length15_SynDataAll_120011_10_100_4_Uniform_Uniform'
elif fold_lambda == 'SynData_Skellam_vs_46_vb_66':
    file_name = 'Sellers_length15_SynDataAll_60024_10_100_4_Skellam_Skellam'
elif fold_lambda == 'SynData_Skellam_vs_42_vb_70':
    file_name = 'Sellers_length15_SynDataAll_60014_10_100_4_Skellam_Skellam'
elif fold_lambda == 'SynData_Skellam_vs_54_vb_54':
    # file_name = 'Sellers_length15_SynDataAll_90011_10_100_4_Skellam_Skellam'
    file_name = 'Sellers_length15_SynDataAll_120013_10_100_4_Skellam_Skellam'
elif fold_lambda == 'LessRs_SynData_Uniform':
    file_name = 'Sellers_length15_SynDataAll_45008_10_100_4_Uniform_Uniform'
elif fold_lambda == 'LessRs_SynData_Skellam_vs_54_vb_54':
    file_name = 'Sellers_length15_SynDataAll_45009_10_100_4_Skellam_Skellam'
else:
    assert 1 == 0    

print(fold_lambda)
print('Train' + str(split_idx) + '_' + file_name +'.csv')
train_name = 'TrainPrior'+str(split_idx)+'_' + file_name
valid_name = 'ValidPrior'+str(split_idx)+'_' + file_name
test_name = 'TestPrior'+str(split_idx)+'_' + file_name

file_train = pd.read_csv(file_path + fold_lambda +'/Corr/' + train_name +'.csv', header = 0)
file_valid = pd.read_csv(file_path + fold_lambda +'/Corr/' + valid_name +'.csv', header = 0)        
file_test = pd.read_csv(file_path + fold_lambda +'/Corr/' + test_name +'.csv', header = 0)

file_train_prior, file_valid_prior, file_test_prior = None, None, None
if 'Vs_min' not in list(file_train):
    file_train_prior = Compute_prior_range_data(file_train)
    file_valid_prior = Compute_prior_range_data(file_valid)
    file_test_prior = Compute_prior_range_data(file_test)
else:
    file_train_prior = file_train
    file_valid_prior = file_valid
    file_test_prior = file_test

save_data_tag = model_root + '/' + fold_lambda + '/' + file_name
syn_tag = True
dis_pars = {'price_min': 10, 'price_max': 100, 'gap': 4}

path_data = save_data_tag +'/Transformer_grouped_data/Split_'+ str(split_idx) +'/'
print('...........Reloading the grouped result...........')
with open(path_data +'True_Clustered_Sellers_dict.pkl', 'rb') as f:
    clusterDict = pickle.load(f) 

clusters_key = list(clusterDict.keys())
g_sellers = []
for g in range(len(clusters_key)):
    g_sellers.append([])
for g_key in clusters_key:
    g_sellers[int(g_key)] = clusterDict[g_key]
# g_sellers, pred_numSum = read_pred_resultsFile(file_pred) 

resultsMse_table = {'data-type':[],
                    'Ours-NoClustering': [], 'Ours-Clustering': [], 
                    'Dual-NoClustering': [], 'Dual-Clustering': [],
                    'Single-NoClustering': [], 'Single-Clustering': []}
resultsAcc_table = {'data-type':[],
                    'Ours-NoClustering': [], 'Ours-Clustering': [], 
                    'Dual-NoClustering': [], 'Dual-Clustering': [],
                    'Single-NoClustering': [], 'Single-Clustering': []}

save_path = './Eval/SynthesizedData/' + fold_lambda + '/' + file_name
if not os.path.exists(save_path):
    os.makedirs(save_path)
################################################# data pars  in our algorithm ###########################################################

alpha = 0.6
vs_disType = 'Uniform'

################# the performance with base model(NoClustering model) #################
print('==================================== Ours-NoClustering=================================================')
path_Basemodel = save_data_tag +'/RNN_Split_'+ str(split_idx) +'_NoClustering_Ours_Classes_'+ str(classes) + '_alpha_' + str(alpha) +'_models/'
Model_Base = load_trained_model(path_Basemodel, 0, classes)
train_mse_base, train_vsAcc_base, _ = Seller_VsPerformance_on_SynDataset_Ours(file_train_prior, alpha, classes, syn_tag, vs_disType, dis_pars, Model_Base)
valid_mse_base, valid_vsAcc_base, _ = Seller_VsPerformance_on_SynDataset_Ours(file_valid_prior, alpha, classes, syn_tag, vs_disType, dis_pars, Model_Base)
test_mse_base, test_vsAcc_base, _ = Seller_VsPerformance_on_SynDataset_Ours(file_test_prior, alpha, classes, syn_tag, vs_disType, dis_pars, Model_Base)
resultsMse_table['Ours-NoClustering'].append(round(train_mse_base, 4))
resultsMse_table['Ours-NoClustering'].append(round(valid_mse_base, 4))
resultsMse_table['Ours-NoClustering'].append(round(test_mse_base, 4))
resultsAcc_table['Ours-NoClustering'].append(round(train_vsAcc_base, 4))
resultsAcc_table['Ours-NoClustering'].append(round(valid_vsAcc_base, 4))
resultsAcc_table['Ours-NoClustering'].append(round(test_vsAcc_base, 4))
print('train_mse_base: {}, valid_mse_base: {}, test_mse_base: {}; train_vsacc_base: {}, valid_vsacc_base: {}, test_vsacc_base: {}; '.format(train_mse_base,
                                                                                                                                            valid_mse_base,
                                                                                                                                            test_mse_base,
                                                                                                                                            train_vsAcc_base,
                                                                                                                                            valid_vsAcc_base,
                                                                                                                                            test_vsAcc_base), end ='\n')
# ################# the performance with clustering model #################
print('==================================== Ours-Clustering=================================================')
path_model = save_data_tag +'/RNN_Split_' + str(split_idx) + '_Ours_Classes_'+ str(classes) + '_alpha_' + str(alpha) +'_models/'
train_mse, train_vsAcc = predict_SellerVs_on_TestData(file_train_prior, g_sellers, path_model, alpha, classes, syn_tag, vs_disType, dis_pars)
valid_mse, valid_vsAcc = predict_SellerVs_on_TestData(file_valid_prior, g_sellers, path_model, alpha, classes, syn_tag, vs_disType, dis_pars)
test_mse, test_vsAcc = predict_SellerVs_on_TestData(file_test_prior, g_sellers, path_model, alpha, classes, syn_tag, vs_disType, dis_pars)
resultsMse_table['Ours-Clustering'].append(round(train_mse, 4))
resultsMse_table['Ours-Clustering'].append(round(valid_mse, 4))
resultsMse_table['Ours-Clustering'].append(round(test_mse, 4))
resultsAcc_table['Ours-Clustering'].append(round(train_vsAcc, 4))
resultsAcc_table['Ours-Clustering'].append(round(valid_vsAcc, 4))
resultsAcc_table['Ours-Clustering'].append(round(test_vsAcc, 4))
print('train_mse: {}, valid_mse: {}, test_mse: {}; train_vsacc: {}, valid_vsacc: {}, test_vsacc: {}; '.format(train_mse,
                                                                                                              valid_mse,
                                                                                                              test_mse,
                                                                                                              train_vsAcc,
                                                                                                              valid_vsAcc,
                                                                                                              test_vsAcc), end ='\n')

# ################# the performance with base model(NoClustering model) #################
print('==================================== Dual-NoClustering=================================================')
path_model_base = save_data_tag + '/Split_'+ str(split_idx) +'_NoClustering_Dual_Classes_'+ str(classes) +'_models/'
Model_vs_Base = Loaded_preTrained_Models(path_model_base , 'vs', 0)
# Model_y_Base = Loaded_preTrained_Models(path_model, 'classify', 0)
train_mse_base, train_vsAcc_base, _ = Seller_VsPerformance_SynDataset_DualLearning(file_train_prior, classes,  Model_vs_Base, dis_pars)
valid_mse_base, valid_vsAcc_base, _ = Seller_VsPerformance_SynDataset_DualLearning(file_valid_prior, classes, Model_vs_Base, dis_pars)
test_mse_base, test_vsAcc_base, _ = Seller_VsPerformance_SynDataset_DualLearning(file_test_prior, classes, Model_vs_Base, dis_pars)
resultsMse_table['Dual-NoClustering'].append(round(train_mse_base, 4))
resultsMse_table['Dual-NoClustering'].append(round(valid_mse_base, 4))
resultsMse_table['Dual-NoClustering'].append(round(test_mse_base, 4))
resultsAcc_table['Dual-NoClustering'].append(round(train_vsAcc_base, 4))
resultsAcc_table['Dual-NoClustering'].append(round(valid_vsAcc_base, 4))
resultsAcc_table['Dual-NoClustering'].append(round(test_vsAcc_base, 4))
print('train_mse_base: {}, valid_mse_base: {}, test_mse_base: {}; train_vsacc_base: {}, valid_vsacc_base: {}, test_vsacc_base: {}; '.format(train_mse_base,
                                                                                                                                            valid_mse_base,
                                                                                                                                            test_mse_base,
                                                                                                                                            train_vsAcc_base,
                                                                                                                                            valid_vsAcc_base,
                                                                                                                                            test_vsAcc_base), end ='\n')
# ################# the performance with clustering model #################
print('==================================== Dual-Clustering=================================================')
path_model = save_data_tag + '/Split_'+ str(split_idx) + '_Dual_Classes_'+ str(classes) + '_models/' 
train_mse, train_vsAcc = Dualpredict_SellerVs_on_TestData(file_train_prior, g_sellers, path_model, classes,  dis_pars)
valid_mse, valid_vsAcc = Dualpredict_SellerVs_on_TestData(file_valid_prior, g_sellers, path_model,  classes,  dis_pars)
test_mse, test_vsAcc = Dualpredict_SellerVs_on_TestData(file_test_prior, g_sellers, path_model,  classes,  dis_pars)
resultsMse_table['Dual-Clustering'].append(round(train_mse, 4))
resultsMse_table['Dual-Clustering'].append(round(valid_mse, 4))
resultsMse_table['Dual-Clustering'].append(round(test_mse, 4))
resultsAcc_table['Dual-Clustering'].append(round(train_vsAcc, 4))
resultsAcc_table['Dual-Clustering'].append(round(valid_vsAcc, 4))
resultsAcc_table['Dual-Clustering'].append(round(test_vsAcc, 4))
print('train_mse: {}, valid_mse: {}, test_mse: {}; train_vsacc: {}, valid_vsacc: {}, test_vsacc: {}; '.format(train_mse,
                                                                                                              valid_mse,
                                                                                                              test_mse,
                                                                                                              train_vsAcc,
                                                                                                              valid_vsAcc,
                                                                                                              test_vsAcc ), end ='\n')

################# the performance with base model(NoClustering model) #################
print('==================================== Single-NoClustering=================================================')
path_model_base = save_data_tag + '/Split_'+ str(split_idx) +'_NoClustering_Single_Classes_'+ str(classes) +'_vs_models/'
Model_vs_Base = Loaded_preTrained_Models(path_model_base , 'classify', 0)
# Model_y_Base = Loaded_preTrained_Models(path_model, 'classify', 0)
train_mse_base, train_vsAcc_base, _ = Seller_VsPerformance_SynDataset_SingleLearning(file_train_prior, classes,  Model_vs_Base, dis_pars)
valid_mse_base, valid_vsAcc_base, _ = Seller_VsPerformance_SynDataset_SingleLearning(file_valid_prior, classes, Model_vs_Base, dis_pars)
test_mse_base, test_vsAcc_base, _ = Seller_VsPerformance_SynDataset_SingleLearning(file_test_prior, classes, Model_vs_Base, dis_pars)
resultsMse_table['Single-NoClustering'].append(round(train_mse_base, 4))
resultsMse_table['Single-NoClustering'].append(round(valid_mse_base, 4))
resultsMse_table['Single-NoClustering'].append(round(test_mse_base, 4))
resultsAcc_table['Single-NoClustering'].append(round(train_vsAcc_base, 4))
resultsAcc_table['Single-NoClustering'].append(round(valid_vsAcc_base, 4))
resultsAcc_table['Single-NoClustering'].append(round(test_vsAcc_base, 4))
print('train_mse_base: {}, valid_mse_base: {}, test_mse_base: {}; train_vsacc_base: {}, valid_vsacc_base: {}, test_vsacc_base: {}; '.format(train_mse_base,
                                                                                                                                            valid_mse_base,
                                                                                                                                            test_mse_base,
                                                                                                                                            train_vsAcc_base,
                                                                                                                                            valid_vsAcc_base,
                                                                                                                                            test_vsAcc_base), end ='\n')
################# the performance with clustering model #################
print('==================================== Single-Clustering=================================================')
path_model = save_data_tag +'/Split_'+ str(split_idx) + '_Single_Classes_'+ str(classes) + '_vs_models/' 
train_mse, train_vsAcc = Singlepredict_SellerVs_on_TestData(file_train_prior, g_sellers, path_model, classes,   dis_pars)
valid_mse, valid_vsAcc = Singlepredict_SellerVs_on_TestData(file_valid_prior, g_sellers, path_model,  classes,   dis_pars)
test_mse, test_vsAcc = Singlepredict_SellerVs_on_TestData(file_test_prior, g_sellers, path_model,  classes,  dis_pars)
resultsMse_table['Single-Clustering'].append(round(train_mse, 4))
resultsMse_table['Single-Clustering'].append(round(valid_mse, 4))
resultsMse_table['Single-Clustering'].append(round(test_mse, 4))
resultsAcc_table['Single-Clustering'].append(round(train_vsAcc, 4))
resultsAcc_table['Single-Clustering'].append(round(valid_vsAcc, 4))
resultsAcc_table['Single-Clustering'].append(round(test_vsAcc, 4))
print('train_mse: {}, valid_mse: {}, test_mse: {}; train_vsacc: {}, valid_vsacc: {}, test_vsacc: {}; '.format(train_mse,
                                                                                                              valid_mse,
                                                                                                              test_mse,
                                                                                                              train_vsAcc,
                                                                                                              valid_vsAcc,
                                                                                                              test_vsAcc ), end ='\n')
############# save the result file #############
resultsMse_table['data-type'].append('train')
resultsMse_table['data-type'].append('valid')
resultsMse_table['data-type'].append('test')
resultsAcc_table['data-type'].append('train')
resultsAcc_table['data-type'].append('valid')
resultsAcc_table['data-type'].append('test')

pd.DataFrame(resultsMse_table).to_csv(save_path + '/NEW_Classes_' + str(classes) + '_Split_'+ str(split_idx) +'_resultMse.csv', index = False)
pd.DataFrame(resultsAcc_table).to_csv(save_path + '/NEW_Classes_' + str(classes) + '_Split_'+ str(split_idx) +'_resultVsAcc.csv', index = False)













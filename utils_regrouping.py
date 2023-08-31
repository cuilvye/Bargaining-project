#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Lvye

"""

import os

from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
from model import *
from utils_dataProcessing import *
from utils_inference import *

def get_grouped_data(clusterDict, file_prior):
    clusters_key = list(clusterDict.keys())
    g_data = []
    for i in range(len(clusters_key)):
        data_gi = pd.DataFrame(columns = file_prior.columns.tolist())
        g_data.append(data_gi)           
    
    seller_set = np.unique(file_prior['anon_slr_id']).tolist()
    sellers_clusters = []
    for seller in seller_set:    
        dataframePrior_sel = file_prior.loc[file_prior['anon_slr_id'] == seller] 
        dataframePrior_sel = dataframePrior_sel.reset_index(drop = True)  
        for c_key in clusters_key:
            if seller in clusterDict[c_key]:
                seller_cid = int(c_key)
                sellers_clusters.append(seller_cid)
                g_data[seller_cid] = g_data[seller_cid].append(dataframePrior_sel, ignore_index = True)
                break
    assert(len(sellers_clusters) == len(seller_set))
    
    return g_data
        


def grouping_data_randomly(file, k):
    # divide all sellers into k group randomly #
    seller_set = np.unique(file['anon_slr_id']).tolist()
    seller_num = len(seller_set)
    
    random.shuffle(seller_set)
    
    each_group_size = math.ceil(seller_num / k) # seller_num // k is not OK   
    grouped_sellers = [seller_set [each_group_size * i: each_group_size * (i+1)] for i in range(k)]
    
    check_num = 0
    for g in grouped_sellers:
        check_num = check_num + len(g)
    assert(check_num == seller_num)
    
    # extract dataframe from file for each grouped seller #  
    grouped_data = []
    for g in grouped_sellers:
        data_g = pd.DataFrame(columns = file.columns.tolist())
        for sel in g:
            file_sel = file.loc[file['anon_slr_id'] == sel]
            data_g = data_g.append(file_sel, ignore_index = True)
        grouped_data.append(data_g)
    # Notable: the ele order in grouped_data and grouped_data should be the same
    return grouped_sellers, grouped_data

def compute_min_index(lossList):
    idx_set = []
    min_val = min(lossList)
    for i, loss in enumerate(lossList):
        if loss == min_val:
            idx_set.append(i)
    idx = random.choice(idx_set)        
    return idx, idx_set
  
def Check_if_existed(path_res, pattern_iter):
    
    start = -1
    
    resList = os.listdir(path_res)
    if len(resList) > 0:
        for it in range(pattern_iter):
            if 'iter_'+str(it) +'_sellers.txt' in resList:
                start = it
                continue  
                   
    return start


def save_clustered_data(g_data, path_data, start, file, file_name):
    
    for g in range(len(g_data)):
        g_dataFrame = g_data[g]
        g_dataFrame.to_csv(path_data +'data_iter_'+ str(start) +'_g_'+str(g) +'.csv', index = False)
    
    with open(file_name, 'w') as output:
        for row in file:
            output.write(str(row) + '\n')

def save_clustered_TrueData(g_data, path_data, start, file, file_name):
    
    for g in range(len(g_data)):
        g_dataFrame = g_data[g]
        g_dataFrame.to_csv(path_data +'data_g_'+str(g) +'.csv', index = False)
    
    with open(file_name, 'w') as output:
        for row in file:
            output.write(str(row) + '\n')

def obtain_dataFrame_each_seller(file, seller_set):
    
    sellersPriorData, sellers = [], []
    
    for sel in seller_set:
        dataframe_sel = file.loc[file['anon_slr_id'] == sel] 
        dataframe_sel = dataframe_sel.reset_index(drop = True)# dataframe.loc[] + dataframe.reset_index(drop=True)
        dataframe_prior = Compute_prior_range_data(dataframe_sel)
        
        sellersPriorData.append(dataframe_prior)
        sellers.append(sel)
    
    return sellersPriorData, sellers

def Reobtain_eachGroup_dataFrame(results, sellers, sellersPriorData, file, k):
    g_sellers, g_data = [], []
    for i in range(k):
        data_gi = pd.DataFrame(columns = file.columns.tolist())
        g_data.append(data_gi)
        g_sellers.append([])
    
    for i, idx_g in enumerate(results):
        sel = sellers[i]
        dataframe_i = sellersPriorData[i]
        g_data[idx_g] = g_data[idx_g].append(dataframe_i, ignore_index = True)  
        g_sellers[idx_g].append(sel)
                   
    check_num = 0
    for i, group in enumerate(g_sellers):
        check_num = check_num + len(group)
    assert(check_num == len(sellers))
    print("The sellers regrouping finished ! ", end = '\n')

   ### remove_nullData_group ###
    i = 0
    while i < len(g_sellers):
       if len(g_sellers[i]) == 0 and g_data[i].shape[0] == 0:
           del g_sellers[i]
           del g_data[i]
           i = i
       else:
           i = i + 1 
           
    
    return g_data, g_sellers


def Seller_VsPerformance_on_SynDataset_Ours(dataFrame, alpha, classes, syn_tag, vs_disType, dis_pars, model):
    
    X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
    if classes == 4:
        X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(dataFrame)  
    elif classes == 6:
        X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(dataFrame)
    elif classes >= 15:
        X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(dataFrame, dis_pars) 
        
    Y = to_categorical(Y, classes)   
    X = X.astype(np.float64)    
    x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
    y_tensorBatch = tf.convert_to_tensor(Y)
    xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
    xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
    x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
    
    # infer vs using threads from the same seller and the same item #
    vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, vs_disType, syn_tag, dis_pars, model)
    mse_vs = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch, [-1,1])) 
         
    # compute inferred vs accuracy #
    difference = tf.zeros([xPrior_tensorBatch.get_shape()[0],], dtype = float).numpy()   
    differ_1 = tf.subtract(vs_tensorBatch, xPrior_tensorBatch[:, 0])
    differ_2 = tf.subtract(xPrior_tensorBatch[:, 1], vs_tensorBatch)       
    for i in range(xPrior_tensorBatch.get_shape()[0]):
        if differ_1[i] < 0:
            difference[i] = differ_1[i]
        elif differ_2[i] < 0:
            difference[i] = differ_2[i]
  
    predTrue_num = len(np.where(difference == 0)[0].tolist())
    vsAcc = predTrue_num/vs_tensorBatch.get_shape()[0]
       
    return mse_vs, vsAcc, vs_tensorBatch.get_shape()[0]          

def Seller_VsPerformance_on_SynDataset_Ours_Transformer(dataFrame, alpha, classes, syn_tag, vs_disType, dis_pars, model):
    
    X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
    if classes == 4:
        X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(dataFrame)  
    elif classes == 6:
        X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(dataFrame)
    elif classes >= 15:
        X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(dataFrame, dis_pars) 
        
    Y = to_categorical(Y, classes)   
    X = X.astype(np.float64)    
    x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
    y_tensorBatch = tf.convert_to_tensor(Y)
    xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
    xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
    x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
    
    # infer vs using threads from the same seller and the same item #
    vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor_transformer(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, vs_disType, syn_tag, dis_pars, model)
    mse_vs = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch, [-1,1])) 
         
    # compute inferred vs accuracy #
    difference = tf.zeros([xPrior_tensorBatch.get_shape()[0],], dtype = float).numpy()   
    differ_1 = tf.subtract(vs_tensorBatch, xPrior_tensorBatch[:, 0])
    differ_2 = tf.subtract(xPrior_tensorBatch[:, 1], vs_tensorBatch)       
    for i in range(xPrior_tensorBatch.get_shape()[0]):
        if differ_1[i] < 0:
            difference[i] = differ_1[i]
        elif differ_2[i] < 0:
            difference[i] = differ_2[i]
  
    predTrue_num = len(np.where(difference == 0)[0].tolist())
    vsAcc = predTrue_num/vs_tensorBatch.get_shape()[0]
       
    return mse_vs, vsAcc, vs_tensorBatch.get_shape()[0]          

def Seller_VsPerformance_SynDataset_SingleLearning(dataFrame, classes, model_vs, dis_pars):
    X, Y, X_prior, X_Vs = None, None, None, None
    if classes == 4:
        X, Y, X_prior, _, X_Vs = Convert_to_training_SynData(dataFrame)  
    elif classes == 6:
        X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_Class6(dataFrame)
    elif classes >= 15:
       X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_MultiClass(dataFrame, dis_pars)        
    Y = to_categorical(Y, classes)   
    X = X.astype(np.float64)

    X_Classify = Convert_to_regressor_SynData(X, Y)
    
    xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
    xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
    xClassify_tensorBatch = tf.convert_to_tensor(X_Classify, dtype = float)
    
    dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
    vs_SetTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)
    
    Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
    Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
    vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))
    
    vs_tensorBatch = tf.reshape(vs_tensorBatch, -1)
    mse = mean_squared_error(xVs_tensorBatch, vs_tensorBatch) 
    # compute inferred vs accuracy #
    difference = tf.zeros([xPrior_tensorBatch.get_shape()[0],], dtype = float).numpy()   
    differ_1 = tf.subtract(vs_tensorBatch, xPrior_tensorBatch[:, 0])
    differ_2 = tf.subtract(xPrior_tensorBatch[:, 1], vs_tensorBatch)       
    for i in range(xPrior_tensorBatch.get_shape()[0]):
        if differ_1[i] < 0:
            difference[i] = differ_1[i]
        elif differ_2[i] < 0:
            difference[i] = differ_2[i]
  
    predTrue_num = len(np.where(difference == 0)[0].tolist())
    vsAcc = predTrue_num/vs_tensorBatch.get_shape()[0]
              
    return mse, vsAcc, vs_tensorBatch.get_shape()[0]
    

def Seller_VsPerformance_SynDataset_DualLearning(dataFrame, classes, model_vs, dis_pars):
    
    X, Y, X_prior, X_Vs = None, None, None, None
    if classes == 4:
        X, Y, X_prior, _, X_Vs = Convert_to_training_SynData(dataFrame)  
    elif classes == 6:
        X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_Class6(dataFrame)
    elif classes >= 15:
       X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_MultiClass(dataFrame, dis_pars)        
    Y = to_categorical(Y, classes)   
    X = X.astype(np.float64)    
    x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
    y_tensorBatch = tf.convert_to_tensor(Y)
    xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
    xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
   
    X_classify = Convert_to_regressor_SynData(X, Y)
    xClassify_tensorBatch = tf.convert_to_tensor(X_classify , dtype = float)
    
    dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
    vs_SetTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)
    ##################### y_true + x -> vs #########################
    Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
    Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
    vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch)) #
    
    # print('Seller_VsPerformance_SynDataset_DualLearning bug finding...')
    # print(Vs_logits)
    # print(Y_Pred_trueIdx_tensorBatch)
    # print(tf.squeeze(Y_Pred_trueIdx_tensorBatch))
    # print(xVs_tensorBatch.get_shape())
    # print(vs_tensorBatch.get_shape())
    vs_tensorBatch = tf.reshape(vs_tensorBatch, -1)
    
    mse = mean_squared_error(xVs_tensorBatch, vs_tensorBatch) 
    # compute inferred vs accuracy #
    difference = tf.zeros([xPrior_tensorBatch.get_shape()[0],], dtype = float).numpy()   
    differ_1 = tf.subtract(vs_tensorBatch, xPrior_tensorBatch[:, 0])
    differ_2 = tf.subtract(xPrior_tensorBatch[:, 1], vs_tensorBatch)       
    for i in range(xPrior_tensorBatch.get_shape()[0]):
        if differ_1[i] < 0:
            difference[i] = differ_1[i]
        elif differ_2[i] < 0:
            difference[i] = differ_2[i]
  
    predTrue_num = len(np.where(difference == 0)[0].tolist())
    vsAcc = predTrue_num/vs_tensorBatch.get_shape()[0]
    
    return mse, vsAcc, vs_tensorBatch.get_shape()[0]

# def Seller_performance_on_SynDataset_learnY(dataFrame, classes, model):
#     X, Y, X_prior, X_Vs = None, None, None, None
#     if classes == 4:
#          X, Y, X_prior, _, X_Vs = Convert_to_training_SynData(dataFrame)  
#     elif classes == 6:
#          X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_Class6(dataFrame)
#     elif classes == 15:
#         X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_MultiClass(dataFrame, dis_pars) 
#     Y = to_categorical(Y, classes)   
#     X = X.astype(np.float64)
    
#     x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
#     y_tensorBatch = tf.convert_to_tensor(Y)
#     xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
       
#     # loss B  computation#  
#     x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
#     cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
#     logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
#     loss_B = cce(y_tensorBatch, logits)      
   
#     loss = tf.reduce_mean(loss_B)           
#     acc = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
    
#     return loss, acc

def Seller_VsPerformance_on_RealDataset_Ours(dataFrame, alpha, classes,  syn_tag, vs_disType, dis_pars, model):
    X, Y, X_prior, X_vsID = None, None, None, None
    if classes == 3:
        X, Y, X_prior, X_vsID = Convert_to_training_RealData(dataFrame)  
    elif classes == 5:
        X, Y, X_prior, X_vsID = Convert_to_training_RealData_Classes5(dataFrame)
        
    Y = to_categorical(Y, classes)   
    X = X.astype(np.float64)
    
    x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
    y_tensorBatch = tf.convert_to_tensor(Y)
    xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)  
    x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
    
    # infer vs using threads from the same seller and the same item #
    vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, vs_disType, syn_tag, dis_pars, model)        
    # compute inferred vs accuracy #
    difference = tf.zeros([xPrior_tensorBatch.get_shape()[0],], dtype = float).numpy()   
    differ_1 = tf.subtract(vs_tensorBatch, xPrior_tensorBatch[:, 0])
    differ_2 = tf.subtract(xPrior_tensorBatch[:, 1], vs_tensorBatch)       
    for i in range(xPrior_tensorBatch.get_shape()[0]):
        if differ_1[i] < 0:
            difference[i] = differ_1[i]
        elif differ_2[i] < 0:
            difference[i] = differ_2[i]
  
    predTrue_num = len(np.where(difference == 0)[0].tolist())
    vsAcc = predTrue_num/vs_tensorBatch.get_shape()[0]
            
    return vsAcc, vs_tensorBatch.get_shape()[0]

def Seller_VsPerformance_on_RealDataset_Ours_Transformer(dataFrame, alpha, classes,  syn_tag, vs_disType, dis_pars, model):
    X, Y, X_prior, X_vsID = None, None, None, None
    if classes == 3:
        X, Y, X_prior, X_vsID = Convert_to_training_RealData(dataFrame)  
    elif classes == 5:
        X, Y, X_prior, X_vsID = Convert_to_training_RealData_Classes5(dataFrame)
        
    Y = to_categorical(Y, classes)   
    X = X.astype(np.float64)
    
    x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
    y_tensorBatch = tf.convert_to_tensor(Y)
    xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)  
    x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
    
    # infer vs using threads from the same seller and the same item #
    vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor_transformer(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, vs_disType, syn_tag, dis_pars, model)        
    # compute inferred vs accuracy #
    difference = tf.zeros([xPrior_tensorBatch.get_shape()[0],], dtype = float).numpy()   
    differ_1 = tf.subtract(vs_tensorBatch, xPrior_tensorBatch[:, 0])
    differ_2 = tf.subtract(xPrior_tensorBatch[:, 1], vs_tensorBatch)       
    for i in range(xPrior_tensorBatch.get_shape()[0]):
        if differ_1[i] < 0:
            difference[i] = differ_1[i]
        elif differ_2[i] < 0:
            difference[i] = differ_2[i]
  
    predTrue_num = len(np.where(difference == 0)[0].tolist())
    vsAcc = predTrue_num/vs_tensorBatch.get_shape()[0]
            
    return vsAcc, vs_tensorBatch.get_shape()[0]

def Seller_VsPerformance_RealDataset_DualLearning(dataFrame, classes, model_vs, dis_pars):
    vs_num = dis_pars['vs_num'] 
    Vs = [i for i in range(10, 100, 2)]
    del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
    Vs.append(100)
    vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
    # vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
    X, Y, X_prior = None, None, None
    if classes == 3:
        X, Y, X_prior, _ = Convert_to_training_RealData(dataFrame)  
    elif classes == 5:
        X, Y, X_prior, _ = Convert_to_training_RealData_Classes5(dataFrame)       
    Y = to_categorical(Y, classes)   
    X = X.astype(np.float64)   
    x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
    y_tensorBatch = tf.convert_to_tensor(Y)
    xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)    
    X_classify = Convert_to_regressor_SynData(X, Y)
    xClassify_tensorBatch = tf.convert_to_tensor(X_classify , dtype = float)

    ##################### y_true + x -> vs #########################
    Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
    Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
    vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))   
    vs_tensorBatch = tf.reshape(vs_tensorBatch, -1)
    # compute inferred vs accuracy #
    difference = tf.zeros([xPrior_tensorBatch.get_shape()[0],], dtype = float).numpy()   
    differ_1 = tf.subtract(vs_tensorBatch, xPrior_tensorBatch[:, 0])
    differ_2 = tf.subtract(xPrior_tensorBatch[:, 1], vs_tensorBatch)       
    for i in range(xPrior_tensorBatch.get_shape()[0]):
        if differ_1[i] < 0:
            difference[i] = differ_1[i]
        elif differ_2[i] < 0:
            difference[i] = differ_2[i]
  
    predTrue_num = len(np.where(difference == 0)[0].tolist())
    vs_acc = predTrue_num/vs_tensorBatch.get_shape()[0]
       
    return vs_acc, vs_tensorBatch.get_shape()[0]

# def Seller_performance_on_RealDataset_learnY(dataFrame, classes, model, model_vs, dis_pars):
#     vs_num = dis_pars['vs_num'] 
#     vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
      # Vs = [i for i in range(10, 100, 2)]
      # del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
      # Vs.append(100)
      # vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
#     ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
#     X, Y, X_prior = None, None, None
#     if classes == 3:
#         X, Y, X_prior, _ = Convert_to_training_RealData(dataFrame)  
#     elif classes == 5:
#         X, Y, X_prior, _ = Convert_to_training_RealData_Classes5(dataFrame) 
#     Y = to_categorical(Y, classes)   
#     X = X.astype(np.float64)    
#     x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
#     y_tensorBatch = tf.convert_to_tensor(Y)
#     X_Classify = Convert_to_regressor_SynData(X, Y)                  
#     xClassify_tensorBatch = tf.convert_to_tensor(X_Classify, dtype = float) 
   
#     Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
#     Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
#     vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))
        
#     # loss B  computation#  
#     x_exBatch_Tensor = insert_VS_into_X(vs_tensorBatch, x_tensorBatch)
#     cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
#     logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
#     loss_B = cce(y_tensorBatch, logits)      
#     loss = tf.reduce_mean(loss_B)           
#     acc = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
    
#     return loss, acc

def Seller_VsPerformance_RealDataset_SingleLearning(dataFrame, classes, model_vs, dis_pars):
    X, Y, X_prior = None, None, None
    if classes == 3:
        X, Y, X_prior, _ = Convert_to_training_RealData(dataFrame)  
    elif classes == 5:
        X, Y, X_prior, _ = Convert_to_training_RealData_Classes5(dataFrame)        
    Y = to_categorical(Y, classes)   
    X = X.astype(np.float64)

    X_Classify = Convert_to_regressor_SynData(X, Y)    
    xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
    XClassify_tensorBatch = tf.convert_to_tensor(X_Classify, dtype = float)
    
    vs_num = dis_pars['vs_num'] 
    Vs = [i for i in range(10, 100, 2)]
    del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
    Vs.append(100)
    vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
    # vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
    # VS LOSS #
    Vs_logits = model_vs(tf.reshape(XClassify_tensorBatch, [XClassify_tensorBatch.get_shape()[0], 3, -1])) 
    Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
    vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))   
    vs_tensorBatch = tf.reshape(vs_tensorBatch, -1)
    # compute inferred vs accuracy #
    difference = tf.zeros([xPrior_tensorBatch.get_shape()[0],], dtype = float).numpy()   
    differ_1 = tf.subtract(vs_tensorBatch, xPrior_tensorBatch[:, 0])
    differ_2 = tf.subtract(xPrior_tensorBatch[:, 1], vs_tensorBatch)       
    for i in range(xPrior_tensorBatch.get_shape()[0]):
        if differ_1[i] < 0:
            difference[i] = differ_1[i]
        elif differ_2[i] < 0:
            difference[i] = differ_2[i]
  
    predTrue_num = len(np.where(difference == 0)[0].tolist())
    vsAcc = predTrue_num/vs_tensorBatch.get_shape()[0]    
                 
    return vsAcc, vs_tensorBatch.get_shape()[0]

def load_trained_model(model_name, model_weights):
    if os.path.exists(model_name):
        json_file = open(model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        Loaded_model_i  = model_from_json(loaded_model_json) 
        Loaded_model_i.load_weights(model_weights)
        
        return Loaded_model_i
    else:
        print(model_name + ' CAN NOT BE FOUND!!!')
        print('THIS MAY DUE TO GROUP REDUCTION OF LAST REGROUPING, PLEASE CHECK FOR THIS!')  
        assert(1 == 0)

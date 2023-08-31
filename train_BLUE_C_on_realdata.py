#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is a python file for training BLUE-C model with real bargaining dataset.
@author: Lvye

"""
import numpy as np
import pandas as pd
import time
import os, sys, math, copy
import pickle
import argparse

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
from joblib import Parallel, delayed

from utils_dataProcessing import *
from model import *
from utils_inference import  *
from utils_regrouping import *

def  training_model_epochVS_V0(dataframe_prior_train, alpha, classes, bt_size, model, optimizer, syn_tag, dis_type, dis_pars, epoch, path_res):
           
     batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior_train, bt_size)        
    
     train_loss, train_acc, train_vsAcc = [], [], []
     train_epoch_log = {'batch': [] , 'total_loss': [], 'acc': [],  'vs_acc': [] }
     for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
       
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
        X_train, Y_train, X_train_prior, X_train_vsID = None, None, None, None
        if classes == 3:
            X_train, Y_train, X_train_prior, X_train_vsID= Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X_train, Y_train, X_train_prior, X_train_vsID = Convert_to_training_RealData_Classes5(batch_idataFrame)

        Y_train = to_categorical(Y_train, classes)   
        X_train = X_train.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X_train, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y_train)
        xPrior_tensorBatch = tf.convert_to_tensor(X_train_prior, dtype= float)
        x_vsID_Batch = tf.convert_to_tensor(X_train_vsID, dtype = float)
        
        # infer vs using threads from the same seller and the same item #
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model)             
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
        vs_accBatch = predTrue_num/vs_tensorBatch.get_shape()[0]
         
        with tf.GradientTape() as tape: 
            # loss B  computation#  
            x_exBatch_Tensor = insert_VS_into_X(vs_tensorBatch, x_tensorBatch)
            cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
            # logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))                     
            logits_org = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))      
            logits = tf.clip_by_value(logits_org, 1e-10, 1)
            assert logits_org.get_shape() == logits.get_shape()
            loss_B = cce(y_tensorBatch, logits)      
            
            # loss A computation # 
            vs_priorSet_TensorBatch, vs_priorset_Tensor = Vs_prior_distr_computeTensor(xPrior_tensorBatch, dis_type, syn_tag, dis_pars)
            y_trueIdx_batchTensor = tf.math.argmax(y_tensorBatch, axis = 1)
            
            x_extends = tf.tile(x_tensorBatch, tf.constant([1, vs_priorset_Tensor.get_shape()[0]]))
            x_extends = tf.reshape(x_extends, [x_tensorBatch.get_shape()[0], vs_priorset_Tensor.get_shape()[0] , -1])
            
            x_exBatch_Tensor_timeStep1 = tf.keras.layers.concatenate([vs_priorSet_TensorBatch[:,:,0:1], x_extends[:,:,0:2]], axis = -1)
   
            vs_priorSet_TensorBatch_copy = copy.deepcopy(vs_priorSet_TensorBatch[:,:,0:1])
            check_X_timeStep2_valid = tf.subtract(vs_priorSet_TensorBatch_copy, x_extends[:,:,2:3])
            vs_priorSet_TensorBatch_copy = tf.where(check_X_timeStep2_valid == vs_priorSet_TensorBatch[:,:,0:1], tf.zeros_like(vs_priorSet_TensorBatch_copy), vs_priorSet_TensorBatch_copy)
            x_exBatch_Tensor_timeStep2 = tf.keras.layers.concatenate([vs_priorSet_TensorBatch_copy, x_extends[:,:,2:4]], axis = -1)
               
            vs_priorSet_TensorBatch_Copy = copy.deepcopy(vs_priorSet_TensorBatch[:,:,0:1])
            check_X_timeStep3_valid = tf.subtract(vs_priorSet_TensorBatch_Copy, x_extends[:,:,4:5])
            vs_priorSet_TensorBatch_Copy = tf.where(check_X_timeStep3_valid == vs_priorSet_TensorBatch[:,:,0:1], tf.zeros_like(vs_priorSet_TensorBatch_Copy), vs_priorSet_TensorBatch_Copy)    
            x_exBatch_Tensor_timeStep3 = tf.keras.layers.concatenate([vs_priorSet_TensorBatch_Copy, x_extends[:,:,4:6]], axis = -1)
               
            x_exBatch_Tensor_timeStep_temp = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep1, x_exBatch_Tensor_timeStep2], axis = -1)
            x_extends_Tensor = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep_temp, x_exBatch_Tensor_timeStep3], axis = -1)

            vs_probs = tf.gather(vs_priorset_Tensor, [1], axis = 1)
            vs_setTensor = tf.gather(vs_priorset_Tensor, [0], axis = 1)
            
            loss_A_grad, real_loss_A = [], []
            for i in range(x_extends_Tensor.get_shape()[0]):
                xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], 3, -1]))
                yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)
    
                yi_probs = tf.clip_by_value(yi_probs, 1e-10, 1)
    
                yiProbs_onVs_pVs = tf.math.multiply(yi_probs, vs_probs)
                sum_Denom = tf.reduce_sum(yiProbs_onVs_pVs) 
                
                res_1 = tf.squeeze(tf.where(tf.squeeze(vs_setTensor, axis = 1) <= xPrior_tensorBatch[i, 1]), axis = 1)
                res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= tf.squeeze(vs_setTensor, axis = 1)), axis = 1)
                res = tf.concat((res_1, res_2), axis = 0)
                unique_res_vals, unique_idx = tf.unique(res)
                count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                           unique_idx,
                                                           tf.shape(res)[0])
                more_than_one = tf.greater(count_res_unique, 1)
                more_than_one_idx = tf.squeeze(tf.where(more_than_one))
                more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
                yiProbs_onVs_pVs_feaSet = tf.gather(tf.squeeze(yiProbs_onVs_pVs, axis = 1), more_than_one_vals)                                  

                # print(yiProbs_onVs_pVs_feaSet)
                sum_Numo = tf.reduce_sum(yiProbs_onVs_pVs_feaSet) 
                # print('{}/{}'.format(sum_Numo, sum_Denom))
                log_pA_i = sum_Numo / sum_Numo.numpy() -  sum_Denom / sum_Denom.numpy()
                loss_A_i = - log_pA_i                
                loss_A_grad.append(loss_A_i)         
                
                # loss_res = sum_Numo/sum_Denom
                # if loss_res == 0:
                #     loss_res = 1e-323
                # real_loss_A_i = - math.log(loss_res) #sum_Numo/sum_Denom
                real_loss_A_i = tf.math.log(sum_Denom) - tf.math.log(sum_Numo) 
                real_loss_A.append(real_loss_A_i)

            loss_total_grad = alpha * tf.reduce_mean(loss_A_grad)  + (1-alpha) * tf.reduce_mean(loss_B)
            loss_total_real = alpha * tf.reduce_mean(real_loss_A) + (1-alpha) * tf.reduce_mean(loss_B)

        gradients = tape.gradient(loss_total_grad, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        # prediction accuracy #
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
        loss_mean = loss_total_real.numpy()
        
        train_acc.append(acc_batch)
        train_vsAcc.append(vs_accBatch) 
        train_loss.append(loss_mean)                 
        train_epoch_log['batch'].append(batch + 1)
        train_epoch_log['acc'].append(acc_batch)
        train_epoch_log['vs_acc'].append(vs_accBatch)
        train_epoch_log['total_loss'].append(loss_mean)   
       
        # if (batch + 1)  % 1 == 0 or (batch + 1) == len(batch_dataFrame):
        #     print("[INFO] training batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f},  vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, vs_accBatch), end = "\n")                       
    
     # pd.DataFrame(train_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_train_epoch_'+ str(epoch) +'.csv', index = False)    
     train_loss_mean = np.mean(train_loss)
     train_acc_mean = np.mean(train_acc)
     train_vsAcc_mean = np.mean(train_vsAcc)
    
     return model, train_loss_mean, train_acc_mean, train_vsAcc_mean


def learning_model_paras_V0(dataframe_prior_train, dataframe_prior_valid, alpha, classes, epochs, batch_size, lr, syn_tag, vs_disType, dis_pars, save_tuple, path_tuple):
    
    path_res, path_model = path_tuple[0], path_tuple[1]
    it, g = save_tuple[0], save_tuple[1]
    
    # dataframe_prior_valid = pd.DataFrame(columns = dataframe_prior_valid_temp.columns.tolist())
    # valid_sellers = np.unique(dataframe_prior_valid_temp['anon_slr_id']).tolist()
    # for seller in valid_sellers:    
    #     if seller in loaded_dict[str(cluster_id)]:
    #         dataframePrior_sel = dataframe_prior_valid_temp.loc[dataframe_prior_valid_temp['anon_slr_id'] == seller] 
    #         dataframePrior_sel = dataframePrior_sel.reset_index(drop = True)
    #         dataframe_prior_valid = dataframe_prior_valid.append(dataframePrior_sel, ignore_index = True)
    # print('the original valid dataset shape is {}-{} '.format(dataframe_prior_valid_temp.shape[0], dataframe_prior_valid_temp.shape[1]))  
    # print('For this subgroup data, valid dataset shape is {}-{} '.format(dataframe_prior_valid.shape[0], dataframe_prior_valid.shape[1]))  
    
    model = None    
    if os.path.exists(path_model + 'model_iter_'+ str(it) +'_g_' + str(g) + '.json'):
        print("[INFO] ================ reloading the model with classes {} ==================".format(classes), end = "\n")    
        model_name = path_model + 'model_iter_'+ str(it) +'_g_' + str(g) + '.json'
        model_weights = path_model + 'model_iter_'+ str(it) +'_g_' + str(g) + '.h5'
        json_file = open(model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model  = model_from_json(loaded_model_json) 
        model.load_weights(model_weights)
    else:
        print("[INFO] ================ creating the model with classes {} ==================".format(classes), end = "\n")   
        model = build_rnn_GRUModel(classes)
     
    optimizer = Adam(learning_rate = lr, amsgrad=True)#
    model.compile(optimizer = 'adam', loss='categorical_crossentropy') 

    ###############################################   start training model ################################# 
    start, train_log = 0, None
    if os.path.exists(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_train_epochs_'+ str(epochs) +'.csv'):
        org_train_log = pd.read_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_train_epochs_'+ str(epochs) +'.csv', header = 0) 
        if org_train_log.shape[0] > 0:
            start = int(np.array(org_train_log)[-1, 0])
        train_log = org_train_log.to_dict(orient = "list")
    else:    
        train_log = {'epoch': [] , 
                     'train_lossTotal': [], 'train_acc': [],'train_vsAcc':[],
                     'valid_lossTotal': [], 'valid_acc': [],'valid_vsAcc':[]}
    
    print("[INFO] ================ starting training the model from {} ==================".format(start), end = "\n")  
    for epoch in range(start, epochs):
        sys.stdout.flush()
        # loop over the data in batch size increments
        epochStart = time.time()
        model, loss_train, acc_train, vsAcc_train = training_model_epochVS_V0(dataframe_prior_train, alpha, classes, batch_size, model, optimizer, syn_tag, vs_disType, dis_pars, epoch, path_res)       
        elapsed = (time.time() - epochStart) / 60.0
        print("[INFO] one epoch took {:.4} minutes".format(elapsed), end = "\n")
        ## check if needing valid dataset to adjust the learning rate #
        isTest = False
        loss_v, acc_v, vsAcc_v  = performance_on_RealDataset_Multirows_V0_alphaLog(dataframe_prior_valid, alpha, classes, batch_size, syn_tag, vs_disType, dis_pars, model, isTest, path_res)
                                                                                    
        train_log['epoch'].append(epoch+1)
        train_log['train_lossTotal'].append(loss_train)
        train_log['train_acc'].append(acc_train)
        train_log['train_vsAcc'].append(vsAcc_train)
        
        train_log['valid_lossTotal'].append(loss_v)
        train_log['valid_acc'].append(acc_v)
        train_log['valid_vsAcc'].append(vsAcc_v)
        print("[INFO] epoch {}/{}, train total_loss: {:.5f}, train accuracy: {:.5f}, train vs_acc: {:.5f}; ".format(epoch+1, epochs, loss_train, acc_train,  vsAcc_train), end = "\n")            
        print("[INFO] epoch {}/{}, valid total_loss: {:.5f}, valid accuracy: {:.5f}, valid vs_acc: {:.5f};".format(epoch+1, epochs, loss_v, acc_v, vsAcc_v), end = "\n")            
        pd.DataFrame(train_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_train_epochs_'+ str(epochs) +'.csv', index = False)    
        
        model_json = model.to_json()
        with open(path_model + 'model_iter_'+ str(it) +'_g_' + str(g) + '.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(path_model + 'model_iter_'+ str(it) +'_g_' + str(g) + '.h5')
        
        stopEarly = Callback_EarlyStopping(train_log['valid_lossTotal'], min_delta = 0.01, patience = 30)
        if stopEarly:
            print('Callback Earlystopping signal received at epoch: {}/{} '.format(epoch+1, epochs))
            print('Terminating training ')
            break
    
    # # performance on test dataset #  
    # isTest = False    
    # loss_test, acc_test, vsAcc_test  = performance_on_RealDataset_Multirows_V0_alphaLog(dataframe_prior_test, alpha, classes, batch_size, syn_tag, vs_disType, dis_pars, model, isTest, path_res)
                                                                                              
    # test_res = {'lossTotal_test': str(loss_test), 'acc_test': str(acc_test), 'vsAcc_test': str(vsAcc_test)} 
    # with open(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_test.txt', 'w') as output:
    #     output.write(json.dumps(test_res)) # use 'json.loads' to do the reverse
    # print("[INFO] test total_loss: {:.5f}, accuracy: {:.5f}, vs_acc: {:.5f}".format(loss_test, acc_test, vsAcc_test), end = "\n")  
    
    # model_json = model.to_json()
    # with open(path_model + 'model_iter_'+ str(it) +'_g_' + str(g) + '.json', 'w') as json_file:
    #     json_file.write(model_json)
    # model.save_weights(path_model + 'model_iter_'+ str(it) +'_g_' + str(g) + '.h5')
    # print(' Saved model it-{}-g-{} finally! '.format(it, g), end = '\n')

# if __name__ == '__main__':
    # each row of the file is for(seller_id, item_id), its one complete bargaining thread with one buyer
    # e.g. 1, a,  20, 10, 15 (buyer 1)
    #      1, a,  22, 15, 22, 16, A (buyer 2)
    #      1, b,  13, 12, A 
    # and each row can at most generate 3 [x, y]s, we need to convert to the form of [x, y] before training the RNN
    
print('*************************** Running BLUE-C Exps on Real Dataset ***************************')
args_parser = argparse.ArgumentParser()

args_parser.add_argument('--file_root',  default ='./Datasets/RealDataset/', help = 'the root path of dataset', type = str)
args_parser.add_argument('--file_name', default = 'Corr_vs_inference_eBay_Selection_Final_Prior_FS_sel_differ_0_30_VsNum_45', help = 'the dataset name', type = str)
args_parser.add_argument('--classes', default = 5, help = 'the class number of action', type = int)
args_parser.add_argument('--alpha', default = 0.6, help = 'the weight of the first cost function', type = float)
args_parser.add_argument('--k', default = 3, help = 'the number of clusters', type = int)
args_parser.add_argument('--epochs', default = 500, help = 'the training epochs', type = int)
args_parser.add_argument('--batch', default = 64, help = 'the size of training batch ', type = int)
args_parser.add_argument('--lr', default = 0.001, help = 'the learning rate of optimizer ', type = float)
args_parser.add_argument('--vsdisType', default = 'Uniform', help = 'the prior distribution of Vs ', type = str)
args_parser.add_argument('--synTag', default = False, help = 'the data is synthesized or real', type = bool)
args_parser.add_argument('--disPars', default = {'vs_num': 45}, help = 'the distribution pars of dataset ', type = dict)
args_parser.add_argument('--iter_num', default = 7, help = 'the iteration times in K-Loss clustering algorithm', type = int)
args_parser.add_argument('--cluster_path',  default ='./RetrainingModels/RealDataset/Grouping_Exps', help = 'the file path of cluster results', type = str)
args_parser.add_argument('--split_idx', default = 1, help = 'different data split: train/valid/test', type = int)
args_parser.add_argument('--save_root', default = './RetrainingModels/RealDataset/Experiments_Results', help = 'save path', type = str)

args = args_parser.parse_args()

file_path = args.file_root
file_name = args.file_name
cluster_path = args.cluster_path
save_root = args.save_root

# hyper parameter #    
classes = args.classes
alpha = args.alpha
k = args.k
epochs = args.epochs
batch_size = args.batch
lr = args.lr  
vs_disType = args.vsdisType

# data pars # 
syn_tag = args.synTag
dis_pars = args.disPars
split_idx = args.split_idx
  
print('...........Reloading the grouped result...........')
iter_num = args.iter_num
path_group = cluster_path + '/' + file_name + '_k_'+str(k)+'/Classes_'+ str(classes) # 
with open(path_group +'/IterPred_'+ str(iter_num) +'_Clustered_Sellers_dict.pkl', 'rb') as f: # pre: 7
    clusterDict = pickle.load(f) 
    clusters_key = list(clusterDict.keys())  

k = len(clusters_key)
print('The clusters number is {}'.format(k))

# save data, results, models into separated data-type #
save_data_tag = save_root
save_data_tag = save_data_tag +'/' +  file_name + '_k_' + str(k) +'_iter_' + str(iter_num)
  
path_data = save_data_tag +'/Transformer_grouped_data/Split_' + str(split_idx) + '/'  
path_model = save_data_tag + '/RNN_Split_' + str(split_idx) +'_Ours_Classes_'+ str(classes) + '_alpha_' + str(alpha) +'_models/'
path_res = save_data_tag +'/RNN_Split_' + str(split_idx) + '_Ours_Classes_'+ str(classes) + '_alpha_' + str(alpha) +'_results/' 

if not os.path.exists(save_data_tag):
    os.makedirs(save_data_tag)
if not os.path.exists(path_data):
    os.makedirs(path_data)
if not os.path.exists(path_model):
    os.makedirs(path_model)
if not os.path.exists(path_res):
    os.makedirs(path_res)  
  
# trainPrior_dataframe, validPrior_dataframe = None, None
# if split_idx == 1:
#     print('Train_' + file_name +'.csv')
#     trainPrior_dataframe = pd.read_csv(file_path +'/Train_' + file_name +'.csv', header = 0)
#     validPrior_dataframe = pd.read_csv(file_path +'/Valid_' + file_name +'.csv', header = 0) 
# else:
print('Train' + str(split_idx) + '_' + file_name +'.csv')
trainPrior_dataframe = pd.read_csv(file_path +'/Train'+str(split_idx)+'_' + file_name +'.csv', header = 0)
validPrior_dataframe = pd.read_csv(file_path +'/Valid'+str(split_idx)+'_' + file_name +'.csv', header = 0) 

##### further check the bounds conditions #####
trainPrior_dataframe['Vs_differ'] = trainPrior_dataframe['Vs_max'] - trainPrior_dataframe['Vs_min']
validPrior_dataframe['Vs_differ'] = validPrior_dataframe['Vs_max'] - validPrior_dataframe['Vs_min']

trainPrior_dataframe_real = trainPrior_dataframe.loc[(trainPrior_dataframe['Vs_differ'] >= 0)]
validPrior_dataframe_real = validPrior_dataframe.loc[(validPrior_dataframe['Vs_differ'] >= 0)]

try:
    assert (trainPrior_dataframe_real.shape[0] == trainPrior_dataframe.shape[0])
    assert (validPrior_dataframe_real.shape[0] == validPrior_dataframe.shape[0])
    print(' Bounds check is passed!!!')
except:
    print('train org number: {}; real number: {}'.format(trainPrior_dataframe.shape[0], trainPrior_dataframe_real.shape[0]))
    print('valid org number: {}; real number: {}'.format(validPrior_dataframe.shape[0], validPrior_dataframe_real.shape[0]))
    

print('training file shape: {}-{}'.format(trainPrior_dataframe_real.shape[0], trainPrior_dataframe_real.shape[1]), end = '\n')
seller_set = np.unique(trainPrior_dataframe_real['anon_slr_id']).tolist()
seller_num = len(seller_set) 
print('seller number: {}'.format(seller_num), end = '\n')  

#### further check for bounds conditions ######
trainPrior_dataframe_real2 = trainPrior_dataframe_real.loc[(trainPrior_dataframe_real['Vs_differ'] >= 0)]
validPrior_dataframe_real2 = validPrior_dataframe_real.loc[(validPrior_dataframe_real['Vs_differ'] >= 0)]
assert (trainPrior_dataframe_real2.shape[0] == trainPrior_dataframe_real.shape[0])
assert (validPrior_dataframe_real2.shape[0] == validPrior_dataframe_real.shape[0])

### get the grouped data of train and test ###
gTrain_data, gValid_data = [], []
if os.path.exists(path_data +'Train_g_0.csv'):
    print(path_data +'Train and valid data are being reloaded!', end ='\n')
    for g in range(k):
        g_data_i = pd.read_csv(path_data +'Train_g_'+str(g) +'.csv', header = 0)
        gv_data_i = pd.read_csv(path_data +'Valid_g_'+str(g) +'.csv', header = 0)
        gTrain_data.append(g_data_i)
        gValid_data.append(gv_data_i)
        
        g_data_i_real = g_data_i.loc[(g_data_i['Vs_differ'] >= 0)]
        gv_data_i_real = gv_data_i.loc[(gv_data_i['Vs_differ'] >= 0)]
        assert (g_data_i_real.shape[0] == g_data_i.shape[0])
        assert (gv_data_i_real.shape[0] == gv_data_i.shape[0])
        
else:
    gTrain_data = get_grouped_data(clusterDict, trainPrior_dataframe_real)
    for g in range(len(gTrain_data)):
        gTrain_data[g].to_csv(path_data +'Train_g_'  + str(g) +'.csv', index = False)

    gValid_data = get_grouped_data(clusterDict, validPrior_dataframe_real)
    for g in range(len(gValid_data)):
        gValid_data[g].to_csv(path_data +'Valid_g_'  + str(g) +'.csv', index = False)


path_tuple = (path_res, path_model)
start_time = time.time()
g=2
learning_model_paras_V0(gTrain_data[g], gValid_data[g], alpha, classes, epochs, batch_size, lr, syn_tag, vs_disType, dis_pars, (0, g), path_tuple)
# Parallel(n_jobs = k)(delayed(learning_model_paras_V0)(gTrain_data[g], gValid_data[g], alpha, classes, epochs, batch_size, lr, syn_tag, vs_disType, dis_pars, (0, g), path_tuple) for g in range(0, len(gTrain_data))) #
print("The Multiprocessing Learning Model took {} minutes. ".format((time.time() - start_time) / 60.))
        
    
    
                    
    
   
        
        
        
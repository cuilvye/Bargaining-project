#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is a python file for training BLUE-C model with synthetic bargaining dataset.
 @author: Lvye Cui
"""

import os, sys, math, copy
import argparse, pickle
import time

from numpy.random import seed


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ""

seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json

from utils_dataProcessing import *
from model import *
from utils_inference import  *
from utils_regrouping import *
from utils_clustering_eval import *
from utils_clustering import *

def  training_model_epochVS_V0(dataframe_prior_train, alpha, classes, bt_size, model, optimizer, syn_tag, dis_type, dis_pars, epoch, save_tuple, path_res):
     
     it, g = save_tuple[0], save_tuple[1]             
     batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior_train, bt_size)        
    
     train_loss, train_acc, train_vsMse, train_vsAcc = [], [], [], []
     train_epoch_log = {'batch': [] , 'total_loss': [], 'acc': [], 'vs_mse': [],  'vs_acc': [] }
     for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
        # X_train, Y_train, X_train_prior, X_train_vsID, X_train_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)        
        X_train, Y_train, X_train_prior, X_train_vsID, X_train_Vs = None, None, None, None, None
        if classes == 4:
            X_train, Y_train, X_train_prior, X_train_vsID, X_train_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X_train, Y_train, X_train_prior, X_train_vsID, X_train_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
           X_train, Y_train, X_train_prior, X_train_vsID, X_train_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars) 
        # print(batch_idataFrame[['s0','b1', 's1', 'b2', 's2', 'b3', 's3']])

        Y_train = to_categorical(Y_train, classes)   
        X_train = X_train.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X_train, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y_train)
        xPrior_tensorBatch = tf.convert_to_tensor(X_train_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_train_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_train_vsID, dtype = float)
        
        # infer vs using threads from the same seller and the same item #
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model)
        vs_mseBatch = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch,[-1,1])) 
        # print(vs_mseBatch)
        # print(afffff)     
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
            # x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
            x_exBatch_Tensor = insert_VS_into_X(vs_tensorBatch, x_tensorBatch)
            cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
          
            logits_org = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))      
            logits = tf.clip_by_value(logits_org, 1e-10, 1)
            assert logits_org.get_shape() == logits.get_shape()         
            
            loss_B = cce(y_tensorBatch, logits)      
            acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))             
            
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
        loss_mean = loss_total_real.numpy()
        
        train_acc.append(acc_batch)
        train_vsMse.append(vs_mseBatch) 
        train_vsAcc.append(vs_accBatch) 
        train_loss.append(loss_mean)          
        
        train_epoch_log['batch'].append(batch + 1)
        train_epoch_log['acc'].append(acc_batch)
        train_epoch_log['vs_mse'].append(vs_mseBatch)
        train_epoch_log['vs_acc'].append(vs_accBatch)
        train_epoch_log['total_loss'].append(loss_mean)   
       
        # if (batch + 1)  % 10 == 0 or (batch + 1) == len(batch_dataFrame):
        # print("[INFO] training batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f}, vs_mse: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, vs_mseBatch, vs_accBatch), end = "\n")                       
    
     # pd.DataFrame(train_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_train_epoch_'+ str(epoch) +'.csv', index = False)    
     train_loss_mean = np.mean(train_loss)
     train_acc_mean = np.mean(train_acc)
     train_vsMse_mean = np.mean(train_vsMse)
     train_vsAcc_mean = np.mean(train_vsAcc)
     # print(afff)
     return model, train_loss_mean, train_acc_mean, train_vsMse_mean, train_vsAcc_mean

def learning_model_paras_V0(dataframe_prior_train, dataframe_prior_valid, alpha, classes, epochs, batch_size, lr, syn_tag, vs_disType, dis_pars, save_tuple, path_tuple):
    
    path_res,  path_model = path_tuple[0], path_tuple[1]
    it, g = save_tuple[0], save_tuple[1] 
    
    ###############################################  building or reloading the model #################################   
    model = None    
    if os.path.exists(path_model +'model_iter_'+ str(it) +'_g_' + str(g) + '.json'):
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
    
    # model = build_model(
    #     classes,
    #     input_shape=(9,1),
    #     head_size=16,
    #     num_heads=2,
    #     ff_dim=4,
    #     num_transformer_blocks=2, #4,
    #     mlp_units=[32],
    #     dropout=0,
    #     mlp_dropout=0,
    # )
    
    # model.summary()
    
    optimizer = Adam(learning_rate = lr, amsgrad=True)#
    model.compile(optimizer = optimizer, loss='categorical_crossentropy') 
    # print(afffff)
    ###############################################   start training model #################################  
    start, train_log = 0, None
    if os.path.exists(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_train_epochs_'+ str(epochs) +'.csv'):
        org_train_log = pd.read_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_train_epochs_'+ str(epochs) +'.csv', header = 0) 
        if org_train_log.shape[0] > 0:
            start = int(np.array(org_train_log)[-1, 0])
        train_log = org_train_log.to_dict(orient = "list")
    else: 
        train_log = {'epoch': [] , 
                     'train_lossTotal': [], 'train_acc': [], 'train_vsMse': [],'train_vsAcc':[],
                     'valid_lossTotal': [], 'valid_acc': [], 'valid_vsMse': [],'valid_vsAcc':[]}
    
    print("[INFO] ================ starting training the model from epoch-{} ==================".format(start), end = "\n")  
    for epoch in range(start, epochs):
        sys.stdout.flush()
        # loop over the data in batch size increments
        epochStart = time.time()
        model, loss_train, acc_train, mse_train, vsAcc_train = training_model_epochVS_V0(dataframe_prior_train, alpha, classes, batch_size, model, optimizer, syn_tag, vs_disType, dis_pars, epoch, save_tuple, path_res)       
        elapsed = (time.time() - epochStart) / 60.0
        print("[INFO] one epoch took {:.4} minutes".format(elapsed), end = "\n")
        ## check if needing valid dataset to adjust the learning rate #        
        loss_v, acc_v, mse_v, vsAcc_v  = performance_on_SynDataset_Multirows_V0_alpha_log(dataframe_prior_valid, alpha, classes, batch_size, syn_tag, vs_disType, dis_pars, model, False, path_res, save_tuple)                                                                                    
        train_log['epoch'].append(epoch+1)
        train_log['train_lossTotal'].append(loss_train)
        train_log['train_acc'].append(acc_train)
        train_log['train_vsMse'].append(mse_train)
        train_log['train_vsAcc'].append(vsAcc_train)
        
        train_log['valid_lossTotal'].append(loss_v)
        train_log['valid_acc'].append(acc_v)
        train_log['valid_vsMse'].append(mse_v)
        train_log['valid_vsAcc'].append(vsAcc_v)
        print("[INFO] epoch {}/{}, train total_loss: {:.5f}, train accuracy: {:.5f}, train vs_mse: {:.5f}, train vs_acc: {:.5f}; ".format(epoch+1, epochs, loss_train, acc_train, mse_train, vsAcc_train), end = "\n")            
        print("[INFO] epoch {}/{}, valid total_loss: {:.5f}, valid accuracy: {:.5f}, valid vs_mse: {:.5f}, valid vs_acc: {:.5f};".format(epoch+1, epochs, loss_v, acc_v, mse_v, vsAcc_v), end = "\n")            
        
        pd.DataFrame(train_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_train_epochs_'+ str(epochs) +'.csv', index = False)            
        
        model_json = model.to_json()
        with open(path_model + 'model_iter_'+ str(it) +'_g_' + str(g) + '.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(path_model + 'model_iter_'+ str(it) +'_g_' + str(g) + '.h5')
        
        stopEarly = Callback_EarlyStopping(train_log['valid_lossTotal'], min_delta = 0.01, patience = 30)#30
        if stopEarly:
            print('Callback Earlystopping signal received at epoch: {}/{} '.format(epoch+1, epochs))
            print('Terminating training ')
            break
                                                                                                   

    
# if __name__ == '__main__':
       
print('*************************** Running BLUE-C Exps on Synthesized Dataset ***************************')
args_parser = argparse.ArgumentParser()

args_parser.add_argument('--file_root',  default ='./Datasets/SynthesizedData/', help = 'the root path of dataset', type = str)
args_parser.add_argument('--fold_lambda', default = 'SynData_Uniform', help = 'the dataset name', type = str)
args_parser.add_argument('--classes', default = 6, help = 'the number of action types', type = int)
args_parser.add_argument('--alpha', default = 0.6, help = 'the weight of the first cost function', type = float)
args_parser.add_argument('--k', default = 3, help = 'the number of clusters', type = int)
args_parser.add_argument('--epochs', default = 500, help = 'the training epochs', type = int)
args_parser.add_argument('--batch', default = 64, help = 'the size of training batch ', type = int)
args_parser.add_argument('--lr', default = 0.001, help = 'the learning rate of Adam optimizer ', type = float)
args_parser.add_argument('--vsdisType', default = 'Uniform', help = 'the prior distribution of Vs ', type = str)
args_parser.add_argument('--synTag', default = True, help = 'the data is synthesized or real ', type = bool)
args_parser.add_argument('--disPars', default = {'price_min': 10, 'price_max': 100, 'gap': 4}, help = 'the distribution pars of dataset ', type = dict)
args_parser.add_argument('--split_idx', default = 1, help = 'different data split:train/valid/test', type = int)
args_parser.add_argument('--save_root', default = './RetrainingModels/SynthesizedData', help = 'save path', type = str)

args = args_parser.parse_args()

fold_lambda = args.fold_lambda
file_path = args.file_root
save_root = args.save_root

# hyper parameter #    
classes = args.classes
alpha = args.alpha
k = args.k
epochs = args.epochs
batch_size = args.batch
lr = args.lr  
vs_disType = args.vsdisType
split_idx = args.split_idx

# data pars # 
syn_tag = args.synTag
dis_pars = args.disPars

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
print('Train'+str(split_idx)+'_' + file_name +'.csv')
train_name = 'TrainPrior'+str(split_idx)+'_' + file_name
valid_name = 'ValidPrior'+str(split_idx)+'_' + file_name
filePrior_train = file_path + fold_lambda +'/' + train_name +'.csv'
filePrior_valid = file_path + fold_lambda +'/' + valid_name +'.csv'
  
save_data_tag = save_root + '/' + fold_lambda + '/' + file_name

path_data = save_data_tag +'/Transformer_grouped_data/Split_'+ str(split_idx) +'/'
path_model = save_data_tag + '/RNN_Split_'+ str(split_idx) + '_Ours_Classes_'+ str(classes) + '_alpha_' + str(alpha) +'_models/'
path_res = save_data_tag + '/RNN_Split_'+ str(split_idx) + '_Ours_Classes_'+ str(classes) + '_alpha_' + str(alpha) +'_results/'           

if not os.path.exists(save_data_tag):
    os.makedirs(save_data_tag)
if not os.path.exists(path_data):
    os.makedirs(path_data)
if not os.path.exists(path_model):
    os.makedirs(path_model)
if not os.path.exists(path_res):
    os.makedirs(path_res)

############################## obtain the grouped sellers based on the whole file ###########################
clusterDict = {}
if not os.path.exists(path_data +'True_Clustered_Sellers_dict.pkl'):
    fileAll_dataframe = pd.read_csv(file_path + fold_lambda +'/' + file_name +'.csv', header = 0 )
    seller_set = np.unique(fileAll_dataframe['anon_slr_id']).tolist()
    seller_num = len(seller_set) 
    print('seller number in total: {}'.format(seller_num), end = '\n')  
    groups_truth, true_numSum = get_true_groups(fileAll_dataframe, k)
    # print(true_numSum)
    # print(seller_num)
    # print(k)
    assert(true_numSum == seller_num)
    g_sellers = groups_truth
    
    for i in range(k):
        clusterDict[str(i)] = []
    check_num = 0
    for c in range(k):
        for seller_idx in g_sellers[c]:
            clusterDict[str(c)].append(seller_idx)
        check_num = check_num + len(clusterDict[str(c)])
    assert(check_num == seller_num)
    
    pre_matrix = evaluating_groups_results(g_sellers, groups_truth)
    recall_matrix = evaluating_groups_results_recall(g_sellers, groups_truth)  
    pd.DataFrame(pre_matrix).to_csv(path_data +'true_sellers_precision.csv' , header = False, index = False)              
    pd.DataFrame(recall_matrix).to_csv(path_data +'true_sellers_recall.csv' , header = False, index = False)
        
    with open(path_data +'True_Clustered_Sellers_dict.pkl', 'wb') as f:
        pickle.dump(clusterDict, f)
else:
    print('...........Reloading the grouped result...........')
    with open(path_data +'True_Clustered_Sellers_dict.pkl', 'rb') as f:
        clusterDict = pickle.load(f) 

############################## start training grouped data in a parallel way  ###########################      
trainPrior_dataframe = pd.read_csv(filePrior_train, header = 0)
validPrior_dataframe = pd.read_csv(filePrior_valid, header = 0)
### get the grouped data of train and test ###
gTrain_data, gValid_data = [], []
if os.path.exists(path_data +'Train_g_0.csv'):
    print(path_data +'Train and valid data are being reloaded!', end ='\n')
    for g in range(k):
        g_data_i = pd.read_csv(path_data +'Train_g_'+str(g) +'.csv', header = 0)
        gTrain_data.append(g_data_i)
        gv_data_i = pd.read_csv(path_data +'Valid_g_'+str(g) +'.csv', header = 0)
        gValid_data.append(gv_data_i)
else:
    gTrain_data = get_grouped_data(clusterDict, trainPrior_dataframe)
    for g in range(len(gTrain_data)):
        gTrain_data[g].to_csv(path_data +'Train_g_'  + str(g) +'.csv', index = False)

    gValid_data = get_grouped_data(clusterDict, validPrior_dataframe)
    for g in range(len(gValid_data)):
        gValid_data[g].to_csv(path_data +'Valid_g_'  + str(g) +'.csv', index = False)


path_tuple = (path_res, path_model)
g=0
learning_model_paras_V0(gTrain_data[g], gValid_data[g], alpha, classes, epochs, batch_size, lr, syn_tag, vs_disType, dis_pars, (0, g), path_tuple) #
# start_time = time.time()
# Parallel(n_jobs = k)(delayed(learning_model_paras_V0)(gTrain_data[g], gValid_data[g], alpha, classes, epochs, batch_size, lr, syn_tag, vs_disType, dis_pars, (0, g), path_tuple) for g in range(0, len(gTrain_data))) #
# print("The Multiprocessing Learning Model took {} minutes. ".format((time.time() - start_time) / 60.))

# file_pred = path_data +'iter_0_sellers.txt'
# g_data = []
# if not os.path.exists(file_pred):
#     print('Getting the dataFrame for each seller in the seller set ...', end = '\n')
#     sellersPriorData, sellers = obtain_dataFrame_each_seller(train_dataframe, seller_set)
#     groups_truth, true_numSum = get_true_groups(train_dataframe, k)
#     assert(true_numSum == seller_num)
#     # divding the dataset into seperate groups #
#     model_VGG = VGG16()
#     model_VGG = Model(inputs=model_VGG.inputs, outputs = model_VGG.layers[-2].output)        
#     save_fig = path_data +'imgs'
#     if not os.path.exists(save_fig):
#         os.mkdir(save_fig)
#     obtain_each_seller_image_autoencoderX(sellersPriorData, sellers, classes, save_fig, autoencoder_path)                             
#     data = []
#     for i in sellers:
#         img = load_img(save_fig + '/seller-'+str(i) +'.png', target_size = (224, 224))
#         img = np.array(img)
#         img_reshaped = img.reshape(1, 224, 224, 3)
#         imgx = preprocess_input(img_reshaped)
#         features = model_VGG.predict(imgx, use_multiprocessing= True)
#         data.append(features)                   
#     data = np.array(data)  
#     reshaped_data = data.reshape(data.shape[0], -1)#(seller_num, 4096)
#     print(reshaped_data.shape)
#     kmeans = KMeans(init = 'random', n_clusters = k, random_state = 0)
#     clusters = kmeans.fit_predict(reshaped_data)
#     assert(len(list(clusters)) == seller_num)
    
#     g_data, g_sellers = Reobtain_eachGroup_dataFrame(list(clusters), sellers, sellersPriorData, train_dataframe, k)               
#     save_clustered_data(g_data, path_data, 0, g_sellers, path_data +'iter_0_sellers.txt')
#     pre_matrix = evaluating_groups_results(g_sellers, groups_truth)
#     recall_matrix = evaluating_groups_results_recall(g_sellers, groups_truth)  
#     pd.DataFrame(pre_matrix).to_csv(path_data +'Result_iter_0_precision.csv' , header = False, index = False)              
#     pd.DataFrame(recall_matrix).to_csv(path_data +'Result_iter_0_recall.csv' , header = False, index = False)

# else:  
#     g_sellers, pred_numSum = read_pred_resultsFile(file_pred) 
#     assert(pred_numSum == seller_num)
#     for g in range(k):
#         g_data_i = path_data +'data_iter_0_g_'+str(g) +'.csv'
#         if os.path.exists(g_data_i):
#             print(path_data +'data_iter_0_g_'+str(g) +'.csv is being reloaded!', end ='\n')
#             g_data_i = pd.read_csv(g_data_i, header = 0)
#             g_data.append(g_data_i)
#         else: 
#             print(path_data +'data_iter_0_g_'+str(g) +'.csv NOT EXISTS!', end ='\n')
#             print('THIS MAY DUE TO GROUP REDUCTION OF LAST REGROUPING, PLEASE CHECK FOR THIS!')  

# start_time = time.time()
# Parallel(n_jobs = k)(delayed(learning_model_paras_V0)(g_data[g], valid_dataframe, alpha, classes, epochs, batch_size, lr, syn_tag, vs_disType, dis_pars, (0, g), path_tuple) for g in range(len(g_data)))
# print("The Multiprocessing Learning Model took {} minutes. ".format((time.time() - start_time) / 60.))




    
    
    

    
   
    
    
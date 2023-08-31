#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The K-Loss clustering algorithm on real bargaining dataset.
@author: Lvye

"""

import argparse, os
import pandas as pd
import numpy as np
import time, random, math

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from joblib import Parallel, delayed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from model import build_rnn_GRUModel_onlyX
from utils_dataProcessing import round_training_XY_prior_RealData
from utils_dataProcessing import round_training_XY_prior_RealData_Classes5
from utils_regrouping import compute_min_index


def Convert_to_OneRecord_RealData(dataframe, classes):   
    
    X_train, Y_train, sellers_id = np.array([]), [], []       
    rounds_col = list(dataframe).index('rounds_total')
    # mask_value = 0.
    for i in tqdm(range(dataframe.shape[0])):        
        seller_i = np.array(dataframe.loc[i, ['anon_slr_id']])[0]        
        if dataframe.iloc[i,rounds_col] == 1:
            x_cols, y_col = ['s0','b1'], ['s1']
            x, y = None, None
            if classes == 3:
                x, y = round_training_XY_prior_RealData(i, dataframe, x_cols, y_col)           
            elif classes == 5:                    
                x, y = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols, y_col)                      
            x = x + [0,0,0,0]    
            if len(X_train) == 0:
                X_train = np.array(x)
            else:
                X_train = np.vstack((X_train, np.array(x)))
            Y_train.append(y)
            sellers_id.append(seller_i)
            
        elif dataframe.iloc[i, rounds_col] == 2:
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = None, None
            if classes == 3:
                x1, y1 = round_training_XY_prior_RealData(i, dataframe, x_cols1, y_col1)           
            elif classes == 5:                    
                x1, y1 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols1, y_col1)   
            # x1, y1 = round_training_x_y_prior_Class6(i, dataframe, x_cols1, y_col1)           
            x1 = x1 + [0,0,0,0]
            if len(X_train) == 0:
                X_train = np.array(x1)
            else:
                X_train = np.vstack((X_train, np.array(x1)))
            Y_train.append(y1)
            sellers_id.append(seller_i)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = None, None
            if classes == 3:
                x2, y2 = round_training_XY_prior_RealData(i, dataframe, x_cols2, y_col2)           
            elif classes == 5:                    
                x2, y2 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols2, y_col2)  
            # x2, y2 = round_training_x_y_prior_Class6(i, dataframe, x_cols2, y_col2)           
            x2 = x2 + [0, 0]
            X_train = np.vstack((X_train, np.array(x2)))
            Y_train.append(y2)
            sellers_id.append(seller_i)
            
        elif dataframe.iloc[i,rounds_col] == 3 :
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = None, None
            if classes == 3:
                x1, y1 = round_training_XY_prior_RealData(i, dataframe, x_cols1, y_col1)           
            elif classes == 5:                    
                x1, y1 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols1, y_col1)         
            x1 = x1 + [0,0,0,0]
            if len(X_train) == 0:
                X_train = np.array(x1)
            else:
                X_train = np.vstack((X_train, np.array(x1)))
            Y_train.append(y1)
            sellers_id.append(seller_i)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = None, None
            if classes == 3:
                x2, y2 = round_training_XY_prior_RealData(i, dataframe, x_cols2, y_col2)           
            elif classes == 5:                    
                x2, y2 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols2, y_col2)          
            x2 = x2 + [0, 0]
            X_train = np.vstack((X_train, np.array(x2)))
            Y_train.append(y2)
            sellers_id.append(seller_i)

            x_cols3, y_col3 = ['s0','b1', 's1', 'b2', 's2', 'b3'], ['s3']
            x3, y3 = None, None
            if classes == 3:
                x3, y3 = round_training_XY_prior_RealData(i, dataframe, x_cols3, y_col3)           
            elif classes == 5:                    
                x3, y3 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols3, y_col3) 
            # x3, y3 = round_training_x_y_prior_Class6(i, dataframe, x_cols3, y_col3)           
            X_train = np.vstack((X_train, np.array(x3)))
            Y_train.append(y3)
            sellers_id.append(seller_i)
    
    return np.array(X_train), np.array(Y_train), np.array(sellers_id)

def Randomly_grouping_sellers(sellers, k):
    
    seller_set = np.unique(sellers).tolist()
    seller_num = len(seller_set)
    random.shuffle(seller_set)
    each_group_size = math.ceil(seller_num / k) # seller_num // k is not OK   
    grouped_sellers = [seller_set[each_group_size * i: each_group_size * (i+1)] for i in range(k)]
        
    return grouped_sellers 

def extract_train_valid_Data_on_regroupedResults(g_sellers, valid_X, valid_Y, valid_sellers):
    gv_xdata, gv_ydata = [], []
    check_num = 0
    for g in g_sellers:
        X_g, Y_g = np.array([]), np.array([])
        for sel in g:
            if len(X_g) == 0:
                X_g = valid_X[np.where(valid_sellers == sel)[0].tolist()]
            else:
                X_g = np.vstack((X_g, valid_X[np.where(valid_sellers == sel)[0].tolist()]))
            if len(Y_g) == 0:
                Y_g = valid_Y[np.where(valid_sellers == sel)[0].tolist()]
            else:
                Y_g = np.vstack((Y_g, valid_Y[np.where(valid_sellers == sel)[0].tolist()]))
        gv_xdata.append(X_g)
        gv_ydata.append(Y_g)
        check_num = check_num + X_g.shape[0]
    assert(check_num == len(valid_sellers))
    assert(check_num == valid_X.shape[0])   
    
    return gv_xdata, gv_ydata
    
def train_model_on_eachGroup(classes, x_train, y_train, x_val, y_val, lr, path_model, it, g, optimizer):
    
    x_train = x_train.reshape(x_train.shape[0], 3, -1)
    x_val = x_val.reshape(x_val.shape[0], 3, -1)    
    
    model = None
    if os.path.exists(path_model + '/model_it_'+str(it)+'_g_'+ str(g) + '.json'):
        print('reloading the pre-trained model_it_{}_g_{}'.format(it, g))
        json_file = open(path_model + '/model_it_'+str(it)+'_g_'+ str(g) + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model  = model_from_json(loaded_model_json) 
        model.load_weights(path_model + '/model_it_'+str(it)+'_g_'+ str(g) + '.h5')
    else:    
        model = build_rnn_GRUModel_onlyX(classes)             
    model.compile(optimizer = optimizer, loss='categorical_crossentropy')         
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 90, restore_best_weights = True)
    history = model.fit(x_train, y_train, 
                        batch_size = 256, 
                        epochs = 2000, 
                        validation_data = (x_val, y_val), 
                        validation_freq = 1, 
                        callbacks = [callback])
    
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss'])      
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.savefig(path_model + '/model_it_'+str(it)+'_g_'+ str(g) + '.png')
    # plt.show() 
    
    model_json = model.to_json()
    with open(path_model + '/model_it_'+str(it)+'_g_'+ str(g) + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(path_model + '/model_it_'+str(it)+'_g_'+ str(g) + '.h5')  

def Regrouping_seller(sellers, X, Y, path_model, it, k, optimizer):
    modelSet = []
    for g in range(k):
        model_name = path_model + '/model_it_'+ str(it) +'_g_' + str(g) + '.json'
        model_weights = path_model + '/model_it_'+ str(it) +'_g_' + str(g) + '.h5'
        if os.path.exists(model_name):
            json_file = open(model_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            Loaded_model_i  = model_from_json(loaded_model_json) 
            Loaded_model_i.load_weights(model_weights)
            
            modelSet.append(Loaded_model_i)
        else:
            print(model_name + ' CAN NOT BE FOUND!!!')
            print('THIS MAY DUE TO GROUP REDUCTION OF LAST REGROUPING, PLEASE CHECK FOR THIS!')   
    
    g_sellers = []
    for i in range(k):
        g_sellers.append([])
    
    sellers_list = np.unique(sellers).tolist()
    for seller in sellers_list:
        seller_X  = X[np.where(sellers == seller)[0].tolist()]
        seller_Y = Y[np.where(sellers == seller)[0].tolist()]
        seller_X = seller_X.reshape(seller_X.shape[0], 3, -1)
        seller_loss = []
        for model in modelSet:    
            model.compile(optimizer = optimizer, loss='categorical_crossentropy')
            loss_m = model.evaluate(seller_X, seller_Y, batch_size=1, verbose=0)
            seller_loss.append(loss_m)
        idx_g, _ = compute_min_index(seller_loss)       
        g_sellers[idx_g].append(seller)
    
    ### remove_nullData_group ###
    i = 0
    while i < len(g_sellers):
       if len(g_sellers[i]) == 0:
           del g_sellers[i]
           i = i
       else:
           i = i + 1 
    
    return g_sellers

def Regrouping_this_seller(sellers, seller, X, Y, path_model, it, k, optimizer):
    
    modelSet = []
    for g in range(k):
        model_name = path_model + '/model_it_'+ str(it) +'_g_' + str(g) + '.json'
        model_weights = path_model + '/model_it_'+ str(it) +'_g_' + str(g) + '.h5'
        if os.path.exists(model_name):
            json_file = open(model_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            Loaded_model_i  = model_from_json(loaded_model_json) 
            Loaded_model_i.load_weights(model_weights)
            
            modelSet.append(Loaded_model_i)
        else:
            print(model_name + ' CAN NOT BE FOUND!!!')
            print('THIS MAY DUE TO GROUP REDUCTION OF LAST REGROUPING, PLEASE CHECK FOR THIS!')   
        
    seller_X  = X[np.where(sellers == seller)[0].tolist()]
    seller_Y = Y[np.where(sellers == seller)[0].tolist()]
    seller_X = seller_X.reshape(seller_X.shape[0], 3, -1)
    seller_loss = []
    for model in modelSet:    
        model.compile(optimizer = optimizer, loss='categorical_crossentropy')
        loss_m = model.evaluate(seller_X, seller_Y, batch_size=1, verbose=0)
        seller_loss.append(loss_m)
    idx_g, _ = compute_min_index(seller_loss)       
   
    return idx_g


def Save_regrouped_results(g_sellers, it, k, path_res):
    
    clusterDict = {}
    for i in range(k):
        clusterDict[str(i)] = []
  
    check_num = 0
    for c in range(k):
        for seller_idx in g_sellers[c]:
            clusterDict[str(c)].append(seller_idx)
        check_num = check_num + len(clusterDict[str(c)])
    assert(check_num == seller_num)
        
    with open(path_res +'/IterPred_' + str(it) +'_Clustered_Sellers_dict.pkl', 'wb') as f:
        pickle.dump(clusterDict, f)
    
    return clusterDict

def Check_if_DataExisted(path_data, pattern_iter):    
    start = -1
    for it in range(pattern_iter):
        if os.path.exists(path_data + '/data_iter_'+ str(it) +'_train_X_g_0.csv'):
            start = it
            continue  
        else:
            break
                   
    return start



args_parser = argparse.ArgumentParser()

args_parser.add_argument('--file_path',  default ='./Datasets/RealDataset', help = 'the root path of dataset', type = str)
args_parser.add_argument('--file_name', default = 'vs_inference_eBay_Selection_Final_Prior_Corr_FS_sel_differ_0_50_Sel_VsDiffer_2_VsNum_45', help = 'the dataset name', type = str)
args_parser.add_argument('--classes', default = 5, help = 'the number of action type', type = int)
args_parser.add_argument('--k', default = 3, help = 'the number of clusters', type = int)
args_parser.add_argument('--lr', default = 0.001, help = 'the learning rate in Adam optimizer', type = float)
args_parser.add_argument('--pattern_iter', default = 50, help = 'the maximum iteration times', type = int)
args_parser.add_argument('--save_root', default = './RetrainingModels/RealDataset/Grouping_Exps', help = 'save path', type = str)
args = args_parser.parse_args()

file_path = args.file_path
file_name = args.file_name
classes = args.classes
k = args.k 
lr = args.lr
pattern_iter = args.pattern_iter
save_root = args.save_root

file_root = file_path + '/' + file_name + '.csv'  
file = pd.read_csv(file_root, header = 0)
seller_set = np.unique(file['anon_slr_id']).tolist()
seller_num = len(seller_set) 
print('seller number: {}'.format(seller_num), end = '\n')  
 
save_data_tag = save_root + '/' + file_name + '_k_' + str(k)
path_model = save_data_tag +'/ActionModel'  

if not os.path.exists(save_root):
    os.makedirs(save_root)
if not os.path.exists(save_data_tag):
    os.makedirs(save_data_tag)
if not os.path.exists(path_model):
    os.makedirs(path_model)

path_data = path_model + '/data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
    
path_res = save_data_tag +'/Classes_'+ str(classes)
if not os.path.exists(path_res):
    os.mkdir(path_res)  

start = Check_if_DataExisted(path_data, pattern_iter)

g_xdata, g_ydata, gv_xdata, gv_ydata = [], [], [], []
X, Y, Sellers_id, g_sellers = None, None, None, []
train_X, train_Y, train_sellers = None, None, None
valid_X, valid_Y, valid_sellers = None, None, None
if start == -1: 
    ################################## y model training with X and Y ##################################        
    X, Y, Sellers_id = Convert_to_OneRecord_RealData(file, classes)
    X = X.astype(np.float64)
    Y = to_categorical(Y, classes)
    
    ######### dividing into train data and valid data #########
    idx_set = list(range(X.shape[0]))
    random.shuffle(idx_set) 
    trainRows_idx = idx_set[0 : math.ceil(len(idx_set) * 0.8)]
    validRows_idx = idx_set[math.ceil(len(idx_set) * 0.8) : len(idx_set)]
    train_X = X[trainRows_idx, :]
    train_Y = Y[trainRows_idx, :]
    train_sellers = Sellers_id[trainRows_idx]
    
    valid_X = X[validRows_idx, :]
    valid_Y = Y[validRows_idx, :]
    valid_sellers = Sellers_id[validRows_idx]    
    assert((len(trainRows_idx) + len(validRows_idx)) == len(idx_set))
    
    ######### firstly regrouping the sellers #########
    g_sellers = Randomly_grouping_sellers(Sellers_id, k) 
    g_xdata, g_ydata = extract_train_valid_Data_on_regroupedResults(g_sellers, train_X, train_Y, train_sellers) 
    gv_xdata, gv_ydata = extract_train_valid_Data_on_regroupedResults(g_sellers, valid_X, valid_Y, valid_sellers)
    
    start =  start + 1  # 0
    res_dict = Save_regrouped_results(g_sellers, start, k, path_res)
   
    ###############  save the data generated #########################
    ###############  save the data generated #########################
    pd.DataFrame(X).to_csv(path_data + '/X.csv', header = None, index = None)
    pd.DataFrame(Y).to_csv(path_data + '/Y.csv', header = None, index = None)
    pd.DataFrame(Sellers_id).to_csv(path_data + '/Sellers.csv', header = None, index = None)
    
    pd.DataFrame(train_X).to_csv(path_data + '/train_X.csv', header = None, index = None)
    pd.DataFrame(train_Y).to_csv(path_data + '/train_Y.csv', header = None, index = None)
    pd.DataFrame(train_sellers).to_csv(path_data + '/train_sellers.csv', header = None, index = None)
    
    pd.DataFrame(valid_X).to_csv(path_data + '/valid_X.csv', header = None, index = None)
    pd.DataFrame(valid_Y).to_csv(path_data + '/valid_Y.csv', header = None, index = None)
    pd.DataFrame(valid_sellers).to_csv(path_data + '/valid_sellers.csv', header = None, index = None)
    for g in range(k):
        pd.DataFrame(g_xdata[g]).to_csv(path_data + '/data_iter_'+ str(start) +'_train_X_g_'+ str(g) +'.csv', header = None, index = None)
        pd.DataFrame(g_ydata[g]).to_csv(path_data + '/data_iter_'+ str(start) +'_train_Y_g_'+ str(g) +'.csv', header = None, index = None)
        pd.DataFrame(gv_xdata[g]).to_csv(path_data + '/data_iter_'+ str(start) +'_valid_X_g_'+ str(g) +'.csv', header = None, index = None)
        pd.DataFrame(gv_ydata[g]).to_csv(path_data + '/data_iter_'+ str(start) +'_valid_Y_g_'+ str(g) +'.csv', header = None, index = None)

else:
    X = np.array(pd.read_csv(path_data +'/X.csv', header = None))
    Y = np.array(pd.read_csv(path_data +'/Y.csv', header = None))
    Sellers_id = np.array(pd.read_csv(path_data +'/Sellers.csv', header = None))
    
    train_X = np.array(pd.read_csv(path_data +'/train_X.csv', header = None))
    train_Y = np.array(pd.read_csv(path_data +'/train_Y.csv', header = None))
    train_sellers = np.array(pd.read_csv(path_data +'/train_sellers.csv', header = None))
    
    valid_X = np.array(pd.read_csv(path_data +'/valid_X.csv', header = None))
    valid_Y = np.array(pd.read_csv(path_data +'/valid_Y.csv', header = None))
    valid_sellers = np.array(pd.read_csv(path_data +'/valid_sellers.csv', header = None))
    
    with open(path_res +'/IterPred_' + str(start) +'_Clustered_Sellers_dict.pkl', 'rb') as f:
        clusterDict = pickle.load(f) 
    for i in range(k):
        g_sellers.append([])
    for key in list(clusterDict.keys()): 
        g_sellers[int(key)] = clusterDict[key]
    
    for g in range(k):
        g_data_i = path_data +'/data_iter_'+ str(start) +'_train_X_g_'+str(g) +'.csv'
        if os.path.exists(g_data_i):
            print(path_data +'/data_iter_'+ str(start) +'_train_X_g_'+str(g) +'.csv is being reloaded!', end ='\n')            
            g_xdata.append(np.array(pd.read_csv(path_data +'/data_iter_'+ str(start) +'_train_X_g_'+str(g) +'.csv', header = None)))
            g_ydata.append(np.array(pd.read_csv(path_data +'/data_iter_'+ str(start) +'_train_Y_g_'+str(g) +'.csv', header = None)))
            gv_xdata.append(np.array(pd.read_csv(path_data +'/data_iter_'+ str(start) +'_valid_X_g_'+str(g) +'.csv', header = None)))
            gv_ydata.append(np.array(pd.read_csv(path_data +'/data_iter_'+ str(start) +'_valid_Y_g_'+str(g) +'.csv', header = None)))
        else: 
            print(path_data +'/data_iter_'+ str(start) +'_train_X_g_'+str(g) +'.csv NOT EXISTS!', end ='\n')
            print('THIS MAY DUE TO GROUP REDUCTION OF LAST REGROUPING, PLEASE CHECK FOR THIS!')  

optimizer = Adam(learning_rate = lr, amsgrad=True)
g_sellers_pre = g_sellers
sellers_list = np.unique(Sellers_id).tolist()
for it in range(start, pattern_iter):  
    # g_data may have [] e.g. k reduces to 4 #    
    start_time = time.time() 
    Parallel(n_jobs = len(g_xdata))(delayed(train_model_on_eachGroup)(classes, g_xdata[g], g_ydata[g], gv_xdata[g], gv_ydata[g], lr, path_model, it, g, optimizer) for g in range(len(g_xdata))) 
    print("The Multiprocessing Learning Model took {} minutes. ".format((time.time() - start_time) / 60.))
    
    ######### the final regrouping result for all sellers #########
    print('Now we are regrouping the sellers ...', end = '\n')  
    # g_sellers = Regrouping_seller(Sellers_id, X, Y, path_model, it, k, optimizer)
    ## the parallel version for all sellers ##
    # parsLists = [(sellers_list, seller, X, Y, path_model, it, k, optimizer) for seller in sellers_list]            
    start_time = time.time()
    results = Parallel(n_jobs = 8)(delayed(Regrouping_this_seller)(Sellers_id, seller, X, Y, path_model, it, k, optimizer) for seller in sellers_list)
    # with multiprocessing.Pool(processes = 16) as pool:
    #        start_time = time.time()
    #        results = pool.starmap(Regrouping_this_seller, parsLists)
    # pool.close()
    # pool.join()
    print("The multiprocessing regrouping took {} minutes. ".format((time.time() - start_time) / 60.))    
    g_sellers = []
    for i in range(k):
        g_sellers.append([])   
    for i, idx_g in enumerate(results):
         g_sellers[idx_g].append(sellers_list[i])                   
    check_num = 0
    for i, group in enumerate(g_sellers):
         check_num = check_num + len(group)
    assert(check_num == len(sellers_list))
    print("The sellers regrouping finished ! ", end = '\n')
    ### remove_nullData_group ###
    i = 0
    while i < len(g_sellers):
        if len(g_sellers[i]) == 0:
            del g_sellers[i]
            i = i
        else:
            i = i + 1            
    
    res_dict_it = Save_regrouped_results(g_sellers, it + 1, len(g_sellers), path_res)
 
    assert len(g_sellers_pre) == len(g_sellers)
    terminating_flag = True
    g_sellers_pre_set = [set(ele) for ele in g_sellers_pre]
    g_sellers_set = [set(ele) for ele in g_sellers]
    for i in range(len(g_sellers_set)):
        if len(g_sellers_set[i].intersection(g_sellers_pre_set[i]))/len(g_sellers_set[i]) < 0.98:
            terminating_flag = False       
    if terminating_flag:
        break
    
    g_xdata, g_ydata = extract_train_valid_Data_on_regroupedResults(g_sellers, train_X, train_Y, train_sellers)        
    gv_xdata, gv_ydata = extract_train_valid_Data_on_regroupedResults(g_sellers, valid_X, valid_Y, valid_sellers)           
    g_sellers_pre = g_sellers
    for g in range(k):
        pd.DataFrame(g_xdata[g]).to_csv(path_data + '/data_iter_'+ str(it + 1) +'_train_X_g_'+ str(g) +'.csv', header = None, index = None)
        pd.DataFrame(g_ydata[g]).to_csv(path_data + '/data_iter_'+ str(it + 1) +'_train_Y_g_'+ str(g) +'.csv', header = None, index = None)
        pd.DataFrame(gv_xdata[g]).to_csv(path_data + '/data_iter_'+ str(it + 1) +'_valid_X_g_'+ str(g) +'.csv', header = None, index = None)
        pd.DataFrame(gv_ydata[g]).to_csv(path_data + '/data_iter_'+ str(it + 1) +'_valid_Y_g_'+ str(g) +'.csv', header = None, index = None)

print('After iterating {} times, it is terminated... '.format(it))





 



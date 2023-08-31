#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Lvye

"""

import numpy as np
import pandas as pd
import random
import math
import copy
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action='ignore', category = FutureWarning)# ignore all future warnings

def divide_train_test_data(dataframe, ratio):
    ## rows with one (seller, item) should all be divided into train(test) and not be separated!
    sel_item_pair = []
    
    item_col_idx = list(dataframe).index('anon_item_id') # list(dataframe): a list of columns name
    slr_col_idx = list(dataframe).index('anon_slr_id')

    for i in range(dataframe.shape[0]):
        item = dataframe.iloc[i, item_col_idx]
        sel = dataframe.iloc[i, slr_col_idx]
        sel_item_pair.append((sel, item))

    sel_item_pair_unique = set(sel_item_pair)
       
    train_dataframe = pd.DataFrame(columns = dataframe.columns.tolist())
    valid_dataframe = pd.DataFrame(columns = dataframe.columns.tolist())
    test_dataframe = pd.DataFrame(columns = dataframe.columns.tolist())
    
    num = len(sel_item_pair_unique)
    idx_set = list(range(num))
    random.shuffle(idx_set) 
    
    train_pair_idx = idx_set[0:math.ceil(num*ratio)]
    valid_pair_idx = idx_set[math.ceil(num*ratio):math.ceil(num*(ratio+0.1))]
    test_pair_idx = idx_set[math.ceil(num*(ratio+0.1)) : len(idx_set)]
    
    sel_item_pair_unique = list(sel_item_pair_unique)
    train_pair, valid_pair, test_pair = [], [], []
    for i in range(len(sel_item_pair_unique)):
        if i in train_pair_idx:
            train_pair.append(sel_item_pair_unique[i])
        elif i in test_pair_idx:
            test_pair.append(sel_item_pair_unique[i])
        elif i in valid_pair_idx:
            valid_pair.append(sel_item_pair_unique[i])
        else:
            print('sth wrong in divide_train_test_data module !')
            assert(i == 0)   
    for sel, item in train_pair:
        data_sel = dataframe.loc[(dataframe['anon_slr_id'] == sel)&(dataframe['anon_item_id'] == item)]
        train_dataframe = train_dataframe.append(data_sel, ignore_index = True)
    for sel, item in valid_pair:
        data_sel = dataframe.loc[(dataframe['anon_slr_id'] == sel)&(dataframe['anon_item_id'] == item)]
        valid_dataframe = valid_dataframe.append(data_sel, ignore_index = True)
    for sel, item in test_pair:
        data_sel = dataframe.loc[(dataframe['anon_slr_id'] == sel)&(dataframe['anon_item_id'] == item)]
        test_dataframe = test_dataframe.append(data_sel, ignore_index = True)
    assert((train_dataframe.shape[0] + valid_dataframe.shape[0] + test_dataframe.shape[0]) == dataframe.shape[0])
    
    return train_dataframe, valid_dataframe, test_dataframe

def extract_batchData_sameSellerItem(dataframe, bt_size):
    sel_item_pair = []
    
    item_col_idx = list(dataframe).index('anon_item_id') # list(dataframe): a list of columns name
    slr_col_idx = list(dataframe).index('anon_slr_id')

    for i in range(dataframe.shape[0]):
        item = dataframe.iloc[i, item_col_idx]
        sel = dataframe.iloc[i, slr_col_idx]
        sel_item_pair.append((sel, item))

    sel_item_pair_unique = set(sel_item_pair)
      
    num = len(sel_item_pair_unique)
    idx_set = list(range(num))
    random.shuffle(idx_set) 
    
    assert(len(np.unique(np.array(dataframe['Vs_idx']))) == num)
    
    
    numUpdates = math.ceil(num / bt_size) # int: small one? should use math.ceil???
    
    batch_pairIdx, batch_dataFrame = [], []
    sel_item_pair_unique = list(sel_item_pair_unique)
    for batch in range(0, numUpdates):
        start  = batch * bt_size
        end  =  start + bt_size
        
        batch_i = idx_set[start:end]
        
        batch_i_selItem_pair = []
        for idx in batch_i:
            batch_i_selItem_pair.append(sel_item_pair_unique[idx])
        
        batch_i_dataFrame = pd.DataFrame(columns = dataframe.columns.tolist())
        for sel, item in batch_i_selItem_pair:
            data_sel = dataframe.loc[(dataframe['anon_slr_id'] == sel)&(dataframe['anon_item_id'] == item)]
            batch_i_dataFrame = batch_i_dataFrame.append(data_sel, ignore_index = True)
            # batch_i_dataFrame = batch_i_dataFrame.concat(data_sel, ignore_index = True)
        batch_pairIdx.append(batch_i_selItem_pair)
        batch_dataFrame.append(batch_i_dataFrame)
    
    check_num = 0
    for batch in batch_pairIdx:
        check_num = check_num + len(batch)
    
    assert(check_num == num)   
    
    return batch_dataFrame, batch_pairIdx
    

def find_b_newest(dataframe):
    b1_col = list(dataframe).index('b1')
    b2_col = list(dataframe).index('b2')
    b3_col = list(dataframe).index('b3')
    b_newest = 0
    for j in [b1_col, b2_col, b3_col]:
        if np.isnan(dataframe.iloc[0,j]) == False:
            b_newest = j
    return b_newest
    

def prior_range_computation_one_row(dataframe):
    # dataframe has only one row #
    s0_col = list(dataframe).index('s0')
    s1_col = list(dataframe).index('s1')
    s2_col = list(dataframe).index('s2')
    s3_col = list(dataframe).index('s3')
    
    val_min, val_max = 0, 0
    # get prior based on buyer's newest price offer #
    b_newest_col = find_b_newest(dataframe)
    if dataframe.iloc[0, b_newest_col+1] == 'A':
        val_max = dataframe.iloc[0, b_newest_col]
    elif dataframe.iloc[0, b_newest_col+1] !='A': # before is !='A' and dataframe.iloc[0,rds_col] == 3: error!
        val_min = dataframe.iloc[0, b_newest_col] # should change to =='D': THIS IS ERROR!!!!
        # since  b_newest_col has already indicated that its next col is  definitely a final action
        # 83556298	10503499	1	68	60	A
        # 83556298	10503499	1	68	43	D       
    # get prior based on seller's price offer #
    
    seller_p = []
    for j in [s0_col, s1_col, s2_col, s3_col]:
        val_j = dataframe.iloc[0, j]    
        try:
            val_j = float(val_j)
            if not np.isnan(val_j):
                seller_p.append(val_j)
        except ValueError:
            val_j = val_j
        # if (not isinstance(val_j, str)) or (isinstance(val_j, str) and str.isdigit(val_j)):
        #     if isinstance(val_j, str) and str.isdigit(val_j):
        #         val_j = int(val_j)
        #     if not np.isnan(val_j):
        #         seller_p.append(val_j) 

    # WHY WRONG ABOUT THIS COMPUTATION
    if val_max > 0:
        val_max = min(val_max, min(seller_p))
    else:
        val_max = min(seller_p)
    
    flag_add_val_max = 0
    # e.g. 77853256	9936051	3	104.99	95	104.99	100	104.99	110	D	110(val_min)	104.99(val_max)
    if val_max < val_min:
        print('strange bargaining row has been found !')
        val_max = 2 * val_min 
        flag_add_val_max = 1
        
    return val_min, val_max, flag_add_val_max

# def check_priorRange_of_synDataset(file):
#     vs_col = list(file).index('vs')
#     results = []
#     for d in range(file.shape[0]): 
#         data_d = pd.DataFrame([file.iloc[d].to_dict()]) #? check for its correctness!
#         # index d starting from 0 is fine for data_sel # wich is different from  index_set = data_sel.index.tolist()
#         val_min, val_max, flag_add_max = prior_range_computation_one_row(data_d)
#         if file.iloc[d, vs_col] <= val_max and  file.iloc[d, vs_col] >= val_min:
#             results.append(0)
#         else:
#             results.append(1)    
#     return results

def compute_multi_buyers_flag_add_max(flag_add_set, val_max_set, val_max):
    flag_add = 0
    if 1 in flag_add_set:
        idx_f = np.where(np.array(flag_add_set) == 1)[0].tolist()
        if len(idx_f) > 0:
            val_max_f = []
            val_max_t = []
            for v in range(len(val_max_set)):
               if v in idx_f:
                   val_max_f.append(val_max_set[v])
               else:
                   val_max_t.append(val_max_set[v])
            if val_max in val_max_f and val_max not in val_max_t: 
                flag_add = 1
    return flag_add

def Compute_prior_range_data(dataframe):
    # look for the (sel,item) set  which has more than one buyer  #
    has_multi_buyers, has_single_buyer = [], [] # has_single_buyer for check the correctness of 'Vs_idx' column
    item_col_idx = list(dataframe).index('anon_item_id')
    slr_col_idx = list(dataframe).index('anon_slr_id')
    for i in range(dataframe.shape[0]):
        item = dataframe.iloc[i, item_col_idx]
        sel = dataframe.iloc[i, slr_col_idx]
        data_sel = dataframe.loc[(dataframe['anon_slr_id'] == sel)&(dataframe['anon_item_id'] == item)]
        if data_sel.shape[0] > 1:
            has_multi_buyers.append((sel, item))
        else:
            has_single_buyer.append((sel, item))
    has_multi_buyers = list(set(has_multi_buyers))
    assert(len(has_single_buyer) == len(list(set(has_single_buyer))))
    
    # compute the prior range set for each row of dataframe
    dataframe_prior = copy.deepcopy(dataframe)
    dataframe_prior['Vs_min'] = np.zeros((dataframe.shape[0],1))
    dataframe_prior['Vs_max'] = np.zeros((dataframe.shape[0],1))
    dataframe_prior['flag_add_Vsmax'] = np.zeros((dataframe.shape[0],1))
    dataframe_prior['Vs_idx'] = np.zeros((dataframe.shape[0],1))
    
    vs_min_col = list(dataframe_prior).index('Vs_min')
    vs_max_col = list(dataframe_prior).index('Vs_max')
    flg_add_col = list(dataframe_prior).index('flag_add_Vsmax')   
    flg_vs_col = list(dataframe_prior).index('Vs_idx')
    
    j_vs, looked_sel_item = 0, []    
    for i in tqdm(range(dataframe_prior.shape[0])):
        item = dataframe_prior.iloc[i,item_col_idx]
        sel = dataframe_prior.iloc[i,slr_col_idx]
        if (sel, item) not in looked_sel_item:
            looked_sel_item.append((sel, item))
            if (sel, item) in has_multi_buyers:
                data_sel = dataframe_prior.loc[(dataframe_prior['anon_slr_id'] == sel)&(dataframe_prior['anon_item_id'] == item)]
                val_min_set, val_max_set, flag_add_set = [], [], []
                for d in range(data_sel.shape[0]): 
                    data_d = pd.DataFrame([data_sel.iloc[d].to_dict()]) #? check for its correctness!
                    # index d starting from 0 is fine for data_sel # wich is different from  index_set = data_sel.index.tolist()
                    val_min, val_max, flag_add_max = prior_range_computation_one_row(data_d)
                    val_min_set.append(val_min)
                    val_max_set.append(val_max)
                    flag_add_set.append(flag_add_max)
                
                val_min, val_max = max(val_min_set), min(val_max_set)               
                flag_add = compute_multi_buyers_flag_add_max(flag_add_set, val_max_set, val_max)         
                # write prior range into these rows  #
                index_set = data_sel.index.tolist() # index_set is the indexes of selected data in original file dataframe_prior
                for row_idx in index_set:
                    dataframe_prior.iloc[row_idx,vs_min_col] = val_min
                    dataframe_prior.iloc[row_idx, vs_max_col] = val_max
                    dataframe_prior.iloc[row_idx, flg_add_col] = flag_add
                    dataframe_prior.iloc[row_idx, flg_vs_col] = j_vs
                j_vs = j_vs +1      
            else:          
                data_sel = pd.DataFrame([dataframe_prior.loc[i].to_dict()])
                val_min, val_max , flag_add_max = prior_range_computation_one_row(data_sel)
                dataframe_prior.iloc[i, vs_min_col] = val_min
                dataframe_prior.iloc[i, vs_max_col] = val_max
                dataframe_prior.iloc[i, flg_add_col] = flag_add_max
                dataframe_prior.iloc[i, flg_vs_col] = j_vs
                j_vs = j_vs +1 
    
    assert(j_vs == (len(has_single_buyer)+len(has_multi_buyers)))
    assert(j_vs == len(looked_sel_item))
    
    return dataframe_prior


def round_training_x_y_prior(i, dataframe, x_cols, y_col):
    x = list(np.array(dataframe.loc[i, x_cols]))
     
    y_info = np.array(dataframe.loc[i, y_col])[0]
    y = None
    if y_info == 'A':
        y = 0
    elif y_info == 'D':
        y = 1         
    elif int(y_info) == int(x[-2]):
        y = 2
    elif int(y_info) < int(x[-2]):
        y = 3   
    else:
        assert(1 == 0)
        
    return x, y

def round_training_XY_prior_RealData(i, dataframe, x_cols, y_col):
    
    x = list(np.array(dataframe.loc[i, x_cols]))
     
    y_info = np.array(dataframe.loc[i, y_col])[0]
    y = None
    
    if y_info == 'A':
        y = 0
    elif y_info == 'D' or float(y_info) == float(x[-2]):
        y = 1         
    elif float(y_info) < float(x[-2]):
        y = 2   
    else:
        assert(1 == 0)
        
    return x, y

def Convert_to_training_RealData(dataframe):   
    
    X_train, Y_train, X_prior, X_vs_id = [], [], [], []        
    rounds_col = list(dataframe).index('rounds_total')
    # mask_value = 0.
    for i in range(dataframe.shape[0]):        
        v_min = np.array(dataframe.loc[i,['Vs_min']])[0]
        v_max = np.array(dataframe.loc[i,['Vs_max']])[0] 
        v_id = np.array(dataframe.loc[i,['Vs_idx']])[0]
        
        if dataframe.iloc[i,rounds_col] == 1:
            x_cols, y_col = ['s0','b1'], ['s1']
            x, y = round_training_XY_prior_RealData(i, dataframe, x_cols, y_col)           
            
            x = x + [0,0,0,0]           
            X_train.append(x)
            Y_train.append(y)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
        elif dataframe.iloc[i,rounds_col] == 2:
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_XY_prior_RealData(i, dataframe, x_cols1, y_col1)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_XY_prior_RealData(i, dataframe, x_cols2, y_col2)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
        elif dataframe.iloc[i,rounds_col] == 3:
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_XY_prior_RealData(i, dataframe, x_cols1, y_col1)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_XY_prior_RealData(i, dataframe, x_cols2, y_col2)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)

            x_cols3, y_col3 = ['s0','b1', 's1', 'b2', 's2', 'b3'], ['s3']
            x3, y3 = round_training_XY_prior_RealData(i, dataframe, x_cols3, y_col3)           
            X_train.append(x3)
            Y_train.append(y3)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
    return np.array(X_train), np.array(Y_train), np.array(X_prior), np.array(X_vs_id)

def extract_y_counter_interval_Classes5(y_counter, y_counter_pre, b_offer_pre, parts):
    
    all_size = y_counter_pre - b_offer_pre
    each_part_size = all_size / parts
    if b_offer_pre < y_counter <=  b_offer_pre + each_part_size:
        y = 2
    elif b_offer_pre + each_part_size < y_counter <=  b_offer_pre + 2 * each_part_size:
        y = 3
    elif b_offer_pre + 2* each_part_size < y_counter <  y_counter_pre:
        y = 4
    else:
        print(y_counter)
        print(y_counter_pre)
        print(b_offer_pre)
        assert(1 == 0)
    
    return y

def round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols, y_col):
    
    x = list(np.array(dataframe.loc[i, x_cols]))
     
    y_info = np.array(dataframe.loc[i, y_col])[0]
    y = None
    
    if y_info == 'A':
        y = 0
    elif y_info == 'D' or float(y_info) == float(x[-2]):
        y = 1         
    elif float(y_info) < float(x[-2]):
        parts = 3
        y = extract_y_counter_interval_Classes5(float(y_info), float(x[-2]), float(x[-1]), parts)   
    else:
        assert(1 == 0)
        
    return x, y

def Convert_to_training_RealData_Classes5(dataframe):   
    
    X_train, Y_train, X_prior, X_vs_id = [], [], [], []        
    rounds_col = list(dataframe).index('rounds_total')
    # mask_value = 0.
    for i in range(dataframe.shape[0]):        
        # print(i)
        v_min = np.array(dataframe.loc[i,['Vs_min']])[0]
        v_max = np.array(dataframe.loc[i,['Vs_max']])[0] 
        v_id = np.array(dataframe.loc[i,['Vs_idx']])[0]
        
        if dataframe.iloc[i,rounds_col] == 1:
            x_cols, y_col = ['s0','b1'], ['s1']
            x, y = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols, y_col)           
            
            x = x + [0,0,0,0]           
            X_train.append(x)
            Y_train.append(y)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
        elif dataframe.iloc[i,rounds_col] == 2:
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols1, y_col1)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols2, y_col2)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
        elif dataframe.iloc[i,rounds_col] == 3:
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols1, y_col1)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols2, y_col2)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)

            x_cols3, y_col3 = ['s0','b1', 's1', 'b2', 's2', 'b3'], ['s3']
            x3, y3 = round_training_XY_prior_RealData_Classes5(i, dataframe, x_cols3, y_col3)           
            X_train.append(x3)
            Y_train.append(y3)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            
    return np.array(X_train), np.array(Y_train), np.array(X_prior), np.array(X_vs_id)


def Convert_to_training_SynData(dataframe):   
    
    X_train, Y_train, X_prior, X_vs_id, vs = [], [], [], [], []        
    rounds_col = list(dataframe).index('rounds_total')
    # mask_value = 0.
    for i in range(dataframe.shape[0]):        
        v_min = np.array(dataframe.loc[i,['Vs_min']])[0]
        v_max = np.array(dataframe.loc[i,['Vs_max']])[0] 
        v_id = np.array(dataframe.loc[i,['Vs_idx']])[0]
        vs_id = np.array(dataframe.loc[i,['vs']])[0]
        
        if dataframe.iloc[i,rounds_col] == 1:
            x_cols, y_col = ['s0','b1'], ['s1']
            x, y = round_training_x_y_prior(i, dataframe, x_cols, y_col)           
            
            x = x + [0,0,0,0]           
            X_train.append(x)
            Y_train.append(y)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
        elif dataframe.iloc[i,rounds_col] == 2:
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_x_y_prior(i, dataframe, x_cols1, y_col1)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_x_y_prior(i, dataframe, x_cols2, y_col2)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
        elif dataframe.iloc[i,rounds_col] == 3:
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_x_y_prior(i, dataframe, x_cols1, y_col1)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_x_y_prior(i, dataframe, x_cols2, y_col2)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)

            x_cols3, y_col3 = ['s0','b1', 's1', 'b2', 's2', 'b3'], ['s3']
            x3, y3 = round_training_x_y_prior(i, dataframe, x_cols3, y_col3)           
            X_train.append(x3)
            Y_train.append(y3)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
    
    return np.array(X_train), np.array(Y_train), np.array(X_prior), np.array(X_vs_id), np.array(vs)


def round_training_x_y_prior_MultiClass(i, dataframe, x_cols, y_col, dis_pars):
    x = list(np.array(dataframe.loc[i, x_cols]))
     
    y_info = np.array(dataframe.loc[i, y_col])[0]     
    
    y = None
    if y_info == 'A':
        y = 0
    elif y_info == 'D':      #'D':
        y = 1  
    else:
        dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  #10, 60, 5
        Q = [i for i in range(dis_min, dis_max, gap)] 
        lists = []
        for i, item in enumerate(Q):
            lists.append((item, i + 2))
        dicts = dict(lists)
        y = dicts[int(y_info)]
        
    return x, y

def Convert_to_training_SynData_MultiClass(dataframe, dis_pars):   
    
    X_train, Y_train, X_prior, X_vs_id, vs = [], [], [], [], []        
    rounds_col = list(dataframe).index('rounds_total')
    # mask_value = 0.
    for i in range(dataframe.shape[0]):        
        v_min = np.array(dataframe.loc[i,['Vs_min']])[0]
        v_max = np.array(dataframe.loc[i,['Vs_max']])[0] 
        v_id = np.array(dataframe.loc[i,['Vs_idx']])[0]
        vs_id = np.array(dataframe.loc[i,['vs']])[0]
        
        if dataframe.iloc[i,rounds_col] == 1:
            x_cols, y_col = ['s0','b1'], ['s1']
            x, y = round_training_x_y_prior_MultiClass(i, dataframe, x_cols, y_col, dis_pars)           
            
            x = x + [0,0,0,0]           
            X_train.append(x)
            Y_train.append(y)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
        elif dataframe.iloc[i, rounds_col] == 2:
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_x_y_prior_MultiClass(i, dataframe, x_cols1, y_col1, dis_pars)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_x_y_prior_MultiClass(i, dataframe, x_cols2, y_col2, dis_pars)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
        elif dataframe.iloc[i,rounds_col] == 3 :
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_x_y_prior_MultiClass(i, dataframe, x_cols1, y_col1, dis_pars)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_x_y_prior_MultiClass(i, dataframe, x_cols2, y_col2, dis_pars)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)

            x_cols3, y_col3 = ['s0','b1', 's1', 'b2', 's2', 'b3'], ['s3']
            x3, y3 = round_training_x_y_prior_MultiClass(i, dataframe, x_cols3, y_col3, dis_pars)           
            X_train.append(x3)
            Y_train.append(y3)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
    
    return np.array(X_train), np.array(Y_train), np.array(X_prior), np.array(X_vs_id), np.array(vs)


def extract_y_counter_interval(y_counter, y_counter_pre, b_offer_pre, parts):
    all_size = y_counter_pre - b_offer_pre
    each_part_size = all_size / parts
    if b_offer_pre < y_counter <=  b_offer_pre + each_part_size:
        y = 3
    elif b_offer_pre + each_part_size < y_counter <=  b_offer_pre + 2 * each_part_size:
        y = 4
    elif b_offer_pre + 2* each_part_size < y_counter <  y_counter_pre:
        y = 5
    else:
        assert(1 == 0)
    
    return y
    

def round_training_x_y_prior_Class6(i, dataframe, x_cols, y_col):
    x = list(np.array(dataframe.loc[i, x_cols]))
     
    y_info = np.array(dataframe.loc[i, y_col])[0]
    y = None
    if y_info == 'A':
        y = 0
    elif y_info == 'D':
        y = 1         
    elif int(y_info) == int(x[-2]):
        y = 2
    elif int(y_info) < int(x[-2]):
        parts = 3
        y = extract_y_counter_interval(int(y_info), int(x[-2]), int(x[-1]), parts)   
    else:
        assert(1 == 0)
        
    return x, y


def Convert_to_training_SynData_Class6(dataframe):   
    
    X_train, Y_train, X_prior, X_vs_id, vs = [], [], [], [], []        
    rounds_col = list(dataframe).index('rounds_total')
    # mask_value = 0.
    for i in range(dataframe.shape[0]):        
        v_min = np.array(dataframe.loc[i,['Vs_min']])[0]
        v_max = np.array(dataframe.loc[i,['Vs_max']])[0] 
        v_id = np.array(dataframe.loc[i,['Vs_idx']])[0]
        vs_id = np.array(dataframe.loc[i,['vs']])[0]
        
        if dataframe.iloc[i,rounds_col] == 1:
            x_cols, y_col = ['s0','b1'], ['s1']
            x, y = round_training_x_y_prior_Class6(i, dataframe, x_cols, y_col)           
            
            x = x + [0,0,0,0]           
            X_train.append(x)
            Y_train.append(y)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
        elif dataframe.iloc[i, rounds_col] == 2:
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_x_y_prior_Class6(i, dataframe, x_cols1, y_col1)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_x_y_prior_Class6(i, dataframe, x_cols2, y_col2)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
        elif dataframe.iloc[i,rounds_col] == 3 :
            x_cols1, y_col1 = ['s0','b1'], ['s1']
            x1, y1 = round_training_x_y_prior_Class6(i, dataframe, x_cols1, y_col1)           
            x1 = x1 + [0,0,0,0]
            X_train.append(x1)
            Y_train.append(y1)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
            
            x_cols2, y_col2 = ['s0','b1', 's1', 'b2'], ['s2']
            x2, y2 = round_training_x_y_prior_Class6(i, dataframe, x_cols2, y_col2)           
            x2 = x2 + [0,0]
            X_train.append(x2)
            Y_train.append(y2)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)

            x_cols3, y_col3 = ['s0','b1', 's1', 'b2', 's2', 'b3'], ['s3']
            x3, y3 = round_training_x_y_prior_Class6(i, dataframe, x_cols3, y_col3)           
            X_train.append(x3)
            Y_train.append(y3)
            X_prior.append((v_min, v_max))
            X_vs_id.append(v_id)
            vs.append(vs_id)
    
    return np.array(X_train), np.array(Y_train), np.array(X_prior), np.array(X_vs_id), np.array(vs)

def Convert_to_regressor_SynData(X_train, Y_train):
   
    X_train_regressor = copy.deepcopy(X_train)
    X_train_regressor[np.where(X_train_regressor == 0)] = -1

    # print(Y_train.shape)
    # print(X_train[:, 2:4].shape) #(batch_size,2)
    X_train_regressor_t1 = np.hstack((Y_train, X_train_regressor[:, 0:2]))   
    X_train_regressor_t2 = np.hstack((Y_train, X_train_regressor[:, 2:4]))
    X_train_regressor_t3 = np.hstack((Y_train, X_train_regressor[:, 4:6]))

    for i in range(X_train.shape[0]):
        nul_num_t2 = len(np.where(X_train_regressor[i, 2:4] == -1)[0].tolist())
        if nul_num_t2 == 2:
            X_train_regressor_t2[i, :-2] = np.zeros((Y_train[i].shape[0])) - 1
        nul_num_t3 = len(np.where(X_train_regressor[i, 4:6] == -1)[0].tolist())
        if nul_num_t3 == 2:
            X_train_regressor_t3[i, :-2] = np.zeros((Y_train[i].shape[0])) - 1    
    
    X_train_regressor = np.hstack((X_train_regressor_t1, X_train_regressor_t2))
    X_train_regressor = np.hstack((X_train_regressor, X_train_regressor_t3))
    
    return X_train_regressor    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Lvye

"""

import numpy as np
import keras, copy
from tqdm import tqdm
import matplotlib.pyplot as  plt
from utils_dataProcessing import Convert_to_training_SynData
from utils_dataProcessing import Convert_to_training_SynData_Class6
from tensorflow.keras.utils import to_categorical

def build_autoencoder_model(n_features, latent_dim):# watch mask_value!!!
    
    input_layer = keras.Input(shape=(3, n_features)) 
    encoded = keras.layers.Masking(mask_value = 0)(input_layer)
    encoded = keras.layers.GRU(2,return_sequences =  True)(encoded)    #4
    encoded = keras.layers.GRU(2,return_sequences =  True)(encoded)
    encoded = lstm_bottleneck(lstm_units = latent_dim, time_steps = 3)(encoded)
    
    decoded = keras.layers.GRU(2, return_sequences=True)(encoded)
    decoded = keras.layers.GRU(2, return_sequences=True)(decoded) #4   
    decoded = keras.layers.Dense(n_features)(decoded)
    
    lstm_ae = keras.models.Model(inputs = input_layer, outputs = decoded)
    lstm_encoder = keras.models.Model(inputs = input_layer, outputs = encoded)
    
    return lstm_ae, lstm_encoder

def build_autoencoder_model_categorical(n_features, latent_dim):# watch mask_value!!!
    
    input_layer = keras.Input(shape=(3, n_features)) 
    encoded = keras.layers.Masking(mask_value = 0)(input_layer)
    encoded = keras.layers.GRU(3,return_sequences =  True)(encoded) #5 
    encoded = keras.layers.GRU(2,return_sequences =  True)(encoded)
    encoded = lstm_bottleneck(lstm_units = latent_dim, time_steps = 3)(encoded)
    
    decoded = keras.layers.GRU(2, return_sequences=True)(encoded)
    decoded = keras.layers.GRU(3, return_sequences=True)(decoded) #5   
    decoded = keras.layers.Dense(n_features)(decoded)
    
    lstm_ae = keras.models.Model(inputs = input_layer, outputs = decoded)
    lstm_encoder = keras.models.Model(inputs = input_layer, outputs = encoded)
    
    return lstm_ae, lstm_encoder

def build_autoencoder_XYmodel(n_features, latent_dim):# watch mask_value!!!
    
    input_layer = keras.Input(shape=(3, n_features)) 
    encoded = keras.layers.Masking(mask_value = -1)(input_layer)
    encoded = keras.layers.GRU(2,return_sequences =  True)(encoded)    
    encoded = keras.layers.GRU(2,return_sequences =  True)(encoded)
    encoded = lstm_bottleneck(lstm_units = latent_dim, time_steps = 3)(encoded)
    
    decoded = keras.layers.GRU(2, return_sequences=True)(encoded)
    decoded = keras.layers.GRU(2, return_sequences=True)(decoded)    
    decoded = keras.layers.Dense(n_features)(decoded)
    
    lstm_ae = keras.models.Model(inputs = input_layer, outputs = decoded)
    lstm_encoder = keras.models.Model(inputs = input_layer, outputs = encoded)
    
    return lstm_ae, lstm_encoder

def build_autoencoder_XYmodel_categorical(n_features, latent_dim):# watch mask_value!!!
    
    input_layer = keras.Input(shape=(3, n_features)) 
    encoded = keras.layers.Masking(mask_value = -1)(input_layer)
    encoded = keras.layers.GRU(4,return_sequences =  True)(encoded)    
    encoded = keras.layers.GRU(2,return_sequences =  True)(encoded)
    encoded = lstm_bottleneck(lstm_units = latent_dim, time_steps = 3)(encoded)
    
    decoded = keras.layers.GRU(2, return_sequences=True)(encoded)
    decoded = keras.layers.GRU(4, return_sequences=True)(decoded)    
    decoded = keras.layers.Dense(n_features)(decoded)
    
    lstm_ae = keras.models.Model(inputs = input_layer, outputs = decoded)
    lstm_encoder = keras.models.Model(inputs = input_layer, outputs = encoded)
    
    return lstm_ae, lstm_encoder

class lstm_bottleneck(keras.layers.Layer):
      def __init__(self, lstm_units, time_steps, **kwargs):
          self.lstm_units = lstm_units
          self. time_steps = time_steps
          self.lstm_layer =  keras.layers.LSTM(lstm_units, return_sequences=False)
          self.repeat_layer = keras.layers.RepeatVector(time_steps)
          super(lstm_bottleneck, self).__init__(**kwargs)
      def call(self, inputs):
          return self.repeat_layer(self.lstm_layer(inputs))
      def compute_mask(self, inputs, mask = None):
          return mask

def extract_y_color_class6(y, classes):
    color = None
    if y == 0: #  traditional green for accept
        color = 'g'
    elif y == 1: # apple red  for terminate
        color = 'r'
    elif y == 2: # munsell yellow for keeping org price
        color = 'y'
    elif y == 3: #  deep violet for counter smaller price
        color = 'cyan'
    elif y == 4: # munsell yellow for keeping org price
        color = 'deepskyblue'
    elif y == 5: #  deep violet for counter smaller price
        color = 'b'
    else:
        assert(1==0)
        
    return color

def extract_y_color(y, classes):
    color = None
    if y == 0: #  traditional green for accept
        color = 'g'
    elif y == 1: # apple red  for terminate
        color = 'r'
    elif y == 2: # munsell yellow for keeping org price
        color = 'y'
    elif y == 3: #  deep violet for counter smaller price
        color = 'b'
    else:
        assert(1==0)
        
    return color
        
def obtain_each_seller_image_SynData(sellersPriorData, sellers, classes, autoencoder, encoder, save_fig):    
    
    seller_fig_name, sellers_embedded, sellers_Y, sellers_label = [], [], [], []
    xmin_set, xmax_set, ymin_set, ymax_set = [], [], [], []                   
    for i in tqdm(range(len(sellersPriorData))):
        seller_data_prior = sellersPriorData[i]
        seller_i = sellers[i]
        seller_label_set =  np.unique(seller_data_prior['cluster_id']).tolist()
        assert(len(seller_label_set) == 1)
        seller_label = seller_label_set[0]
    
        Xi_seller, Yi_seller,  Vs_Seller = None, None, None
        if classes == 4:
            Xi_seller, Yi_seller, _, _, Vs_Seller = Convert_to_training_SynData(seller_data_prior) 
        elif classes == 6:
            Xi_seller, Yi_seller, _, _, Vs_Seller = Convert_to_training_SynData_Class6(seller_data_prior)
      
        Xi_seller = Xi_seller.astype(np.float64)
        Xi_seller_Em = Xi_seller
        
        print('this seller has {} records'.format(Xi_seller_Em.shape[0]), end = '\n')
        autoencoder.evaluate(Xi_seller_Em.reshape(Xi_seller_Em.shape[0], 3,-1), 
                             Xi_seller_Em.reshape(Xi_seller_Em.shape[0], 3,-1),
                             batch_size = 256)
    
        encoded_res = encoder.predict(Xi_seller_Em.reshape(Xi_seller_Em.shape[0], 3,-1)) ## 829*3*2
        Seller_Embedded = encoded_res[:, 0, :] # 829*2: extract any row is fine
       
        sellers_embedded.append(Seller_Embedded)
        sellers_Y.append(Yi_seller)
        sellers_label.append(seller_label)
        seller_fig_name.append(save_fig +'/'+'seller-'+str(seller_i) +'.png')            
        x_min, x_max = np.min(Seller_Embedded[:, 0]), np.max(Seller_Embedded[:, 0])
        y_min, y_max = np.min(Seller_Embedded[:, 1]), np.max(Seller_Embedded[:, 1])            
        xmin_set.append(x_min)
        xmax_set.append(x_max)
        ymin_set.append(y_min)
        ymax_set.append(y_max)
        
    print('\n')        
    print('starting plotting every seller......')    
    xaxis_min = min(xmin_set) - 0.005
    xaxis_max = max(xmax_set) + 0.005
    yaxis_min = min(ymin_set) - 0.005
    yaxis_max = max(ymax_set) + 0.005
    for i in tqdm(range(len(seller_fig_name))): 
        fig_name = seller_fig_name[i]
        Seller_Embedded = sellers_embedded[i]
        Seller_Yi = sellers_Y[i]
        fig = plt.figure(figsize=(6,5))
        plt.xlim(xaxis_min, xaxis_max)
        plt.ylim(yaxis_min, yaxis_max)
        for j in range(Seller_Embedded.shape[0]):
            encoded_j = Seller_Embedded[j, :]
            # print('{},{}'.format(encoded_j[0], encoded_j[1]))
            y_color = None
            if classes == 4:
                y_color = extract_y_color(Seller_Yi[j], classes)  
            elif classes == 6:
                y_color = extract_y_color_class6(Seller_Yi[j], classes) 
            plt.scatter(encoded_j[0], encoded_j[1], c = y_color)
        # plt.legend()
        fig.savefig(fig_name)
        # plt.show()
        plt.close()
        
    return sellers_label
        
        
def extract_eachGroup_idx(labelArray):
    labelSet = np.unique(labelArray)
    grouped_truth = []
    for label in labelSet:
        idx = np.where(labelArray == label)[0].tolist()
        grouped_truth.append(idx)
    return grouped_truth

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

def get_true_groups(file, k):
    groups_truth = []
    true_numSum = 0
    for i in range(k):
        file_sel = file.loc[file['cluster_id'] == i]
        group_i = np.unique(file_sel['anon_slr_id']).tolist()
        groups_truth.append(group_i)
        true_numSum =  true_numSum + len(group_i)
    return groups_truth, true_numSum

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

def evaluating_results(clusters, labelArray):
    groups_truth = extract_eachGroup_idx(labelArray)
    groups_pred = extract_eachGroup_idx(clusters)
    res_p = evaluating_groups_results(groups_pred, groups_truth)
    res_r = evaluating_groups_results_recall(groups_pred, groups_truth)    
    return res_p, res_r

def Convert_to_EmbeddedData(X, Y):   
    X_new = copy.deepcopy(X)
    X_new[np.where(X_new == 0)] = -1

    # print(X_train[:, 2:4].shape) #(batch_size,2)
    X_new_t1 = np.hstack((Y, X_new[:, 0:2]))   
    X_new_t2 = np.hstack((Y, X_new[:, 2:4]))
    X_new_t3 = np.hstack((Y, X_new[:, 4:6]))

    for i in range(X.shape[0]):
        nul_num_t2 = len(np.where(X_new[i, 2:4] == -1)[0].tolist())
        if nul_num_t2 == 2:
            X_new_t2[i, :-2] = np.zeros((Y[i].shape[0])) - 1
        nul_num_t3 = len(np.where(X_new[i, 4:6] == -1)[0].tolist())
        if nul_num_t3 == 2:
            X_new_t3[i, :-2] = np.zeros((Y[i].shape[0])) - 1   
    
    X_final = np.hstack((X_new_t1, X_new_t2))
    X_final = np.hstack((X_final, X_new_t3))
    
    return X_final    

def obtain_embeddedVector_eachSeller_SynData(sellersPriorData, sellers, classes, autoencoder, encoder):
    sellers_embedded, sellers_label = [], []
    for i in tqdm(range(len(sellersPriorData))):
        seller_data_prior = sellersPriorData[i]
        seller = sellers[i]
        seller_label_set =  np.unique(seller_data_prior['cluster_id']).tolist()
        assert(len(seller_label_set) == 1)
        seller_label = seller_label_set[0]
        
        Xi_seller, Yi_seller = None, None
        if classes == 4:
            Xi_seller, Yi_seller, _, _, _ = Convert_to_training_SynData(seller_data_prior) 
        elif classes == 6:
            Xi_seller, Yi_seller, _, _, _ = Convert_to_training_SynData_Class6(seller_data_prior)
        elif classes >= 15:
            Xi_seller, Yi_seller, _, _, _ = Convert_to_training_SynData_MultiClass(seller_data_prior)      
        Xi_seller = Xi_seller.astype(np.float64)
        Yi_seller_Categ = to_categorical(Yi_seller, classes)
        Xi_seller_Em = Convert_to_EmbeddedData(Xi_seller, Yi_seller_Categ)
       
        print('this seller has {} records'.format(Xi_seller_Em.shape[0]), end = '\n')
        autoencoder.evaluate(Xi_seller_Em.reshape(Xi_seller_Em.shape[0], 3,-1), 
                             Xi_seller_Em.reshape(Xi_seller_Em.shape[0], 3,-1),
                             batch_size = 256)
    
        encoded_res = encoder.predict(Xi_seller_Em.reshape(Xi_seller_Em.shape[0], 3,-1)) ## 829*3*2
        Seller_Embedded = encoded_res[:, 0, :] # 829*2: extract any row is fine    
        
        sellers_embedded.append(Seller_Embedded)
        sellers_label.append(seller_label)
        
    return sellers_embedded, sellers_label
    

    

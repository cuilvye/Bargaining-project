#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Lvye

"""

from tensorflow import keras
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import to_categorical
from utils_inference import *
from utils_dataProcessing import *

from tensorflow.keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    n_classes,
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    ):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

def transformer_encoder_real(inputs, head_size, num_heads, ff_dim):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-4)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=0
    )(x, x)
    # x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-4)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    # x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res



def build_model_real(
    n_classes,
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    ):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_real(x, head_size, num_heads, ff_dim)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        # x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

def build_rnn_model():   

    allow_cudnn_kernal = True
    if allow_cudnn_kernal:
        units = 64
        lstm_layer = keras.layers.LSTM(units, input_shape = (input_dim,))
    else:
        units = 64
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape = (input_dim,)
            )
    # layers.BatchNormalization() could be tried later  
    # input_shape do not need to give the batch_size, only the dimensions of data not sample number
    model = keras.models.Sequential(
        [ keras.layers.Masking(mask_value = 0., input_shape = (1, input_dim)),
          keras.layers.SimpleRNN(64,  return_sequences = True), 
          keras.layers.SimpleRNN(128, return_sequences = True),
          keras.layers.SimpleRNN(64, return_sequences = False),
          keras.layers.Dense(classes, activation=keras.activations.softmax),
        ]
        ) 
    model.summary()
    
    return model

def build_rnn_GRUModel_onlyX(classes):
    model = tf.keras.Sequential([
        keras.layers.Masking(mask_value = 0, input_shape=(3, 2)),
        keras.layers.GRU(4, return_sequences=True),
        keras.layers.GRU(4, return_sequences=False),
        keras.layers.Dense(classes, activation=keras.activations.softmax)
      ])
    model.summary()
    return model

def build_rnn_GRUModel(classes):
    input_dim = 3
    model = tf.keras.Sequential([
        keras.layers.Masking(mask_value = 0, input_shape=(3, input_dim)),
        keras.layers.GRU(4, return_sequences=True),
        keras.layers.GRU(4, return_sequences=False),
        keras.layers.Dense(classes, activation=keras.activations.softmax)
      ])
    model.summary()
    return model

def build_rnn_GRUModel_Mask(classes, real_classes):
    input_dim = 2 + classes
    model = tf.keras.Sequential([
        keras.layers.Masking(mask_value = -1, input_shape=(3, input_dim)),
        keras.layers.GRU(4, return_sequences=True),
        keras.layers.GRU(4, return_sequences=False),
        keras.layers.Dense(real_classes, activation=keras.activations.softmax)
      ])
    model.summary()
    return model

def build_rnn_GRUModel_regressor(extend_dim):
    input_dim = 2 + extend_dim
    model = tf.keras.Sequential([
        keras.layers.Masking(mask_value = -1, input_shape=(3, input_dim)),
        keras.layers.GRU(4, return_sequences=True),
        keras.layers.GRU(4, return_sequences=False),
        keras.layers.Dense(1)
      ])
    model.summary()
    return model

def build_rnn_GRUModel_COMPLEX(classes):
    input_dim = 3
    model = tf.keras.Sequential([
        keras.layers.Masking(mask_value = 0, input_shape=(3, input_dim)),
        keras.layers.GRU(8, return_sequences=True),
        keras.layers.GRU(8, return_sequences=False),
        keras.layers.Dense(classes, activation=keras.activations.softmax)
      ])
    model.summary()
    return model

def build_rnn_GRUModel_Mask_COMPLEX(classes, real_classes):
    input_dim = 2 + classes
    model = tf.keras.Sequential([
        keras.layers.Masking(mask_value = -1, input_shape=(3, input_dim)),
        keras.layers.GRU(8, return_sequences=True),
        keras.layers.GRU(8, return_sequences=False),
        keras.layers.Dense(real_classes, activation=keras.activations.softmax)
      ])
    model.summary()
    return model

def accuracy_compute(y, logits):
        
    actual_label = np.argmax(y, axis = 1) # find the maximum in each row
    pred_label = np.argmax(logits, axis = 1)
    
    acc = len(np.where(actual_label == pred_label)[0]) / len(actual_label)
    
    return acc

def Expect_totalLoss_computation(rBatch_A, rBatch_B, postBatch):
    
    def Expect_reward_i_compute(i_rA, i_rB, vs_i_postSet):
        res = [ p_vs_iSet[1] * (i_rA + i_rB) for p_vs_iSet in vs_i_postSet]
        res_sum = np.sum(res)    
        
        return res_sum
    
    batch_rewards = list(map(lambda x, y, z: Expect_reward_i_compute(x, y, z), rBatch_A, list(rBatch_B), postBatch))    
    batch_meanReward = np.mean(batch_rewards)
    batch_meanLoss = - batch_meanReward
    
    return batch_meanLoss

def performance_on_Dataset_ExpectLosses(X, Y, X_prior, X_vs, bt_size, syn_tag, vs_priorDistr_type, dis_pars, model, r_cons, isTest, path_res, save_tuple):
    
    numBatches = math.ceil(X.shape[0] / bt_size)
    loss_total, acc_total = [], []
    
    for batch in range(0, numBatches):
        start  = batch * bt_size
        end  =  start + bt_size
        x_batch, y_batch, x_batchPrior = X[start:end], Y[start:end], X_prior[start:end]    
        vs_batch, priorBatch, postBatch = Vs_inference_with_rowsNumpy(x_batch, y_batch, x_batchPrior, syn_tag, vs_priorDistr_type, dis_pars, model)
                                              
        x_exBatch = np.insert(x_batch, 0, vs_batch, axis = 1)
        x_exBatch = x_exBatch.reshape(-1, 1, x_exBatch.shape[1])
        logits = model(x_exBatch) 
        
        #  accuracy computation #          
        acc_batch = accuracy_compute(y_batch, logits.numpy())        
        # y_trueIdx_batch = list(np.argmax(y_batch, axis = 1)) 
        # continue compute other loss value #
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)                
        y_batch = tf.convert_to_tensor(y_batch)
        loss_B = cce(y_batch, logits)
        rBatch_B = -loss_B
        
        # model loss value #
        loss_batch = tf.reduce_mean(loss_B) 
        rBatch_A = rewardBatch_A_compute(r_cons, vs_batch, x_batchPrior) 
               
        # loss_mean = math.exp(opt_min.numpy() / bt_size)  
        loss_mean = Expect_totalLoss_computation(rBatch_A, np.array(rBatch_B), postBatch)         
        
        loss_total.append(loss_mean)
        acc_total.append(acc_batch)

    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    
    return loss, acc


def Callback_EarlyStopping(LossList, min_delta = 0.005, patience = 30):
    # no early stoping for 2*patience epochs
    if len(LossList) // patience < 2:
        return False
    # mean loss for last patience epochs and second-last patience epochs:
    mean_previous = np.mean(LossList[::-1][patience:2*patience]) # second-last
    mean_recent = np.mean(LossList[::-1][:patience]) # last
    delta_abs = np.abs(mean_recent - mean_previous)
    delta_abs = np.abs(delta_abs / mean_previous) # relative change
    if delta_abs < min_delta:
        print('Loss did not change much from last {} epochs '.format(patience), end ='\n')
        print(' Percent change in loss value: {:.4f}% '.format(delta_abs*1e2), end = '\n')
        return True
    else:
        return False
 
def performance_on_SynDataset_V0_alpha_log(dataframe_prior, alpha, classes, bt_size, syn_tag, dis_type, dis_pars, model, isTest, path_res, save_tuple):
    
    loss_total, acc_total, mse_total, vsAcc_total = [], [], [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [], 'vs_mse': [], 'vs_acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes == 15:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars) 
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        # infer vs using threads from the same seller and the same item #
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model)
        mse_batch = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch,[-1,1])) 
             
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
        vsAcc_batch = predTrue_num/vs_tensorBatch.get_shape()[0]
         
        # loss B  computation#  # insert the true Vs in Learned action prediction model
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
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
        
        real_loss_A = []
        for i in range(x_extends_Tensor.get_shape()[0]):
            xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], 3, -1]))
            yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)

            yiProbs_onVs_pVs = tf.math.multiply(yi_probs, vs_probs)
            # print(yiProbs_onVs_pVs.get_shape()) #(13,1)
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

            real_loss_A_i = - tf.math.log(sum_Numo/sum_Denom)
            real_loss_A.append(real_loss_A_i)

        loss_total_real = alpha * tf.reduce_mean(real_loss_A) + (1-alpha) * tf.reduce_mean(loss_B)        
        
        loss_mean = loss_total_real.numpy()
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
        
        loss_total.append(loss_mean)
        acc_total.append(acc_batch)
        mse_total.append(mse_batch)
        vsAcc_total.append(vsAcc_batch)  
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['acc'].append(acc_batch)
            test_logs['vs_mse'].append(mse_batch)
            test_logs['vs_acc'].append(vsAcc_batch)
            test_logs['total_loss'].append(loss_mean) 
            it, g = save_tuple[0], save_tuple[1]  
            if batch % 10 == 0 or batch == len(batch_dataFrame)-1:
                print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f}, vs_mse: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, mse_batch, vsAcc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    mse_mean = np.mean(mse_total)
    vsAcc_mean = np.mean(vsAcc_total)
    
    return loss, acc, mse_mean, vsAcc_mean                                                         
def performance_on_SynDataset_ExpectLosses_Tensor_Multirows(dataframe_prior, classes, alpha, bt_size, syn_tag, dis_type, dis_pars, model, r_cons, path_res, save_tuple, isTest):
    
    loss_total, acc_total, mse_total, vsAcc_total = [], [], [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [], 'vs_mse': [], 'vs_acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)         
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame) 
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        # vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_rowsTensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, syn_tag, dis_type, dis_pars, model) 
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model)
        # based on x_vsID_Batch_temp to combine their inference into a unifying one
        
        mse_batch = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch,[-1,1]))          
        rA_tensorBatch = rewardBatch_A_computeTensor(r_cons, vs_tensorBatch, xPrior_tensorBatch) 
        
        predTrue_num = len(np.where(rA_tensorBatch.numpy() > 0)[0].tolist())
        vsAcc_batch = predTrue_num/x_tensorBatch.get_shape()[0]
        
        # x_exBatch_Tensor = tf.keras.layers.concatenate([tf.reshape(vs_tensorBatch, [-1, 1]), x_tensorBatch], axis = -1)  
        vs_tensorBatch_reshape = tf.reshape(vs_tensorBatch, [-1, 1])
        x_exBatch_Tensor_timeStep1 = tf.keras.layers.concatenate([vs_tensorBatch_reshape, x_tensorBatch[:,0:2]], axis = -1)
   
        vs_tensorBatch_copy = copy.deepcopy(vs_tensorBatch_reshape)
        check_X_timeStep2_valid = tf.subtract(vs_tensorBatch_copy, x_tensorBatch[:,2:3])
        vs_tensorBatch_copy = tf.where(check_X_timeStep2_valid == vs_tensorBatch_reshape, tf.zeros_like(vs_tensorBatch_copy), vs_tensorBatch_copy)
        x_exBatch_Tensor_timeStep2 = tf.keras.layers.concatenate([vs_tensorBatch_copy, x_tensorBatch[:,2:4]], axis = -1)
           
        vs_tensorBatch_Copy = copy.deepcopy(vs_tensorBatch_reshape)
        check_X_timeStep3_valid = tf.subtract(vs_tensorBatch_Copy, x_tensorBatch[:,4:5])
        vs_tensorBatch_Copy = tf.where(check_X_timeStep3_valid == vs_tensorBatch_reshape, tf.zeros_like(vs_tensorBatch_Copy), vs_tensorBatch_Copy)
        x_exBatch_Tensor_timeStep3 = tf.keras.layers.concatenate([vs_tensorBatch_Copy, x_tensorBatch[:,4:6]], axis = -1)
           
        x_exBatch_Tensor_timeStep_temp = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep1, x_exBatch_Tensor_timeStep2], axis = -1)
        x_exBatch_Tensor = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep_temp, x_exBatch_Tensor_timeStep3], axis = -1)

        # y_trueIdx_batchTensor = tf.math.argmax(y_tensorBatch, axis = 1)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))
        # print(logits)        
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))          
        
        # continue compute other loss value #              
        loss_B = cce(y_tensorBatch, logits)
        rBatch_B = -loss_B              
        loss_batch = tf.reduce_mean(loss_B) 
             
        postBatch_probs = post_tensorBatch[:,:,1] # (10,13)
        r_total = tf.add(alpha * rA_tensorBatch, (1-alpha) * rBatch_B)
        r_total = tf.repeat(r_total, repeats = postBatch_probs.get_shape()[1])
        r_total = tf.reshape(r_total, [postBatch_probs.get_shape()[0], -1])
        # print(r_total)
        opt_min = -tf.multiply(postBatch_probs, r_total)
        
        opt_min = tf.reduce_sum(opt_min)/x_tensorBatch.get_shape()[0] 
        loss_mean =  opt_min
        
        loss_total.append(loss_mean)
        acc_total.append(acc_batch)
        mse_total.append(mse_batch)
        vsAcc_total.append(vsAcc_batch)  
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['acc'].append(acc_batch)
            test_logs['vs_mse'].append(mse_batch)
            test_logs['vs_acc'].append(vsAcc_batch)
            test_logs['total_loss'].append(loss_mean) 
            it, g = save_tuple[0], save_tuple[1]  
            if batch % 8 == 0 or batch == len(batch_dataFrame)-1:
                print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f}, vs_mse: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, mse_batch, vsAcc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    mse_mean = np.mean(mse_total)
    vsAcc_mean = np.mean(vsAcc_total)
    
    return loss, acc, mse_mean, vsAcc_mean

def performance_on_SynDataset_Multirows_V0_alpha_log(dataframe_prior, alpha, classes, bt_size, syn_tag, dis_type, dis_pars, model, isTest, path_res, save_tuple):
    
    loss_total, acc_total, mse_total, vsAcc_total = [], [], [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [], 'vs_mse': [], 'vs_acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars) 
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        # infer vs using threads from the same seller and the same item #
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model)
        mse_batch = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch,[-1,1])) 
             
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
        vsAcc_batch = predTrue_num/vs_tensorBatch.get_shape()[0]
         
        # loss B  computation#  
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        # logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        
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
        
        real_loss_A = []
        for i in range(x_extends_Tensor.get_shape()[0]):
            xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], 3, -1]))
            yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)

            yi_probs = tf.clip_by_value(yi_probs, 1e-10, 1)

            yiProbs_onVs_pVs = tf.math.multiply(yi_probs, vs_probs)
            # print(yiProbs_onVs_pVs.get_shape()) #(13,1)
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
                         
            sum_Numo = tf.reduce_sum(yiProbs_onVs_pVs_feaSet) 

            real_loss_A_i =  tf.math.log(sum_Denom) - tf.math.log(sum_Numo)
            real_loss_A.append(real_loss_A_i)

        loss_total_real = alpha * tf.reduce_mean(real_loss_A) + (1-alpha) * tf.reduce_mean(loss_B)              
        loss_mean = loss_total_real.numpy()
     
        loss_total.append(loss_mean)
        acc_total.append(acc_batch)
        mse_total.append(mse_batch)
        vsAcc_total.append(vsAcc_batch)  
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['acc'].append(acc_batch)
            test_logs['vs_mse'].append(mse_batch)
            test_logs['vs_acc'].append(vsAcc_batch)
            test_logs['total_loss'].append(loss_mean) 
            it, g = save_tuple[0], save_tuple[1]  
            if batch == len(batch_dataFrame)-1:
            #     print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f}, vs_mse: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, mse_batch, vsAcc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    mse_mean = np.mean(mse_total)
    vsAcc_mean = np.mean(vsAcc_total)
    
    return loss, acc, mse_mean, vsAcc_mean

def performance_on_SynDataset_Multirows_V0_alpha_log_transformer(dataframe_prior, alpha, classes, bt_size, syn_tag, dis_type, dis_pars, model, isTest, path_res, save_tuple):
    
    loss_total, acc_total, mse_total, vsAcc_total = [], [], [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [], 'vs_mse': [], 'vs_acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars) 
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        # infer vs using threads from the same seller and the same item #
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor_transformer(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model)
        mse_batch = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch,[-1,1])) 
             
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
        vsAcc_batch = predTrue_num/vs_tensorBatch.get_shape()[0]
         
        # loss B  computation#  
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        # logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], -1, 1]))           
        logits_org = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], -1, 1]))      
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
        
        real_loss_A = []
        for i in range(x_extends_Tensor.get_shape()[0]):
            xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], -1, 1]))
            yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)

            yi_probs = tf.clip_by_value(yi_probs, 1e-10, 1)
            
            yiProbs_onVs_pVs = tf.math.multiply(yi_probs, vs_probs)
            # print(yiProbs_onVs_pVs.get_shape()) #(13,1)
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
                         
            sum_Numo = tf.reduce_sum(yiProbs_onVs_pVs_feaSet) 

            real_loss_A_i = tf.math.log(sum_Denom) - tf.math.log(sum_Numo)
            real_loss_A.append(real_loss_A_i)

        loss_total_real = alpha * tf.reduce_mean(real_loss_A) + (1-alpha) * tf.reduce_mean(loss_B)              
        loss_mean = loss_total_real.numpy()
     
        loss_total.append(loss_mean)
        acc_total.append(acc_batch)
        mse_total.append(mse_batch)
        vsAcc_total.append(vsAcc_batch)  
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['acc'].append(acc_batch)
            test_logs['vs_mse'].append(mse_batch)
            test_logs['vs_acc'].append(vsAcc_batch)
            test_logs['total_loss'].append(loss_mean) 
            it, g = save_tuple[0], save_tuple[1]  
            if batch == len(batch_dataFrame)-1:
            #     print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f}, vs_mse: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, mse_batch, vsAcc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    mse_mean = np.mean(mse_total)
    vsAcc_mean = np.mean(vsAcc_total)
    
    return loss, acc, mse_mean, vsAcc_mean


def performance_on_SynDataset_Multirows_V0_alpha_log_BF(dataframe_prior, t_val, epsilon, classes, bt_size, syn_tag, dis_type,
                                                     dis_pars, model):
    loss_total, acc_total, mse_total, vsAcc_total = [], [], [], []

    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)

    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]

        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##

        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars)

        Y = to_categorical(Y, classes)
        X = X.astype(np.float64)

        x_tensorBatch = tf.convert_to_tensor(X, dtype=float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype=float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype=float)
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype=float)

        # infer vs using threads from the same seller and the same item #
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch,
                                                                                                 y_tensorBatch,
                                                                                                 xPrior_tensorBatch,
                                                                                                 x_vsID_Batch, dis_type,
                                                                                                 syn_tag, dis_pars,
                                                                                                 model)
        mse_batch = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch, [-1, 1]))

        # compute inferred vs accuracy #
        difference = tf.zeros([xPrior_tensorBatch.get_shape()[0], ], dtype=float).numpy()
        differ_1 = tf.subtract(vs_tensorBatch, xPrior_tensorBatch[:, 0])
        differ_2 = tf.subtract(xPrior_tensorBatch[:, 1], vs_tensorBatch)
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            if differ_1[i] < 0:
                difference[i] = differ_1[i]
            elif differ_2[i] < 0:
                difference[i] = differ_2[i]

        predTrue_num = len(np.where(difference == 0)[0].tolist())
        vsAcc_batch = predTrue_num / vs_tensorBatch.get_shape()[0]

        # loss B  computation#
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))

        logits_org = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits = tf.clip_by_value(logits_org, 1e-10, 1)
        assert logits_org.get_shape() == logits.get_shape()

        loss_B = cce(y_tensorBatch, logits)
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))

        # loss A computation #
        vs_priorSet_TensorBatch, vs_priorset_Tensor = Vs_prior_distr_computeTensor(xPrior_tensorBatch, dis_type,
                                                                                   syn_tag, dis_pars)
        y_trueIdx_batchTensor = tf.math.argmax(y_tensorBatch, axis=1)

        x_extends = tf.tile(x_tensorBatch, tf.constant([1, vs_priorset_Tensor.get_shape()[0]]))
        x_extends = tf.reshape(x_extends, [x_tensorBatch.get_shape()[0], vs_priorset_Tensor.get_shape()[0], -1])

        x_exBatch_Tensor_timeStep1 = tf.keras.layers.concatenate(
            [vs_priorSet_TensorBatch[:, :, 0:1], x_extends[:, :, 0:2]], axis=-1)

        vs_priorSet_TensorBatch_copy = copy.deepcopy(vs_priorSet_TensorBatch[:, :, 0:1])
        check_X_timeStep2_valid = tf.subtract(vs_priorSet_TensorBatch_copy, x_extends[:, :, 2:3])
        vs_priorSet_TensorBatch_copy = tf.where(check_X_timeStep2_valid == vs_priorSet_TensorBatch[:, :, 0:1],
                                                tf.zeros_like(vs_priorSet_TensorBatch_copy),
                                                vs_priorSet_TensorBatch_copy)
        x_exBatch_Tensor_timeStep2 = tf.keras.layers.concatenate([vs_priorSet_TensorBatch_copy, x_extends[:, :, 2:4]],
                                                                 axis=-1)

        vs_priorSet_TensorBatch_Copy = copy.deepcopy(vs_priorSet_TensorBatch[:, :, 0:1])
        check_X_timeStep3_valid = tf.subtract(vs_priorSet_TensorBatch_Copy, x_extends[:, :, 4:5])
        vs_priorSet_TensorBatch_Copy = tf.where(check_X_timeStep3_valid == vs_priorSet_TensorBatch[:, :, 0:1],
                                                tf.zeros_like(vs_priorSet_TensorBatch_Copy),
                                                vs_priorSet_TensorBatch_Copy)
        x_exBatch_Tensor_timeStep3 = tf.keras.layers.concatenate([vs_priorSet_TensorBatch_Copy, x_extends[:, :, 4:6]],
                                                                 axis=-1)

        x_exBatch_Tensor_timeStep_temp = tf.keras.layers.concatenate(
            [x_exBatch_Tensor_timeStep1, x_exBatch_Tensor_timeStep2], axis=-1)
        x_extends_Tensor = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep_temp, x_exBatch_Tensor_timeStep3],
                                                       axis=-1)

        vs_probs = tf.gather(vs_priorset_Tensor, [1], axis=1)
        vs_setTensor = tf.gather(vs_priorset_Tensor, [0], axis=1)

        real_loss_A = []
        for i in range(x_extends_Tensor.get_shape()[0]):
            xi_logits = model(tf.reshape(x_extends_Tensor[i, :, :], [x_extends_Tensor.get_shape()[1], 3, -1]))
            yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis=1)

            yi_probs = tf.clip_by_value(yi_probs, 1e-10, 1)

            yiProbs_onVs_pVs = tf.math.multiply(yi_probs, vs_probs)
            # print(yiProbs_onVs_pVs.get_shape()) #(13,1)
            sum_Denom = tf.reduce_sum(yiProbs_onVs_pVs)

            res_1 = tf.squeeze(tf.where(tf.squeeze(vs_setTensor, axis=1) <= xPrior_tensorBatch[i, 1]), axis=1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= tf.squeeze(vs_setTensor, axis=1)), axis=1)
            res = tf.concat((res_1, res_2), axis=0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                            unique_idx,
                                                            tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            yiProbs_onVs_pVs_feaSet = tf.gather(tf.squeeze(yiProbs_onVs_pVs, axis=1), more_than_one_vals)

            sum_Numo = tf.reduce_sum(yiProbs_onVs_pVs_feaSet)

            real_loss_A_i_temp = sum_Numo / sum_Denom - epsilon
            if real_loss_A_i_temp <= 0:
                real_loss_A_i_temp = 1e-323 # tf.math.log only handle 1e-37, so need math.log
            real_loss_A_i = math.log(real_loss_A_i_temp)  # real_loss_A_i = tf.math.log(sum_Numo / sum_Denom - epsilon)
            real_loss_A_i = tf.convert_to_tensor(real_loss_A_i, dtype=float)
            real_loss_A.append(real_loss_A_i)

        loss_total_real = tf.reduce_mean(loss_B) - (1 / t_val) * tf.reduce_mean(real_loss_A)
        loss_mean = loss_total_real.numpy()

        loss_total.append(loss_mean)
        acc_total.append(acc_batch)
        mse_total.append(mse_batch)
        vsAcc_total.append(vsAcc_batch)

    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    mse_mean = np.mean(mse_total)
    vsAcc_mean = np.mean(vsAcc_total)

    return loss, acc, mse_mean, vsAcc_mean

def performance_on_SynDataset_Multirows_V0_alpha_log_revised(dataframe_prior, alpha, classes, bt_size, syn_tag, dis_type, dis_pars, model, isTest, path_res, save_tuple):
    
    loss_total, acc_total, mse_total, vsAcc_total = [], [], [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [], 'vs_mse': [], 'vs_acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars) 
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        # infer vs using threads from the same seller and the same item #
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model)
        mse_batch = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch,[-1,1])) 
             
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
        vsAcc_batch = predTrue_num/vs_tensorBatch.get_shape()[0]
         
        # loss B  computation#  
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
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
        
        real_loss_A, visited_vsID = [], [] 
        for i in range(x_extends_Tensor.get_shape()[0]):
            vs_ID = x_vsID_Batch[i]
            if vs_ID.numpy() not in visited_vsID:
                visited_vsID.append(vs_ID.numpy())                     
                multi_res = tf.where(tf.equal(vs_ID, x_vsID_Batch))                                 
                
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
                
                data_multi = tf.gather(x_extends_Tensor, tf.reshape(multi_res, [-1]), axis = 0)
                y_trueIdx_new = tf.gather(y_trueIdx_batchTensor, tf.reshape(multi_res, [-1]), axis = 0)
                
                sum_Numo = tf.convert_to_tensor(0.)
                yi_probs_temp = tf.convert_to_tensor(0.)
                for d in range(data_multi.get_shape()[0]):
                    yi_d_logits = model(tf.reshape(data_multi[d,:,:], [data_multi.get_shape()[1], 3, -1])) #(13,4)
                    yi_d_true = tf.gather(y_trueIdx_new, d, axis = 0) 
                    yi_d_probs = tf.gather(yi_d_logits, yi_d_true, axis = 1) #(13,)   
                    yi_d_probs = tf.reshape(yi_d_probs, [-1, 1])
                    
                    ## the second part for agent A
                    if d == 0:
                        yi_probs_temp = yi_d_probs
                    else:
                        yi_probs_temp = tf.concat([yi_probs_temp, yi_d_probs], axis = 1) # (13,9)
                    yiProbs_onVs_pVs = tf.math.multiply(yi_d_probs, vs_probs)
                    yiProbs_onVs_pVs_feaSet = tf.gather(tf.squeeze(yiProbs_onVs_pVs, axis = 1), more_than_one_vals)
                    sum_Numo_d = tf.reduce_sum(yiProbs_onVs_pVs_feaSet)    
                    sum_Numo = sum_Numo + tf.math.log(sum_Numo_d)
  
                yi_probs = tf.math.reduce_prod(yi_probs_temp, axis = 1) # the product of each row   
                yi_probs = tf.reshape(yi_probs, [-1, 1])            
                sum_Denom = tf.reduce_sum(tf.math.multiply(yi_probs, vs_probs))   
                assert(yi_probs_temp.get_shape()[1] == multi_res.get_shape()[0])
        
                log_pA_i = sum_Numo - tf.math.log(sum_Denom)        
                real_loss_A.append(-log_pA_i/data_multi.get_shape()[0]) 

        loss_total_real = alpha * tf.reduce_mean(real_loss_A) + (1-alpha) * tf.reduce_mean(loss_B)              
        loss_mean = loss_total_real.numpy()
     
        loss_total.append(loss_mean)
        acc_total.append(acc_batch)
        mse_total.append(mse_batch)
        vsAcc_total.append(vsAcc_batch)  
        
        # if isTest:
        #     test_logs['batch'].append(batch + 1)
        #     test_logs['acc'].append(acc_batch)
        #     test_logs['vs_mse'].append(mse_batch)
        #     test_logs['vs_acc'].append(vsAcc_batch)
        #     test_logs['total_loss'].append(loss_mean) 
        #     it, g = save_tuple[0], save_tuple[1]  
        #     if batch == len(batch_dataFrame)-1:
        #     #     print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f}, vs_mse: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, mse_batch, vsAcc_batch), end = "\n")
        #         pd.DataFrame(test_logs).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    mse_mean = np.mean(mse_total)
    vsAcc_mean = np.mean(vsAcc_total)
    
    return loss, acc, mse_mean, vsAcc_mean

def performance_on_SynDataset_Multirows_V0_InferOnly(dataframe_prior, classes, bt_size, syn_tag, dis_type, dis_pars, model, isTest, path_res, save_tuple):
    
    loss_total, mse_total, vsAcc_total = [], [], []
    test_logs = {'batch': [] , 'total_Vs': [],  'vs_mse': [], 'vs_acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars) 
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        # infer vs using threads from the same seller and the same item #
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model)
        mse_batch = mean_squared_error(xVs_tensorBatch, tf.reshape(vs_tensorBatch,[-1,1])) 
             
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
        vsAcc_batch = predTrue_num/vs_tensorBatch.get_shape()[0]
        
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
        
        real_loss_A = []
        for i in range(x_extends_Tensor.get_shape()[0]):
            xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], 3, -1]))
            yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)

            yiProbs_onVs_pVs = tf.math.multiply(yi_probs, vs_probs)
            # print(yiProbs_onVs_pVs.get_shape()) #(13,1)
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

            real_loss_A_i = - math.log(sum_Numo/sum_Denom)
            real_loss_A.append(real_loss_A_i)

        loss_total_real = tf.reduce_mean(real_loss_A)       
        
        loss_mean = loss_total_real.numpy()
        
        loss_total.append(loss_mean)
        mse_total.append(mse_batch)
        vsAcc_total.append(vsAcc_batch)  
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['vs_mse'].append(mse_batch)
            test_logs['vs_acc'].append(vsAcc_batch)
            test_logs['total_Vs'].append(loss_mean) 
            it, g = save_tuple[0], save_tuple[1]  
            if batch % 10 == 0 or batch == len(batch_dataFrame)-1:
                print("[INFO] test dataset batch {}/{}, loss_Vs: {:.5f},  vs_mse: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean,  mse_batch, vsAcc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    mse_mean = np.mean(mse_total)
    vsAcc_mean = np.mean(vsAcc_total)
    
    return loss, mse_mean, vsAcc_mean


def performance_on_SynDataset_DualLearning(dataframe_prior, classes, bt_size, model_classify, model_regressor, dis_pars, path_res, save_tuple, isTest):
    
    it, g = save_tuple[0], save_tuple[1]                 
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss, test_loss_regressor, test_acc, test_vsMse, test_vsAcc = [], [], [], [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [], 'regressor_loss': [], 'acc': [], 'vs_mse': [],  'vs_acc': [] }
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior, X_Vs = None, None, None, None
        if classes == 4:
            X, Y, X_prior, _, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
           X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars)        
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   

        dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
        vs_SetTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)
        
        X_regressor = Convert_to_regressor_SynData(X, Y)
        xRegressor_tensorBatch = tf.convert_to_tensor(X_regressor, dtype = float)

        ##################### y_true + x -> vs #########################
        vs_tensorBatch = model_regressor(tf.reshape(xRegressor_tensorBatch, [xRegressor_tensorBatch.get_shape()[0], 3, -1])) 
        
        vs_tensorBatch_Corr = tf.abs(tf.subtract(tf.reshape(vs_tensorBatch,[-1, 1]), tf.reshape(vs_SetTensor, [1, -1])))
        vs_tensorBatch_Corr = tf.reshape(vs_tensorBatch_Corr, [-1, vs_SetTensor.get_shape()[0]])
        vs_idx = np.argmin(vs_tensorBatch_Corr, axis = 1)
        vs_tensorBatch_Corr = tf.gather(vs_SetTensor, vs_idx.tolist())
        
        vs_mseBatch = mean_squared_error(xVs_tensorBatch, vs_tensorBatch_Corr) 
        differ_cor1 = tf.subtract(tf.reshape(vs_tensorBatch_Corr, [-1]), tf.reshape(xPrior_tensorBatch[:, 0], [-1]))
        differ_cor2 = tf.subtract(tf.reshape(xPrior_tensorBatch[:, 1], [-1]), tf.reshape(vs_tensorBatch_Corr, [-1]))       
        predTrue_num = 0
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            if differ_cor1[i] >= 0 and differ_cor2[i] >= 0:
                predTrue_num = predTrue_num + 1
        vs_accBatch = predTrue_num/vs_tensorBatch_Corr.get_shape()[0]
        
        loss_regressor = []   
        differ_1 = tf.subtract(tf.reshape(vs_tensorBatch, [-1]), tf.reshape(xPrior_tensorBatch[:, 0], [-1]))
        differ_2 = tf.subtract(tf.reshape(xPrior_tensorBatch[:, 1], [-1]), tf.reshape(vs_tensorBatch, [-1]))       
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            if differ_1[i] < 0:
                loss_regressor.append(differ_1[i] ** 2)
            elif differ_2[i] < 0:
                loss_regressor.append(differ_2[i] ** 2)
            else:
                loss_regressor.append(0)
    
        loss_regressor = tf.convert_to_tensor(loss_regressor)
        loss_regressor_mean = tf.reduce_mean(loss_regressor)          
        
        ##################### vs + x -> y #########################
        # vs_RandomSet = []
        # for i in range(xPrior_tensorBatch.get_shape()[0]):
        #     res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
        #     res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
        #     res = tf.concat((res_1, res_2), axis = 0)
        #     unique_res_vals, unique_idx = tf.unique(res)
        #     count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
        #                                                unique_idx,
        #                                                tf.shape(res)[0])
        #     more_than_one = tf.greater(count_res_unique, 1)
        #     more_than_one_idx = tf.squeeze(tf.where(more_than_one))
        #     more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
        #     feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
        #     feasible_set = tf.reshape(feasible_set, [-1])
            
        #     idxs = tf.range(tf.shape(feasible_set)[0])
        #     ridxs =  tf.random.shuffle(idxs)[:1] # :sample_num
        #     vs_random = tf.gather(feasible_set, ridxs)[0]
        #     vs_RandomSet.append(vs_random.numpy())
        
        # vs_Random_tensorBatch = tf.convert_to_tensor(vs_RandomSet)
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model_classify(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)   
        
        loss_classify = tf.reduce_mean(loss_B)
        acc_batch = accuracy_compute(np.array(y_tensorBatch), logits.numpy()) 
        
        # summary the results #           
        test_loss.append(loss_classify.numpy())        
        test_acc.append(acc_batch)               
        test_loss_regressor.append(loss_regressor_mean.numpy())
        test_vsMse.append(vs_mseBatch) 
        test_vsAcc.append(vs_accBatch)  
        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['regressor_loss'].append(loss_regressor_mean.numpy())  
            test_epoch_log['vs_mse'].append(vs_mseBatch)
            test_epoch_log['vs_acc'].append(vs_accBatch)
            test_epoch_log['acc'].append(acc_batch)
            test_epoch_log['classify_loss'].append(loss_classify.numpy())  
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)    
    test_loss_mean = np.mean(test_loss)
    test_loss_regressor_mean = np.mean(test_loss_regressor)
    test_acc_mean = np.mean(test_acc)
    test_vsMse_mean = np.mean(test_vsMse)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_mean, test_loss_regressor_mean, test_acc_mean, test_vsMse_mean, test_vsAcc_mean

def performance_on_SynDataset_DualLearning_Classify(dataframe_prior, classes, bt_size, model_classify, model_vs, dis_pars, path_res, save_tuple, isTest):
                
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss, test_loss_vs, test_acc, test_vsMse, test_vsAcc = [], [], [], [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [], 'vs_loss': [], 'acc': [], 'vs_mse': [],  'vs_acc': [] }
   
    dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
    vs_SetTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)
    
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior, X_Vs = None, None, None, None
        if classes == 4:
            X, Y, X_prior, _, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
           X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars)        
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        
        X_classify = Convert_to_regressor_SynData(X, Y)
        xClassify_tensorBatch = tf.convert_to_tensor(X_classify , dtype = float)

        ##################### y_true + x -> vs #########################
        Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
        Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
        vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))
        vs_tensorBatch = tf.reshape(vs_tensorBatch, -1)
        
        vs_mseBatch = mean_squared_error(xVs_tensorBatch, vs_tensorBatch) 
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
        
        loss_classify1 = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            yi_probs = tf.gather(Vs_logits, [i]) # (1,13)
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            yiProbs_feaSet = tf.gather(tf.squeeze(yi_probs, axis = 0), more_than_one_vals)             
            vs_iSum =  tf.reduce_sum(yiProbs_feaSet)
            loss_classify1.append(-tf.math.log(vs_iSum))
        
        loss_classify1 = tf.convert_to_tensor(loss_classify1)
        loss_vs_mean = tf.reduce_mean(loss_classify1)              
        
        ##################### vs + x -> y #########################
        # vs_RandomSet = []
        # for i in range(xPrior_tensorBatch.get_shape()[0]):
        #     res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
        #     res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
        #     res = tf.concat((res_1, res_2), axis = 0)
        #     unique_res_vals, unique_idx = tf.unique(res)
        #     count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
        #                                                unique_idx,
        #                                                tf.shape(res)[0])
        #     more_than_one = tf.greater(count_res_unique, 1)
        #     more_than_one_idx = tf.squeeze(tf.where(more_than_one))
        #     more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
        #     feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
        #     feasible_set = tf.reshape(feasible_set, [-1])
            
        #     idxs = tf.range(tf.shape(feasible_set)[0])
        #     ridxs =  tf.random.shuffle(idxs)[:1] # :sample_num
        #     vs_random = tf.gather(feasible_set, ridxs)[0]
        #     vs_RandomSet.append(vs_random.numpy())
        
        # vs_Random_tensorBatch = tf.convert_to_tensor(vs_RandomSet)
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model_classify(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)   
        
        loss_classify = tf.reduce_mean(loss_B)
        acc_batch = accuracy_compute(np.array(y_tensorBatch), logits.numpy()) 
        
        # summary the results #           
        test_loss.append(loss_classify.numpy())        
        test_acc.append(acc_batch)               
        test_loss_vs.append(loss_vs_mean.numpy())
        test_vsMse.append(vs_mseBatch) 
        test_vsAcc.append(vs_accBatch)  
        if isTest:
            it, g = save_tuple[0], save_tuple[1]     
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['vs_loss'].append(loss_vs_mean.numpy())  
            test_epoch_log['vs_mse'].append(vs_mseBatch)
            test_epoch_log['vs_acc'].append(vs_accBatch)
            test_epoch_log['acc'].append(acc_batch)
            test_epoch_log['classify_loss'].append(loss_classify.numpy())  
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)    
    
    test_loss_mean = np.mean(test_loss)
    test_loss_vs_mean = np.mean(test_loss_vs)
    test_acc_mean = np.mean(test_acc)
    test_vsMse_mean = np.mean(test_vsMse)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_mean, test_loss_vs_mean, test_acc_mean, test_vsMse_mean, test_vsAcc_mean


# def performance_on_SynDataset_SingleLearning(dataframe_prior, classes, bt_size, model_regressor, dis_pars, path_res, save_tuple, isTest):
    
#     it, g = save_tuple[0], save_tuple[1]                 
#     batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
#     test_loss_regressor, test_vsMse, test_vsAcc = [], [], []
#     test_epoch_log = {'batch': [] , 'regressor_loss': [], 'vs_mse': [],  'vs_acc': [] }
#     for batch in range(len(batch_dataFrame)):
#         batch_idataFrame = batch_dataFrame[batch]

#         X, Y, X_prior, X_Vs = None, None, None, None
#         if classes == 4:
#             X, Y, X_prior, _, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
#         elif classes == 6:
#             X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
#         elif classes >= 15:
#            X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars)        
#         Y = to_categorical(Y, classes)   
#         X = X.astype(np.float64)
    
#         X_regressor = Convert_to_regressor_SynData(X, Y)
        
#         xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
#         xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
#         xRegressor_tensorBatch = tf.convert_to_tensor(X_regressor, dtype = float)
        
#         dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
#         vs_SetTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)
#         # VS LOSS #
#         vs_tensorBatch = model_regressor(tf.reshape(xRegressor_tensorBatch, [xRegressor_tensorBatch.get_shape()[0], 3, -1])) 
            
#         vs_tensorBatch_Corr = tf.abs(tf.subtract(tf.reshape(vs_tensorBatch,[-1, 1]), tf.reshape(vs_SetTensor, [1, -1])))
#         vs_tensorBatch_Corr = tf.reshape(vs_tensorBatch_Corr, [-1, vs_SetTensor.get_shape()[0]])
#         vs_idx = np.argmin(vs_tensorBatch_Corr, axis = 1)
#         vs_tensorBatch_Corr = tf.gather(vs_SetTensor, vs_idx.tolist())
        
#         vs_mseBatch = mean_squared_error(xVs_tensorBatch, vs_tensorBatch_Corr) 
#         differ_cor1 = tf.subtract(tf.reshape(vs_tensorBatch_Corr, [-1]), tf.reshape(xPrior_tensorBatch[:, 0], [-1]))
#         differ_cor2 = tf.subtract(tf.reshape(xPrior_tensorBatch[:, 1], [-1]), tf.reshape(vs_tensorBatch_Corr, [-1]))       
#         predTrue_num = 0
#         for i in range(xPrior_tensorBatch.get_shape()[0]):
#             if differ_cor1[i] >= 0 and differ_cor2[i] >= 0:
#                 predTrue_num = predTrue_num + 1
#         vs_accBatch = predTrue_num/vs_tensorBatch_Corr.get_shape()[0]
        
#         ## update its gradients ##           
#         loss_regressor = []   
#         differ_1 = tf.subtract(tf.reshape(vs_tensorBatch, [-1]), tf.reshape(xPrior_tensorBatch[:, 0], [-1]))
#         differ_2 = tf.subtract(tf.reshape(xPrior_tensorBatch[:, 1], [-1]), tf.reshape(vs_tensorBatch, [-1]))       
#         for i in range(xPrior_tensorBatch.get_shape()[0]):
#             if differ_1[i] < 0:
#                 loss_regressor.append(differ_1[i] ** 2)
#             elif differ_2[i] < 0:
#                 loss_regressor.append(differ_2[i] ** 2)
#             else:
#                 loss_regressor.append(0)

#         loss_regressor = tf.convert_to_tensor(loss_regressor)
#         loss_regressor_mean = tf.reduce_mean(loss_regressor)
                  
#         # summary the results #                        
#         test_loss_regressor.append(loss_regressor_mean.numpy())
#         test_vsMse.append(vs_mseBatch) 
#         test_vsAcc.append(vs_accBatch)  
#         if isTest:
#             test_epoch_log['batch'].append(batch + 1)
#             test_epoch_log['regressor_loss'].append(loss_regressor_mean.numpy())  
#             test_epoch_log['vs_mse'].append(vs_mseBatch)
#             test_epoch_log['vs_acc'].append(vs_accBatch)
#             pd.DataFrame(test_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)    

#     test_loss_regressor_mean = np.mean(test_loss_regressor)
#     test_vsMse_mean = np.mean(test_vsMse)
#     test_vsAcc_mean = np.mean(test_vsAcc)

#     return test_loss_regressor_mean, test_vsMse_mean, test_vsAcc_mean

def performance_on_SynDataset_SingleLearning_Classify(dataframe_prior, classes, bt_size, model_C, dis_pars, path_res, save_tuple, isTest):
    
    it, g = save_tuple[0], save_tuple[1]                 
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss_c, test_vsMse, test_vsAcc = [], [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [], 'vs_mse': [],  'vs_acc': [] }
    
    dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
    vs_SetTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)
    
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior, X_Vs = None, None, None, None
        if classes == 4:
            X, Y, X_prior, _, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
           X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars)        
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
    
        X_classify = Convert_to_regressor_SynData(X, Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        xClassify_tensorBatch = tf.convert_to_tensor(X_classify, dtype = float)
        
        X_Vs_Corr = copy.deepcopy(X_Vs)
        for v in range(vs_SetTensor.get_shape()[0]):
            X_Vs_Corr[np.where(X_Vs_Corr == vs_SetTensor[v].numpy())] = v        
        Y_Classify = to_categorical(X_Vs_Corr, vs_SetTensor.get_shape()[0])       
        # Y_Classify_tensorBatch = tf.convert_to_tensor(Y_trainClassify, dtype = float)       
        # Y_Classify_trueIdx_tensorBatch = tf.math.argmax(Y_Classify_tensorBatch, axis = 1) # for check

        Vs_logits = model_C(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
        Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
        vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))
        vs_tensorBatch = tf.reshape(vs_tensorBatch, -1)
        
        vs_mseBatch = mean_squared_error(xVs_tensorBatch, vs_tensorBatch) 
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
        
        loss_classify = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            yi_probs = Vs_logits[i]
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            yiProbs_feaSet = tf.gather(yi_probs, more_than_one_vals)             
            vs_iSum =  tf.reduce_sum(yiProbs_feaSet)
            loss_classify.append(-math.log(vs_iSum))
            
        loss_classify = tf.convert_to_tensor(loss_classify)
        loss_classify_mean = tf.reduce_mean(loss_classify)
        
        test_loss_c.append(loss_classify_mean.numpy())
        test_vsMse.append(vs_mseBatch) 
        test_vsAcc.append(vs_accBatch) 
        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['classify_loss'].append(loss_classify_mean.numpy())  
            test_epoch_log['vs_mse'].append(vs_mseBatch)
            test_epoch_log['vs_acc'].append(vs_accBatch)
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)    

    test_loss_c_mean = np.mean(test_loss_c)
    test_vsMse_mean = np.mean(test_vsMse)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_c_mean, test_vsMse_mean, test_vsAcc_mean

def performance_on_SynDataset_SingleLearning_Classify_Bayesian(dataframe_prior, classes, bt_size, model_Classify, model_vs, dis_pars, path_res, save_tuple, isTest):
    
    it, g = save_tuple[0], save_tuple[1]                 
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss_c, test_vsMse, test_vsAcc = [], [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [], 'vs_mse': [],  'vs_acc': [] }
    
    dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
    vs_SetTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)
    
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
        
        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
           X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars)        
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        Y_set = obtain_y_set(classes)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        X_classify = Convert_to_regressor_SynData(X, Y)
        xClassify_tensorBatch = tf.convert_to_tensor(X_classify , dtype = float)

        # ##################### infer vs based on action prediction model #########################
        # dis_type = 'Uniform'
        # syn_tag = True
        # vs_tensorBatch, _, _ = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model_Classify)         
        Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
        # Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
        # vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))

        ############################# new added #######################
        X_Classify = []
        for j in range(len(Y_set)):
            Yj = [Y_set[j]] * X.shape[0]
            X_Yj_Classify = Convert_to_regressor_SynData(X, Yj)
            X_Classify.append(X_Yj_Classify)
        
        X_Classify = tf.convert_to_tensor(X_Classify)        
        # print(X_Classify)    # (4, 26, 18)   
        X_Classify_reshape1 = tf.keras.layers.concatenate([X_Classify[0,:,:], X_Classify[1,:,:]], axis = -1)
        X_Classify_reshape2 = tf.keras.layers.concatenate([X_Classify[2,:,:], X_Classify[3,:,:]], axis = -1)
        X_Classify_reshape = tf.keras.layers.concatenate([X_Classify_reshape1, X_Classify_reshape2], axis = -1)
        if classes == 6:
              X_Classify_reshape3 = tf.keras.layers.concatenate([X_Classify[4,:,:], X_Classify[5,:,:]], axis = -1)
              X_Classify_reshape = tf.keras.layers.concatenate([X_Classify_reshape, X_Classify_reshape3], axis = -1)  
        X_Classify_reshape = tf.reshape(X_Classify_reshape, [X_Classify_reshape.get_shape()[0], classes, -1])
        # print(X_Classify_reshape.get_shape())  ## (26, 4, 18) 
        Vs_logitSet = []
        for i in range(X_Classify_reshape.get_shape()[0]):
            XiClassify_tensorBatch = X_Classify_reshape[i, :, :]
            # print(XiClassify_tensorBatch.get_shape())#(classes, 18)
            Vs_logits_i = model_vs(tf.reshape(XiClassify_tensorBatch, [XiClassify_tensorBatch.get_shape()[0], 3, -1])) 
            Vs_logitSet.append(Vs_logits_i)
  
        ##################### y_true + x -> vs #########################
        visited_vsID, Vs_inferred_dict = [], {} 
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            vs_ID = x_vsID_Batch[i]
            if vs_ID.numpy() not in visited_vsID:
                visited_vsID.append(vs_ID.numpy())                     
                multi_res = tf.where(tf.equal(vs_ID, x_vsID_Batch))
                VsLogits = None
                if multi_res.get_shape()[0] == 1: # only-one row                              
                    data_single = tf.gather(xClassify_tensorBatch, tf.reshape(multi_res, [-1]), axis = 0)                            
                    Vs_prob = model_vs(tf.reshape(data_single, [data_single.get_shape()[0], 3, -1])) 
                    Vs_prob = tf.squeeze(Vs_prob)  
                    demo_d = Vs_logitSet[tf.reshape(multi_res, [-1])[0].numpy()]
                    demo_d_Sum = tf.math.reduce_sum(demo_d, axis = 0) # the product of each column
                    VsLogits = Vs_prob/demo_d_Sum                                
                elif multi_res.get_shape()[0] > 1: # more than one row exist!
                    data_multi = tf.gather(xClassify_tensorBatch, tf.reshape(multi_res, [-1]), axis = 0)
                    vs_multi_logits = model_vs(tf.reshape(data_multi, [data_multi.get_shape()[0], 3, -1]))
                    # (rows, 13): vs_multi_logits
                    result = []
                    for d in range(data_multi.get_shape()[0]):
                        idx = tf.reshape(multi_res, [-1])[d].numpy()
                        numo_d = vs_multi_logits[d]
                        demo_d = Vs_logitSet[idx]
                        demo_d_Sum = tf.math.reduce_sum(demo_d, axis = 0) # the product of each column
                        res = numo_d/demo_d_Sum
                        result.append(res)  
                    result_tensor = tf.convert_to_tensor(result)
                    VsLogits = tf.math.reduce_prod(result_tensor, axis = 0) # the product of each column             
                    assert(vs_multi_logits.get_shape()[0] == multi_res.get_shape()[0])

                vs_Pred_idx = tf.math.argmax(VsLogits)
                vs_i = tf.gather(vs_SetTensor, tf.squeeze(vs_Pred_idx))               
                for d in range(multi_res.get_shape()[0]):
                    multi_resReshape = tf.reshape(multi_res, [-1])
                    Vs_inferred_dict[str(multi_resReshape[d].numpy())] = vs_i
        
        vs_TensorBatch = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            vs_i = Vs_inferred_dict[str(i)]   
            vs_TensorBatch.append(vs_i.numpy())
        #print(vs_TensorBatch)

        # print(vs_TensorBatch)
        vs_TensorBatch = tf.convert_to_tensor(vs_TensorBatch)
        vs_tensorBatch = vs_TensorBatch
        vs_mseBatch = mean_squared_error(xVs_tensorBatch, vs_tensorBatch) 
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
        
        loss_classify = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            yi_probs = Vs_logits[i]
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            yiProbs_feaSet = tf.gather(yi_probs, more_than_one_vals)             
            vs_iSum =  tf.reduce_sum(yiProbs_feaSet)
            loss_classify.append(-math.log(vs_iSum))
            
        loss_classify = tf.convert_to_tensor(loss_classify)
        loss_classify_mean = tf.reduce_mean(loss_classify)
        
        test_loss_c.append(loss_classify_mean.numpy())
        test_vsMse.append(vs_mseBatch) 
        test_vsAcc.append(vs_accBatch) 
        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['classify_loss'].append(loss_classify_mean.numpy())  
            test_epoch_log['vs_mse'].append(vs_mseBatch)
            test_epoch_log['vs_acc'].append(vs_accBatch)
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)    

    test_loss_c_mean = np.mean(test_loss_c)
    test_vsMse_mean = np.mean(test_vsMse)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_c_mean, test_vsMse_mean, test_vsAcc_mean


def recompute_vs_metrics(xVs_tensorBatch, vs_tensorBatch, rA_tensorBatch, x_vsID_Batch):
    
    vsID_unique_val, _ = tf.unique(x_vsID_Batch)
    vs_unique_pred, vs_unique_true, rA_unique = [], [], []
    for v in range(vsID_unique_val.get_shape()[0]):
        vsID = vsID_unique_val[v]
        idx = tf.where(tf.equal(x_vsID_Batch, vsID))
        vs_pred, _ = tf.unique(tf.gather_nd(vs_tensorBatch, idx))
        vs_true, _ = tf.unique(tf.gather_nd(xVs_tensorBatch, idx))
        rA, _ = tf.unique(tf.gather_nd(rA_tensorBatch, idx))
        vs_unique_pred.append(vs_pred)
        vs_unique_true.append(vs_true)
        rA_unique.append(rA)
        if vs_pred.get_shape()[0] != 1 or vs_true.get_shape()[0] != 1 or rA.get_shape()[0] != 1:
            assert(1 == 0)
    vs_unique_pred = tf.convert_to_tensor(vs_unique_pred)
    vs_unique_true = tf.convert_to_tensor(vs_unique_true)
    rA_unique = tf.convert_to_tensor(rA_unique)
    
    vs_mseBatch = mean_squared_error(vs_unique_true, vs_unique_pred)
    
    predTrue_num = len(np.where(rA_unique.numpy() > 0)[0].tolist())
    vs_accBatch = predTrue_num/vsID_unique_val.get_shape()[0]
    
    return vs_mseBatch, vs_accBatch

def performance_on_SynDataset_baseline(dataframe_prior, classes, bt_size, model, isTest, path_res, save_tuple):
    
    loss, acc_total = [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)         
        elif classes >= 15:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars) 
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        # xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        # x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
            
        # loss B  computation#  
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)      
        loss_total = tf.reduce_mean(loss_B)        
        
        loss_mean = loss_total.numpy()
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
        
        loss.append(loss_mean)
        acc_total.append(acc_batch)
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['acc'].append(acc_batch)
            test_logs['total_loss'].append(loss_mean) 
            it, g = save_tuple[0], save_tuple[1]  
            if batch % 10 == 0 or batch == len(batch_dataFrame)-1:
                print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    
    return loss, acc

def performance_on_SynDataset_learnY(dataframe_prior, classes, dis_pars, bt_size, model, isTest, path_res, save_tuple):
    
    loss_all, acc_total = [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
        X, Y, X_prior, X_Vs = None, None, None, None
        if classes == 4:
             X, Y, X_prior, _, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
             X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
            X, Y, X_prior, _, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars) 
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        # xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        # x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
            
        # loss B  computation#  
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)      
        loss_total = tf.reduce_mean(loss_B)        
        
        loss_mean = loss_total.numpy()
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
        
        loss_all.append(loss_mean)
        acc_total.append(acc_batch)
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['acc'].append(acc_batch)
            test_logs['total_loss'].append(loss_mean) 
            it, g = save_tuple[0], save_tuple[1]  
            if batch % 10 == 0 or batch == len(batch_dataFrame)-1:
                print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)       
    
    loss = np.mean(loss_all)
    acc = np.mean(acc_total)
    
    return loss, acc

def obtain_y_set(classes):
    Y_set = []
    Y_temp = np.zeros((classes,))
    for i  in range(classes):
        Y_temp_i = copy.deepcopy(Y_temp)
        Y_temp_i[i] = 1
        Y_set.append(Y_temp_i)
    return Y_set

def performance_on_SynDataset_DualLearning_Classify_Constraints(dataframe_prior, classes, bt_size, model_classify, model_vs, dis_pars, path_res, save_tuple, isTest):
    
    it, g = save_tuple[0], save_tuple[1]                 
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss, test_loss_vs, test_acc, test_vsMse, test_vsAcc = [], [], [], [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [], 'vs_loss': [], 'acc': [], 'vs_mse': [],  'vs_acc': [] }
   
    dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
    vs_SetTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)
    
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame) 
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
           X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars)
        
        Y_set = obtain_y_set(classes)
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        X_classify = Convert_to_regressor_SynData(X, Y)
        xClassify_tensorBatch = tf.convert_to_tensor(X_classify , dtype = float)
        ############################# new added #######################
        X_Classify = []
        for j in range(len(Y_set)):
            Yj = [Y_set[j]] * X.shape[0]
            X_Yj_Classify = Convert_to_regressor_SynData(X, Yj)
            X_Classify.append(X_Yj_Classify)
        
        X_Classify = tf.convert_to_tensor(X_Classify)        
        # print(X_Classify)    # (4, 26, 18)   
        X_Classify_reshape1 = tf.keras.layers.concatenate([X_Classify[0,:,:], X_Classify[1,:,:]], axis = -1)
        X_Classify_reshape2 = tf.keras.layers.concatenate([X_Classify[2,:,:], X_Classify[3,:,:]], axis = -1)
        X_Classify_reshape = tf.keras.layers.concatenate([X_Classify_reshape1, X_Classify_reshape2], axis = -1)
        if classes == 6:
              X_Classify_reshape3 = tf.keras.layers.concatenate([X_Classify[4,:,:], X_Classify[5,:,:]], axis = -1)
              X_Classify_reshape = tf.keras.layers.concatenate([X_Classify_reshape, X_Classify_reshape3], axis = -1)  
        X_Classify_reshape = tf.reshape(X_Classify_reshape, [X_Classify_reshape.get_shape()[0], classes, -1])
        # print(X_Classify_reshape.get_shape())  ## (26, 4, 18) 
        Vs_logitSet = []
        for i in range(X_Classify_reshape.get_shape()[0]):
            XiClassify_tensorBatch = X_Classify_reshape[i, :, :]
            # print(XiClassify_tensorBatch.get_shape())#(classes, 18)
            Vs_logits_i = model_vs(tf.reshape(XiClassify_tensorBatch, [XiClassify_tensorBatch.get_shape()[0], 3, -1])) 
            Vs_logitSet.append(Vs_logits_i)
        assert(xClassify_tensorBatch.get_shape()[0] == X_Classify_reshape.get_shape()[0])
        ##################### y_true + x -> vs #########################
        visited_vsID, Vs_inferred_dict = [], {} 
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            vs_ID = x_vsID_Batch[i]
            if vs_ID.numpy() not in visited_vsID:
                visited_vsID.append(vs_ID.numpy())                     
                multi_res = tf.where(tf.equal(vs_ID, x_vsID_Batch))
                VsLogits = None
                if multi_res.get_shape()[0] == 1: # only-one row                              
                    data_single = tf.gather(xClassify_tensorBatch, tf.reshape(multi_res, [-1]), axis = 0)                            
                    Vs_prob = model_vs(tf.reshape(data_single, [data_single.get_shape()[0], 3, -1])) 
                    Vs_prob = tf.squeeze(Vs_prob)  
                    demo_d = Vs_logitSet[tf.reshape(multi_res, [-1])[0].numpy()]
                    demo_d_Sum = tf.math.reduce_sum(demo_d, axis = 0) # the sum of each column
                    VsLogits = Vs_prob/demo_d_Sum                                
                elif multi_res.get_shape()[0] > 1: # more than one row exist!
                    data_multi = tf.gather(xClassify_tensorBatch, tf.reshape(multi_res, [-1]), axis = 0)
                    vs_multi_logits = model_vs(tf.reshape(data_multi, [data_multi.get_shape()[0], 3, -1]))
                    # (rows, 13): vs_multi_logits
                    result = []
                    for d in range(data_multi.get_shape()[0]):
                        idx = tf.reshape(multi_res, [-1])[d].numpy()
                        numo_d = vs_multi_logits[d]
                        demo_d = Vs_logitSet[idx]
                        demo_d_Sum = tf.math.reduce_sum(demo_d, axis = 0) # the sum of each column
                        res = numo_d/demo_d_Sum
                        result.append(res)  
                    result_tensor = tf.convert_to_tensor(result)
                    VsLogits = tf.math.reduce_prod(result_tensor, axis = 0) # the product of each column             
                    assert(vs_multi_logits.get_shape()[0] == multi_res.get_shape()[0])

                vs_Pred_idx = tf.math.argmax(VsLogits)
                vs_i = tf.gather(vs_SetTensor, tf.squeeze(vs_Pred_idx))               
                for d in range(multi_res.get_shape()[0]):
                    multi_resReshape = tf.reshape(multi_res, [-1])
                    Vs_inferred_dict[str(multi_resReshape[d].numpy())] = vs_i
        
        vs_TensorBatch = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            vs_i = Vs_inferred_dict[str(i)]   
            vs_TensorBatch.append(vs_i.numpy())
        vs_tensorBatch = tf.convert_to_tensor(vs_TensorBatch)
        
        Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
        # Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
        # vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))
        vs_mseBatch = mean_squared_error(xVs_tensorBatch, vs_tensorBatch) 
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
        
        loss_classify1 = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            yi_probs = tf.gather(Vs_logits, [i]) # (1,13)
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            yiProbs_feaSet = tf.gather(tf.squeeze(yi_probs, axis = 0), more_than_one_vals)             
            vs_iSum =  tf.reduce_sum(yiProbs_feaSet)
            loss_classify1.append(-tf.math.log(vs_iSum))
        
        loss_classify1 = tf.convert_to_tensor(loss_classify1)
        loss_vs_mean = tf.reduce_mean(loss_classify1)              
        
        ##################### vs + x -> y #########################
        # vs_RandomSet = []
        # for i in range(xPrior_tensorBatch.get_shape()[0]):
        #     res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
        #     res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
        #     res = tf.concat((res_1, res_2), axis = 0)
        #     unique_res_vals, unique_idx = tf.unique(res)
        #     count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
        #                                                unique_idx,
        #                                                tf.shape(res)[0])
        #     more_than_one = tf.greater(count_res_unique, 1)
        #     more_than_one_idx = tf.squeeze(tf.where(more_than_one))
        #     more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
        #     feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
        #     feasible_set = tf.reshape(feasible_set, [-1])
            
        #     idxs = tf.range(tf.shape(feasible_set)[0])
        #     ridxs =  tf.random.shuffle(idxs)[:1] # :sample_num
        #     vs_random = tf.gather(feasible_set, ridxs)[0]
        #     vs_RandomSet.append(vs_random.numpy())
        
        # vs_Random_tensorBatch = tf.convert_to_tensor(vs_RandomSet)
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model_classify(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)   
        
        loss_classify = tf.reduce_mean(loss_B)
        acc_batch = accuracy_compute(np.array(y_tensorBatch), logits.numpy()) 
        
        # summary the results #           
        test_loss.append(loss_classify.numpy())        
        test_acc.append(acc_batch)               
        test_loss_vs.append(loss_vs_mean.numpy())
        test_vsMse.append(vs_mseBatch) 
        test_vsAcc.append(vs_accBatch)  
        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['vs_loss'].append(loss_vs_mean.numpy())  
            test_epoch_log['vs_mse'].append(vs_mseBatch)
            test_epoch_log['vs_acc'].append(vs_accBatch)
            test_epoch_log['acc'].append(acc_batch)
            test_epoch_log['classify_loss'].append(loss_classify.numpy())  
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)    
    
    test_loss_mean = np.mean(test_loss)
    test_loss_vs_mean = np.mean(test_loss_vs)
    test_acc_mean = np.mean(test_acc)
    test_vsMse_mean = np.mean(test_vsMse)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_mean, test_loss_vs_mean, test_acc_mean, test_vsMse_mean, test_vsAcc_mean

def performance_on_SynDataset_DualLearning_Classify_Bayesian(dataframe_prior, classes, bt_size, model_classify, model_vs, dis_pars, path_res, save_tuple, isTest):
    
    it, g = save_tuple[0], save_tuple[1]                 
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss, test_loss_vs, test_acc, test_vsMse, test_vsAcc = [], [], [], [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [], 'vs_loss': [], 'acc': [], 'vs_mse': [],  'vs_acc': [] }
   
    dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
    vs_SetTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)
    
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior, X_vsID, X_Vs = None, None, None, None, None
        if classes == 4:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData(batch_idataFrame)  
        elif classes == 6:
            X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_Class6(batch_idataFrame)
        elif classes >= 15:
           X, Y, X_prior, X_vsID, X_Vs = Convert_to_training_SynData_MultiClass(batch_idataFrame, dis_pars)        
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xVs_tensorBatch = tf.convert_to_tensor(X_Vs, dtype= float)   
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        X_classify = Convert_to_regressor_SynData(X, Y)
        xClassify_tensorBatch = tf.convert_to_tensor(X_classify , dtype = float)

        ##################### y_true + x -> vs #########################
        dis_type = 'Uniform'
        syn_tag = True
        vs_tensorBatch, _, _ = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model_classify)
        Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
        # Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
        # vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))
        vs_mseBatch = mean_squared_error(xVs_tensorBatch, vs_tensorBatch) 
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
        
        loss_classify1 = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            yi_probs = tf.gather(Vs_logits, [i]) # (1,13)
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            yiProbs_feaSet = tf.gather(tf.squeeze(yi_probs, axis = 0), more_than_one_vals)             
            vs_iSum =  tf.reduce_sum(yiProbs_feaSet)
            loss_classify1.append(-tf.math.log(vs_iSum))
        
        loss_classify1 = tf.convert_to_tensor(loss_classify1)
        loss_vs_mean = tf.reduce_mean(loss_classify1)              
        
        ##################### vs + x -> y #########################
        # vs_RandomSet = []
        # for i in range(xPrior_tensorBatch.get_shape()[0]):
        #     res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
        #     res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
        #     res = tf.concat((res_1, res_2), axis = 0)
        #     unique_res_vals, unique_idx = tf.unique(res)
        #     count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
        #                                                unique_idx,
        #                                                tf.shape(res)[0])
        #     more_than_one = tf.greater(count_res_unique, 1)
        #     more_than_one_idx = tf.squeeze(tf.where(more_than_one))
        #     more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
        #     feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
        #     feasible_set = tf.reshape(feasible_set, [-1])
            
        #     idxs = tf.range(tf.shape(feasible_set)[0])
        #     ridxs =  tf.random.shuffle(idxs)[:1] # :sample_num
        #     vs_random = tf.gather(feasible_set, ridxs)[0]
        #     vs_RandomSet.append(vs_random.numpy())
        
        # vs_Random_tensorBatch = tf.convert_to_tensor(vs_RandomSet)
        x_exBatch_Tensor = insert_VS_into_X(xVs_tensorBatch, x_tensorBatch)
        
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model_classify(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)   
        
        loss_classify = tf.reduce_mean(loss_B)
        acc_batch = accuracy_compute(np.array(y_tensorBatch), logits.numpy()) 
        
        # summary the results #           
        test_loss.append(loss_classify.numpy())        
        test_acc.append(acc_batch)               
        test_loss_vs.append(loss_vs_mean.numpy())
        test_vsMse.append(vs_mseBatch) 
        test_vsAcc.append(vs_accBatch)  
        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['vs_loss'].append(loss_vs_mean.numpy())  
            test_epoch_log['vs_mse'].append(vs_mseBatch)
            test_epoch_log['vs_acc'].append(vs_accBatch)
            test_epoch_log['acc'].append(acc_batch)
            test_epoch_log['classify_loss'].append(loss_classify.numpy())  
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_testData_batchAcc.csv', index = False)    
    
    test_loss_mean = np.mean(test_loss)
    test_loss_vs_mean = np.mean(test_loss_vs)
    test_acc_mean = np.mean(test_acc)
    test_vsMse_mean = np.mean(test_vsMse)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_mean, test_loss_vs_mean, test_acc_mean, test_vsMse_mean, test_vsAcc_mean

################################# real data metric ###############################################

def performance_on_RealDataset_learnY(dataframe_prior, classes, bt_size, model, model_vs, dis_pars, isTest, path_res):
    
    loss, acc_total = [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    vs_num = dis_pars['vs_num'] 
    # vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
    Vs = [i for i in range(10, 100, 2)]
    del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
    Vs.append(100)
    vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
        X, Y, X_prior = None, None, None
        if classes == 3:
            X, Y, X_prior, _ = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, _ = Convert_to_training_RealData_Classes5(batch_idataFrame) 
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)

        X_Classify = Convert_to_regressor_SynData(X, Y)                  
        xClassify_tensorBatch = tf.convert_to_tensor(X_Classify, dtype = float) 
       
        Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
        Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
        vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))
            
        # loss B  computation#  
        x_exBatch_Tensor = insert_VS_into_X(vs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)      
        loss_total = tf.reduce_mean(loss_B)        
        
        loss_mean = loss_total.numpy()
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
        
        loss.append(loss_mean)
        acc_total.append(acc_batch)
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['acc'].append(acc_batch)
            test_logs['total_loss'].append(loss_mean) 
            if batch % 10 == 0 or batch == len(batch_dataFrame)-1:
                print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    
    return loss, acc


def performance_on_RealDataset_Multirows_V0_alphaLog(dataframe_prior, alpha, classes, bt_size, syn_tag, dis_type, dis_pars, model, isTest, path_res):
    
    loss_total, acc_total,  vsAcc_total = [], [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [], 'vs_acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID = None, None, None, None
        if classes == 3:
            X, Y, X_prior, X_vsID = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, X_vsID = Convert_to_training_RealData_Classes5(batch_idataFrame)
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)  
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
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
        vsAcc_batch = predTrue_num/vs_tensorBatch.get_shape()[0]
         
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
        
        real_loss_A = []
        for i in range(x_extends_Tensor.get_shape()[0]):
            xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], 3, -1]))
            yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)

            yi_probs = tf.clip_by_value(yi_probs, 1e-10, 1)
            
            yiProbs_onVs_pVs = tf.math.multiply(yi_probs, vs_probs)
            # print(yiProbs_onVs_pVs.get_shape()) #(13,1)
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

            # loss_res = sum_Numo/sum_Denom
            # if loss_res == 0:
            #     loss_res = 1e-323
            # real_loss_A_i = - math.log(loss_res) #sum_Numo/sum_Denom
            real_loss_A_i = tf.math.log(sum_Denom) - tf.math.log(sum_Numo) 
            real_loss_A.append(real_loss_A_i)

        loss_total_real = alpha * tf.reduce_mean(real_loss_A) + (1-alpha) * tf.reduce_mean(loss_B)        
        
        loss_mean = loss_total_real.numpy()
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
        
        loss_total.append(loss_mean)
        acc_total.append(acc_batch)
        vsAcc_total.append(vsAcc_batch)  
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['acc'].append(acc_batch)
            test_logs['vs_acc'].append(vsAcc_batch)
            test_logs['total_loss'].append(loss_mean)  
            if batch % 10 == 0 or batch == len(batch_dataFrame)-1:
                print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, vsAcc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    vsAcc_mean = np.mean(vsAcc_total)
    
    return loss, acc, vsAcc_mean

def performance_on_RealDataset_Multirows_V0_alphaLog_transformer(dataframe_prior, alpha, classes, bt_size, syn_tag, dis_type, dis_pars, model, isTest, path_res):
    
    loss_total, acc_total,  vsAcc_total = [], [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [], 'vs_acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID = None, None, None, None
        if classes == 3:
            X, Y, X_prior, X_vsID = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, X_vsID = Convert_to_training_RealData_Classes5(batch_idataFrame)
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)  
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        # infer vs using threads from the same seller and the same item #
        vs_tensorBatch, prior_tensorBatch, post_tensorBatch = Vs_inference_with_MultiRows_Tensor_transformer(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model)
             
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
        vsAcc_batch = predTrue_num/vs_tensorBatch.get_shape()[0]
         
        # loss B  computation#  
        x_exBatch_Tensor = insert_VS_into_X(vs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        # logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], -1, 1]))                   
        logits_org = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], -1, 1]))      
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
        
        real_loss_A = []
        for i in range(x_extends_Tensor.get_shape()[0]):
            xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], -1, 1]))
            yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)
            
            yi_probs = tf.clip_by_value(yi_probs, 1e-10, 1)
            
            yiProbs_onVs_pVs = tf.math.multiply(yi_probs, vs_probs)
            # print(yiProbs_onVs_pVs.get_shape()) #(13,1)
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

            # loss_res = sum_Numo/sum_Denom
            # if loss_res == 0:
            #     loss_res = 1e-323
            # real_loss_A_i = - math.log(loss_res) #sum_Numo/sum_Denom
            real_loss_A_i = tf.math.log(sum_Denom) - tf.math.log(sum_Numo) 
            real_loss_A.append(real_loss_A_i)

        loss_total_real = alpha * tf.reduce_mean(real_loss_A) + (1-alpha) * tf.reduce_mean(loss_B)        
        
        loss_mean = loss_total_real.numpy()
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
        
        loss_total.append(loss_mean)
        acc_total.append(acc_batch)
        vsAcc_total.append(vsAcc_batch)  
        
        if isTest:
            test_logs['batch'].append(batch + 1)
            test_logs['acc'].append(acc_batch)
            test_logs['vs_acc'].append(vsAcc_batch)
            test_logs['total_loss'].append(loss_mean)  
            if batch % 10 == 0 or batch == len(batch_dataFrame)-1:
                print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, vsAcc_batch), end = "\n")
                pd.DataFrame(test_logs).to_csv(path_res +'data_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    vsAcc_mean = np.mean(vsAcc_total)
    
    return loss, acc, vsAcc_mean

def performance_on_RealDataset_Multirows_V0_alphaLog_revised(dataframe_prior, alpha, classes, bt_size, syn_tag, dis_type, dis_pars, model, isTest, path_res):
    
    loss_total, acc_total,  vsAcc_total = [], [], []
    test_logs = {'batch': [] , 'total_loss': [], 'acc': [], 'vs_acc': [] }
    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)        
    
    for batch in range(len(batch_dataFrame)):
        # determine starting and ending slice indexes for the current
        batch_idataFrame = batch_dataFrame[batch]
        
        ## this is for only one bargaining thread case, for those with multiple threads should re-implement it ##
               
        X, Y, X_prior, X_vsID = None, None, None, None
        if classes == 3:
            X, Y, X_prior, X_vsID = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, X_vsID = Convert_to_training_RealData_Classes5(batch_idataFrame)
            
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)  
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
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
        vsAcc_batch = predTrue_num/vs_tensorBatch.get_shape()[0]
         
        # loss B  computation#  
        x_exBatch_Tensor = insert_VS_into_X(vs_tensorBatch, x_tensorBatch)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
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
        
        real_loss_A, visited_vsID = [], []  
        for i in range(x_extends_Tensor.get_shape()[0]):
            vs_ID = x_vsID_Batch[i]
            if vs_ID.numpy() not in visited_vsID:
                visited_vsID.append(vs_ID.numpy())                     
                multi_res = tf.where(tf.equal(vs_ID, x_vsID_Batch))                                 
                
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
                
                data_multi = tf.gather(x_extends_Tensor, tf.reshape(multi_res, [-1]), axis = 0)
                y_trueIdx_new = tf.gather(y_trueIdx_batchTensor, tf.reshape(multi_res, [-1]), axis = 0)
                
                sum_Numo = tf.convert_to_tensor(0.)
                yi_probs_temp = tf.convert_to_tensor(0.)
                for d in range(data_multi.get_shape()[0]):
                    yi_d_logits = model(tf.reshape(data_multi[d,:,:], [data_multi.get_shape()[1], 3, -1])) #(13,4)
                    yi_d_true = tf.gather(y_trueIdx_new, d, axis = 0) 
                    yi_d_probs = tf.gather(yi_d_logits, yi_d_true, axis = 1) #(13,)   
                    yi_d_probs = tf.reshape(yi_d_probs, [-1, 1])

                    ## the second part for agent A
                    if d == 0:
                        yi_probs_temp = yi_d_probs
                    else:
                        yi_probs_temp = tf.concat([yi_probs_temp, yi_d_probs], axis = 1) # (13,9)

                    yiProbs_onVs_pVs = tf.math.multiply(yi_d_probs, vs_probs)
                    yiProbs_onVs_pVs_feaSet = tf.gather(tf.squeeze(yiProbs_onVs_pVs, axis = 1), more_than_one_vals)
                    sum_Numo_d = tf.reduce_sum(yiProbs_onVs_pVs_feaSet)    
                    sum_Numo = sum_Numo + tf.math.log(sum_Numo_d)
  
                yi_probs = tf.math.reduce_prod(yi_probs_temp, axis = 1) # the product of each row   
                yi_probs = tf.reshape(yi_probs, [-1, 1])            
                sum_Denom = tf.reduce_sum(tf.math.multiply(yi_probs, vs_probs))   
                assert(yi_probs_temp.get_shape()[1] == multi_res.get_shape()[0])
        
                log_pA_i = sum_Numo - tf.math.log(sum_Denom)        
                real_loss_A.append(-log_pA_i/data_multi.get_shape()[0]) 
                    
        loss_total_real = alpha * tf.reduce_mean(real_loss_A) + (1-alpha) * tf.reduce_mean(loss_B)        
        
        loss_mean = loss_total_real.numpy()
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy())) 
        
        loss_total.append(loss_mean)
        acc_total.append(acc_batch)
        vsAcc_total.append(vsAcc_batch)  
        
        # if isTest:
        #     test_logs['batch'].append(batch + 1)
        #     test_logs['acc'].append(acc_batch)
        #     test_logs['vs_acc'].append(vsAcc_batch)
        #     test_logs['total_loss'].append(loss_mean)  
        #     if batch % 10 == 0 or batch == len(batch_dataFrame)-1:
        #         print("[INFO] test dataset batch {}/{}, total_loss: {:.5f}, accuracy: {:.5f}, vs_acc: {:.5f} ".format(batch+1, len(batch_dataFrame), loss_mean, acc_batch, vsAcc_batch), end = "\n")
        #         pd.DataFrame(test_logs).to_csv(path_res +'data_testData_batchAcc.csv', index = False)       
    loss = np.mean(loss_total)
    acc = np.mean(acc_total)
    vsAcc_mean = np.mean(vsAcc_total)
    
    return loss, acc, vsAcc_mean

def performance_on_RealDataset_DualLearning(dataframe_prior, classes, bt_size, model_classify, model_regressor, dis_pars, path_res, isTest):
              
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss, test_loss_regressor, test_acc, test_vsAcc = [], [], [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [], 'regressor_loss': [], 'acc': [], 'vs_acc': [] }
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior = None, None, None
        if classes == 3:
            X, Y, X_prior, _ = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, _ = Convert_to_training_RealData_Classes5(batch_idataFrame)       
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)  

        vs_num = dis_pars['vs_num'] 
        Vs = [i for i in range(10, 100, 2)]
        del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
        Vs.append(100)
        vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
        # vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
        
        X_regressor = Convert_to_regressor_SynData(X, Y)
        xRegressor_tensorBatch = tf.convert_to_tensor(X_regressor, dtype = float)

        ##################### y_true + x -> vs #########################
        vs_tensorBatch = model_regressor(tf.reshape(xRegressor_tensorBatch, [xRegressor_tensorBatch.get_shape()[0], 3, -1])) 
        
        vs_tensorBatch_Corr = tf.abs(tf.subtract(tf.reshape(vs_tensorBatch,[-1, 1]), tf.reshape(vs_SetTensor, [1, -1])))
        vs_tensorBatch_Corr = tf.reshape(vs_tensorBatch_Corr, [-1, vs_SetTensor.get_shape()[0]])
        vs_idx = np.argmin(vs_tensorBatch_Corr, axis = 1)
        vs_tensorBatch_Corr = tf.gather(vs_SetTensor, vs_idx.tolist())
        
        differ_cor1 = tf.subtract(tf.reshape(vs_tensorBatch_Corr, [-1]), tf.reshape(xPrior_tensorBatch[:, 0], [-1]))
        differ_cor2 = tf.subtract(tf.reshape(xPrior_tensorBatch[:, 1], [-1]), tf.reshape(vs_tensorBatch_Corr, [-1]))       
        predTrue_num = 0
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            if differ_cor1[i] >= 0 and differ_cor2[i] >= 0:
                predTrue_num = predTrue_num + 1
        vs_accBatch = predTrue_num/vs_tensorBatch_Corr.get_shape()[0]
        
        loss_regressor = []   
        differ_1 = tf.subtract(tf.reshape(vs_tensorBatch, [-1]), tf.reshape(xPrior_tensorBatch[:, 0], [-1]))
        differ_2 = tf.subtract(tf.reshape(xPrior_tensorBatch[:, 1], [-1]), tf.reshape(vs_tensorBatch, [-1]))       
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            if differ_1[i] < 0:
                loss_regressor.append(differ_1[i] ** 2)
            elif differ_2[i] < 0:
                loss_regressor.append(differ_2[i] ** 2)
            else:
                loss_regressor.append(0)
    
        loss_regressor = tf.convert_to_tensor(loss_regressor)
        loss_regressor_mean = tf.reduce_mean(loss_regressor)          
        
        ##################### vs + x -> y #########################
        vs_RandomSet = []
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            feasible_set = tf.reshape(feasible_set, [-1])
            
            idxs = tf.range(tf.shape(feasible_set)[0])
            ridxs =  tf.random.shuffle(idxs)[:1] # :sample_num
            vs_random = None
            if feasible_set.get_shape()[0] > 0:                      
                idxs = tf.range(tf.shape(feasible_set)[0])
                ridxs =  tf.random.shuffle(idxs)[:1] # :sample_num
                vs_random = tf.gather(feasible_set, ridxs)[0]
            else:
                vs_random = xPrior_tensorBatch[i, 0]
            vs_RandomSet.append(vs_random.numpy())
        
        vs_Random_tensorBatch = tf.convert_to_tensor(vs_RandomSet)
        x_exBatch_Tensor = insert_VS_into_X(vs_Random_tensorBatch, x_tensorBatch)
        
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model_classify(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)   
        
        loss_classify = tf.reduce_mean(loss_B)
        acc_batch = accuracy_compute(np.array(y_tensorBatch), logits.numpy()) 
        
        # summary the results #           
        test_loss.append(loss_classify.numpy())        
        test_acc.append(acc_batch)               
        test_loss_regressor.append(loss_regressor_mean.numpy())
        test_vsAcc.append(vs_accBatch)  
        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['regressor_loss'].append(loss_regressor_mean.numpy())  
            test_epoch_log['vs_acc'].append(vs_accBatch)
            test_epoch_log['acc'].append(acc_batch)
            test_epoch_log['classify_loss'].append(loss_classify.numpy())  
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_testData_batchAcc.csv', index = False)    
    test_loss_mean = np.mean(test_loss)
    test_loss_regressor_mean = np.mean(test_loss_regressor)
    test_acc_mean = np.mean(test_acc)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_mean, test_loss_regressor_mean, test_acc_mean, test_vsAcc_mean

def performance_on_RealDataset_DualLearning_Classify(dataframe_prior, classes, bt_size, model_classify, model_vs, dis_pars, path_res, isTest):
              
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss, test_loss_vs, test_acc, test_vsAcc = [], [], [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [], 'vs_loss': [], 'acc': [], 'vs_acc': [] }
   
    vs_num = dis_pars['vs_num'] 
    Vs = [i for i in range(10, 100, 2)]
    del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
    Vs.append(100)
    vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
    # vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
    
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior = None, None, None
        if classes == 3:
            X, Y, X_prior, _ = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, _ = Convert_to_training_RealData_Classes5(batch_idataFrame)       
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
        
        loss_classify1 = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            yi_probs = tf.gather(Vs_logits, [i]) # (1,13)
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            yiProbs_feaSet = tf.gather(tf.squeeze(yi_probs, axis = 0), more_than_one_vals)             
            vs_iSum =  tf.reduce_sum(yiProbs_feaSet)
            loss_classify1.append(-tf.math.log(vs_iSum))
        
        loss_classify1 = tf.convert_to_tensor(loss_classify1)
        loss_vs_mean = tf.reduce_mean(loss_classify1)              
        
        ##################### vs + x -> y #########################
        vs_RandomSet = []
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                        unique_idx,
                                                        tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            feasible_set = tf.reshape(feasible_set, [-1])
            
            idxs = tf.range(tf.shape(feasible_set)[0])
            ridxs =  tf.random.shuffle(idxs)[:1] # :sample_num
            # vs_random = tf.gather(feasible_set, ridxs)[0]
            vs_random = None
            if feasible_set.get_shape()[0] > 0:                      
                idxs = tf.range(tf.shape(feasible_set)[0])
                ridxs =  tf.random.shuffle(idxs)[:1] # :sample_num
                vs_random = tf.gather(feasible_set, ridxs)[0]
            else:
                vs_random = xPrior_tensorBatch[i, 0]
            vs_RandomSet.append(vs_random.numpy())
        
        vs_Random_tensorBatch = tf.convert_to_tensor(vs_RandomSet)
        x_exBatch_Tensor = insert_VS_into_X(vs_Random_tensorBatch, x_tensorBatch)
        
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model_classify(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)   
        
        loss_classify = tf.reduce_mean(loss_B)
        acc_batch = accuracy_compute(np.array(y_tensorBatch), logits.numpy()) 
        
        # summary the results #           
        test_loss.append(loss_classify.numpy())        
        test_acc.append(acc_batch)               
        test_loss_vs.append(loss_vs_mean.numpy())
        test_vsAcc.append(vs_accBatch)  
        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['vs_loss'].append(loss_vs_mean.numpy())  
            test_epoch_log['vs_acc'].append(vs_accBatch)
            test_epoch_log['acc'].append(acc_batch)
            test_epoch_log['classify_loss'].append(loss_classify.numpy())  
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_testData_batchAcc.csv', index = False)    
    
    test_loss_mean = np.mean(test_loss)
    test_loss_vs_mean = np.mean(test_loss_vs)
    test_acc_mean = np.mean(test_acc)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_mean, test_loss_vs_mean, test_acc_mean, test_vsAcc_mean

def performance_on_RealDataset_DualLearning_Classify_Bayesian(dataframe_prior, classes, bt_size, model_classify, model_vs, dis_pars, path_res, isTest):
              
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss, test_loss_vs, test_acc, test_vsAcc = [], [], [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [], 'vs_loss': [], 'acc': [], 'vs_acc': [] }
   
    vs_num = dis_pars['vs_num'] 
    Vs = [i for i in range(10, 100, 2)]
    del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
    Vs.append(100)
    vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
    # vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
    
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior, X_vsID = None, None, None, None
        if classes == 3:
            X, Y, X_prior, X_vsID = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, X_vsID = Convert_to_training_RealData_Classes5(batch_idataFrame)       
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
        Y_set = obtain_y_set(classes)
        
        x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float) 
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        X_classify = Convert_to_regressor_SynData(X, Y)
        xClassify_tensorBatch = tf.convert_to_tensor(X_classify , dtype = float)

        ##################### y_true + x -> vs #########################
        Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
        # Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
        # vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))
        # dis_type = 'Uniform'
        # syn_tag = False
        # vs_tensorBatch, _, _ = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model_classify)

        ############################# new added #######################
        X_Classify = []
        for j in range(len(Y_set)):
            Yj = [Y_set[j]] * X.shape[0]
            X_Yj_Classify = Convert_to_regressor_SynData(X, Yj)
            X_Classify.append(X_Yj_Classify)
        
        X_Classify = tf.convert_to_tensor(X_Classify)        
        # print(X_Classify)    # (4, 26, 18)   
        X_Classify_reshape1 = tf.keras.layers.concatenate([X_Classify[0,:,:], X_Classify[1,:,:]], axis = -1)
        # X_Classify_reshape2 = tf.keras.layers.concatenate([X_Classify[2,:,:], X_Classify[3,:,:]], axis = -1)
        X_Classify_reshape = tf.keras.layers.concatenate([X_Classify_reshape1, X_Classify[2,:,:]], axis = -1)
        if classes == 5:
              X_Classify_reshape3 = tf.keras.layers.concatenate([X_Classify[3,:,:], X_Classify[4,:,:]], axis = -1)
              X_Classify_reshape = tf.keras.layers.concatenate([X_Classify_reshape, X_Classify_reshape3], axis = -1)  
        X_Classify_reshape = tf.reshape(X_Classify_reshape, [X_Classify_reshape.get_shape()[0], classes, -1])
        # print(X_Classify_reshape.get_shape())  ## (26, 4, 18) 
        Vs_logitSet = []
        for i in range(X_Classify_reshape.get_shape()[0]):
            XiClassify_tensorBatch = X_Classify_reshape[i, :, :]
            # print(XiClassify_tensorBatch.get_shape())#(classes, 18)
            Vs_logits_i = model_vs(tf.reshape(XiClassify_tensorBatch, [XiClassify_tensorBatch.get_shape()[0], 3, -1])) 
            Vs_logitSet.append(Vs_logits_i)
  
        ##################### y_true + x -> vs #########################
        visited_vsID, Vs_inferred_dict = [], {} 
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            vs_ID = x_vsID_Batch[i]
            if vs_ID.numpy() not in visited_vsID:
                visited_vsID.append(vs_ID.numpy())                     
                multi_res = tf.where(tf.equal(vs_ID, x_vsID_Batch))
                VsLogits = None
                if multi_res.get_shape()[0] == 1: # only-one row                              
                    data_single = tf.gather(xClassify_tensorBatch, tf.reshape(multi_res, [-1]), axis = 0)                            
                    Vs_prob = model_vs(tf.reshape(data_single, [data_single.get_shape()[0], 3, -1])) 
                    Vs_prob = tf.squeeze(Vs_prob)  
                    demo_d = Vs_logitSet[tf.reshape(multi_res, [-1])[0].numpy()]
                    demo_d_Sum = tf.math.reduce_sum(demo_d, axis = 0) # the product of each column
                    VsLogits = Vs_prob/demo_d_Sum                                
                elif multi_res.get_shape()[0] > 1: # more than one row exist!
                    data_multi = tf.gather(xClassify_tensorBatch, tf.reshape(multi_res, [-1]), axis = 0)
                    vs_multi_logits = model_vs(tf.reshape(data_multi, [data_multi.get_shape()[0], 3, -1]))
                    # (rows, 13): vs_multi_logits
                    result = []
                    for d in range(data_multi.get_shape()[0]):
                        idx = tf.reshape(multi_res, [-1])[d].numpy()
                        numo_d = vs_multi_logits[d]
                        demo_d = Vs_logitSet[idx]
                        demo_d_Sum = tf.math.reduce_sum(demo_d, axis = 0) # the product of each column
                        res = numo_d/demo_d_Sum
                        result.append(res)  
                    result_tensor = tf.convert_to_tensor(result)
                    VsLogits = tf.math.reduce_prod(result_tensor, axis = 0) # the product of each column             
                    assert(vs_multi_logits.get_shape()[0] == multi_res.get_shape()[0])

                vs_Pred_idx = tf.math.argmax(VsLogits)
                vs_i = tf.gather(vs_SetTensor, tf.squeeze(vs_Pred_idx))               
                for d in range(multi_res.get_shape()[0]):
                    multi_resReshape = tf.reshape(multi_res, [-1])
                    Vs_inferred_dict[str(multi_resReshape[d].numpy())] = vs_i
        
        vs_TensorBatch = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            vs_i = Vs_inferred_dict[str(i)]   
            vs_TensorBatch.append(vs_i.numpy())
        vs_tensorBatch = tf.convert_to_tensor(vs_TensorBatch)
        
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
        
        loss_classify1 = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            yi_probs = tf.gather(Vs_logits, [i]) # (1,13)
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            yiProbs_feaSet = tf.gather(tf.squeeze(yi_probs, axis = 0), more_than_one_vals)             
            vs_iSum =  tf.reduce_sum(yiProbs_feaSet)
            loss_classify1.append(-tf.math.log(vs_iSum))
        
        loss_classify1 = tf.convert_to_tensor(loss_classify1)
        loss_vs_mean = tf.reduce_mean(loss_classify1)              
        
        ##################### vs + x -> y #########################
        vs_RandomSet = []
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                        unique_idx,
                                                        tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            feasible_set = tf.reshape(feasible_set, [-1])
            
            idxs = tf.range(tf.shape(feasible_set)[0])
            ridxs =  tf.random.shuffle(idxs)[:1] # :sample_num
            vs_random = tf.gather(feasible_set, ridxs)[0]
            vs_RandomSet.append(vs_random.numpy())
        
        vs_Random_tensorBatch = tf.convert_to_tensor(vs_RandomSet)
        x_exBatch_Tensor = insert_VS_into_X(vs_Random_tensorBatch, x_tensorBatch)
        
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model_classify(tf.reshape(x_exBatch_Tensor, [x_exBatch_Tensor.get_shape()[0], 3, -1]))           
        loss_B = cce(y_tensorBatch, logits)   
        
        loss_classify = tf.reduce_mean(loss_B)
        acc_batch = accuracy_compute(np.array(y_tensorBatch), logits.numpy()) 
        
        # summary the results #           
        test_loss.append(loss_classify.numpy())        
        test_acc.append(acc_batch)               
        test_loss_vs.append(loss_vs_mean.numpy())
        test_vsAcc.append(vs_accBatch)  
        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['vs_loss'].append(loss_vs_mean.numpy())  
            test_epoch_log['vs_acc'].append(vs_accBatch)
            test_epoch_log['acc'].append(acc_batch)
            test_epoch_log['classify_loss'].append(loss_classify.numpy())  
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_testData_batchAcc.csv', index = False)    
    
    test_loss_mean = np.mean(test_loss)
    test_loss_vs_mean = np.mean(test_loss_vs)
    test_acc_mean = np.mean(test_acc)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_mean, test_loss_vs_mean, test_acc_mean, test_vsAcc_mean


def performance_on_RealDataset_SingleLearning(dataframe_prior, classes, bt_size, model_regressor, dis_pars, path_res, isTest):
                    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss_regressor,  test_vsAcc = [], []
    test_epoch_log = {'batch': [] , 'regressor_loss': [],   'vs_acc': [] }
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior = None, None, None
        if classes == 3:
            X, Y, X_prior, _ = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, _= Convert_to_training_RealData_Classes5(batch_idataFrame)
      
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)
    
        X_regressor = Convert_to_regressor_SynData(X, Y)
        
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)
        xRegressor_tensorBatch = tf.convert_to_tensor(X_regressor, dtype = float)
        
        vs_num = dis_pars['vs_num'] 
        Vs = [i for i in range(10, 100, 2)]
        del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
        Vs.append(100)
        vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
        # vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
        # VS LOSS #
        vs_tensorBatch = model_regressor(tf.reshape(xRegressor_tensorBatch, [xRegressor_tensorBatch.get_shape()[0], 3, -1])) 
            
        vs_tensorBatch_Corr = tf.abs(tf.subtract(tf.reshape(vs_tensorBatch,[-1, 1]), tf.reshape(vs_SetTensor, [1, -1])))
        vs_tensorBatch_Corr = tf.reshape(vs_tensorBatch_Corr, [-1, vs_SetTensor.get_shape()[0]])
        vs_idx = np.argmin(vs_tensorBatch_Corr, axis = 1)
        vs_tensorBatch_Corr = tf.gather(vs_SetTensor, vs_idx.tolist())
        
        differ_cor1 = tf.subtract(tf.reshape(vs_tensorBatch_Corr, [-1]), tf.reshape(xPrior_tensorBatch[:, 0], [-1]))
        differ_cor2 = tf.subtract(tf.reshape(xPrior_tensorBatch[:, 1], [-1]), tf.reshape(vs_tensorBatch_Corr, [-1]))       
        predTrue_num = 0
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            if differ_cor1[i] >= 0 and differ_cor2[i] >= 0:
                predTrue_num = predTrue_num + 1
        vs_accBatch = predTrue_num/vs_tensorBatch_Corr.get_shape()[0]
        
        ## update its gradients ##           
        loss_regressor = []   
        differ_1 = tf.subtract(tf.reshape(vs_tensorBatch, [-1]), tf.reshape(xPrior_tensorBatch[:, 0], [-1]))
        differ_2 = tf.subtract(tf.reshape(xPrior_tensorBatch[:, 1], [-1]), tf.reshape(vs_tensorBatch, [-1]))       
        for i in range(xPrior_tensorBatch.get_shape()[0]):
            if differ_1[i] < 0:
                loss_regressor.append(differ_1[i] ** 2)
            elif differ_2[i] < 0:
                loss_regressor.append(differ_2[i] ** 2)
            else:
                loss_regressor.append(0)

        loss_regressor = tf.convert_to_tensor(loss_regressor)
        loss_regressor_mean = tf.reduce_mean(loss_regressor)
                  
        # summary the results #                        
        test_loss_regressor.append(loss_regressor_mean.numpy())
        test_vsAcc.append(vs_accBatch)  
        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['regressor_loss'].append(loss_regressor_mean.numpy())  
            test_epoch_log['vs_acc'].append(vs_accBatch)
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_testData_batchAcc.csv', index = False)    

    test_loss_regressor_mean = np.mean(test_loss_regressor)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_regressor_mean, test_vsAcc_mean

def performance_on_RealDataset_SingleLearning_Classify(dataframe_prior, classes, bt_size, model_C, dis_pars, path_res, isTest):
                    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss_c,  test_vsAcc = [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [],   'vs_acc': [] }
    
    vs_num = dis_pars['vs_num'] 
    # vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
    Vs = [i for i in range(10, 100, 2)]
    del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
    Vs.append(100)
    vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior = None, None, None
        if classes == 3:
            X, Y, X_prior, _ = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, _= Convert_to_training_RealData_Classes5(batch_idataFrame)
      
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)   
        
        X_Classify = Convert_to_regressor_SynData(X, Y)
                
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)  
        xClassify_tensorBatch = tf.convert_to_tensor(X_Classify, dtype = float)
        
        Vs_logits = model_C(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
        Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
        vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))

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
        
        loss_classify = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            yi_probs = tf.gather(Vs_logits, [i]) # (1,13)
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            yiProbs_feaSet = tf.gather(tf.squeeze(yi_probs, axis = 0), more_than_one_vals)             
            vs_iSum =  tf.reduce_sum(yiProbs_feaSet)
            loss_classify.append(-tf.math.log(vs_iSum))
        
        loss_classify = tf.convert_to_tensor(loss_classify)
        loss_classify_mean = tf.reduce_mean(loss_classify)
        
        test_loss_c.append(loss_classify_mean.numpy())
        test_vsAcc.append(vs_accBatch) 

        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['classify_loss'].append(loss_classify_mean.numpy())  
            test_epoch_log['vs_acc'].append(vs_accBatch)
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_testData_batchAcc.csv', index = False)    

    test_loss_c_mean = np.mean(test_loss_c)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_c_mean, test_vsAcc_mean

def performance_on_RealDataset_SingleLearning_Classify_Bayesian(dataframe_prior, classes, bt_size, model_Classify, model_vs, dis_pars, path_res, isTest):
                    
    batch_dataFrame, batch_pairIdx = extract_batchData_sameSellerItem(dataframe_prior, bt_size)            
    
    test_loss_c,  test_vsAcc = [], []
    test_epoch_log = {'batch': [] , 'classify_loss': [],   'vs_acc': [] }
    
    vs_num = dis_pars['vs_num'] 
    # vs_SetTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
    Vs = [i for i in range(10, 100, 2)]
    del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
    Vs.append(100)
    vs_SetTensor = tf.convert_to_tensor(Vs, dtype = float)
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]

        X, Y, X_prior, X_vsID = None, None, None, None
        if classes == 3:
            X, Y, X_prior, X_vsID = Convert_to_training_RealData(batch_idataFrame)  
        elif classes == 5:
            X, Y, X_prior, X_vsID= Convert_to_training_RealData_Classes5(batch_idataFrame)
      
        Y = to_categorical(Y, classes)   
        X = X.astype(np.float64)   
        Y_set = obtain_y_set(classes)
        
        X_classify = Convert_to_regressor_SynData(X, Y)                
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)  
        xClassify_tensorBatch = tf.convert_to_tensor(X_classify, dtype = float)
        
        Vs_logits = model_vs(tf.reshape(xClassify_tensorBatch, [xClassify_tensorBatch.get_shape()[0], 3, -1])) 
        # Y_Pred_trueIdx_tensorBatch = tf.math.argmax(Vs_logits, axis = 1)
        # vs_tensorBatch = tf.gather(vs_SetTensor, tf.squeeze(Y_Pred_trueIdx_tensorBatch))
       
        # x_tensorBatch = tf.convert_to_tensor(X, dtype= float)
        # y_tensorBatch = tf.convert_to_tensor(Y)
        xPrior_tensorBatch = tf.convert_to_tensor(X_prior, dtype= float)  
        x_vsID_Batch = tf.convert_to_tensor(X_vsID, dtype = float)
        
        ##################### y_true + x -> vs #########################
        #dis_type = 'Uniform'
        #syn_tag = False
        #  vs_tensorBatch, _, _ = Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model_Classify)
        # vs_tensorBatch, _, _ = Vs_inference_with_MultiRows_Tensor_InferTwice(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model_vs)
        ############################# new added #######################
        X_Classify = []
        for j in range(len(Y_set)):
            Yj = [Y_set[j]] * X.shape[0]
            X_Yj_Classify = Convert_to_regressor_SynData(X, Yj)
            X_Classify.append(X_Yj_Classify)
        
        X_Classify = tf.convert_to_tensor(X_Classify)        
        # print(X_Classify)    # (4, 26, 18)   
        X_Classify_reshape1 = tf.keras.layers.concatenate([X_Classify[0,:,:], X_Classify[1,:,:]], axis = -1)
        # X_Classify_reshape2 = tf.keras.layers.concatenate([X_Classify[2,:,:], X_Classify[3,:,:]], axis = -1)
        X_Classify_reshape = tf.keras.layers.concatenate([X_Classify_reshape1, X_Classify[2,:,:]], axis = -1)
        if classes == 5:
              X_Classify_reshape3 = tf.keras.layers.concatenate([X_Classify[3,:,:], X_Classify[4,:,:]], axis = -1)
              X_Classify_reshape = tf.keras.layers.concatenate([X_Classify_reshape, X_Classify_reshape3], axis = -1)  
        X_Classify_reshape = tf.reshape(X_Classify_reshape, [X_Classify_reshape.get_shape()[0], classes, -1])
        # print(X_Classify_reshape.get_shape())  ## (26, 4, 18) 
        Vs_logitSet = []
        for i in range(X_Classify_reshape.get_shape()[0]):
            XiClassify_tensorBatch = X_Classify_reshape[i, :, :]
            # print(XiClassify_tensorBatch.get_shape())#(classes, 18)
            Vs_logits_i = model_vs(tf.reshape(XiClassify_tensorBatch, [XiClassify_tensorBatch.get_shape()[0], 3, -1])) 
            Vs_logitSet.append(Vs_logits_i)
 
        ##################### y_true + x -> vs #########################
        visited_vsID, Vs_inferred_dict = [], {} 
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            vs_ID = x_vsID_Batch[i]
            if vs_ID.numpy() not in visited_vsID:
                visited_vsID.append(vs_ID.numpy())                     
                multi_res = tf.where(tf.equal(vs_ID, x_vsID_Batch))
                VsLogits = None
                if multi_res.get_shape()[0] == 1: # only-one row                              
                    data_single = tf.gather(xClassify_tensorBatch, tf.reshape(multi_res, [-1]), axis = 0)                            
                    Vs_prob = model_vs(tf.reshape(data_single, [data_single.get_shape()[0], 3, -1])) 
                    Vs_prob = tf.squeeze(Vs_prob)  
                    demo_d = Vs_logitSet[multi_res[0].numpy()]
                    demo_d_Sum = tf.math.reduce_sum(demo_d, axis = 0) # the product of each column
                    VsLogits = Vs_prob/demo_d_Sum                                
                elif multi_res.get_shape()[0] > 1: # more than one row exist!
                    data_multi = tf.gather(xClassify_tensorBatch, tf.reshape(multi_res, [-1]), axis = 0)
                    vs_multi_logits = model_vs(tf.reshape(data_multi, [data_multi.get_shape()[0], 3, -1]))
                    # (rows, 13): vs_multi_logits
                    result = []
                    for d in range(data_multi.get_shape()[0]):
                        idx = tf.reshape(multi_res, [-1])[d].numpy()
                        numo_d = vs_multi_logits[d]
                        demo_d = Vs_logitSet[idx]
                        demo_d_Sum = tf.math.reduce_sum(demo_d, axis = 0) # the product of each column
                        res = numo_d/demo_d_Sum
                        result.append(res)  
                    result_tensor = tf.convert_to_tensor(result)
                    VsLogits = tf.math.reduce_prod(result_tensor, axis = 0) # the product of each column             
                    assert(vs_multi_logits.get_shape()[0] == multi_res.get_shape()[0])

                vs_Pred_idx = tf.math.argmax(VsLogits)
                vs_i = tf.gather(vs_SetTensor, tf.squeeze(vs_Pred_idx))               
                for d in range(multi_res.get_shape()[0]):
                    multi_resReshape = tf.reshape(multi_res, [-1])
                    Vs_inferred_dict[str(multi_resReshape[d].numpy())] = vs_i     
        
        vs_TensorBatch = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            vs_i = Vs_inferred_dict[str(i)]   
            vs_TensorBatch.append(vs_i.numpy())
        vs_tensorBatch = tf.convert_to_tensor(vs_TensorBatch)        
        
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
        
        loss_classify = []
        for i in range(xClassify_tensorBatch.get_shape()[0]):
            yi_probs = tf.gather(Vs_logits, [i]) # (1,13)
            res_1 = tf.squeeze(tf.where(vs_SetTensor <= xPrior_tensorBatch[i, 1]), axis = 1)
            res_2 = tf.squeeze(tf.where(xPrior_tensorBatch[i, 0] <= vs_SetTensor), axis = 1)
            res = tf.concat((res_1, res_2), axis = 0)
            unique_res_vals, unique_idx = tf.unique(res)
            count_res_unique = tf.math.unsorted_segment_sum(tf.ones_like(res),
                                                       unique_idx,
                                                       tf.shape(res)[0])
            more_than_one = tf.greater(count_res_unique, 1)
            more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            more_than_one_vals = tf.squeeze(tf.gather(unique_res_vals, more_than_one_idx))
            feasible_set = tf.gather(vs_SetTensor, more_than_one_vals)
            yiProbs_feaSet = tf.gather(tf.squeeze(yi_probs, axis = 0), more_than_one_vals)             
            vs_iSum =  tf.reduce_sum(yiProbs_feaSet)
            loss_classify.append(-tf.math.log(vs_iSum))
        
        loss_classify = tf.convert_to_tensor(loss_classify)
        loss_classify_mean = tf.reduce_mean(loss_classify)
        
        test_loss_c.append(loss_classify_mean.numpy())
        test_vsAcc.append(vs_accBatch) 

        if isTest:
            test_epoch_log['batch'].append(batch + 1)
            test_epoch_log['classify_loss'].append(loss_classify_mean.numpy())  
            test_epoch_log['vs_acc'].append(vs_accBatch)
            pd.DataFrame(test_epoch_log).to_csv(path_res +'data_testData_batchAcc.csv', index = False)    

    test_loss_c_mean = np.mean(test_loss_c)
    test_vsAcc_mean = np.mean(test_vsAcc)

    return test_loss_c_mean, test_vsAcc_mean


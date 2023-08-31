#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Lvye

"""

import numpy as np
import tensorflow as tf
import keras.backend as K
import copy

def read_v_distribution(file_name):

    with open(file_name, 'r') as f:
        lines = [eval(line.strip()) for line in f]
    
    return lines[0]

def Vs_prior_distr_compute(prior_i, dis_type, syn_tag, dis_pars):
    # generate the prior distribution of vs based on its distribution type and distribution pars
    # dis_pars = {'extend_rangeScalar':2, 'vs_num': 30} # for real data
    # dis_pars = {'price_min':10, 'price_max':60, 'gap': 5} # for this synthesized data
    vs_priorset = []
    if dis_type == 'Uniform':
        vs_set = []
        if not syn_tag: # real dataset 
           extend_rangeScalar, vs_num = dis_pars['extend_rangeScalar'], dis_pars['vs_num'] #extend_rangeScalar, vs_num = 2, 30 # extend_k used for extending the range of prior range
           dis_min, dis_max = prior_i[0] * 1/extend_rangeScalar, prior_i[1] * extend_rangeScalar
           if dis_min == 0:
               dis_min = 1
           vs_set = np.linspace(dis_min,dis_max, vs_num).astype(float)
        else: # synthesized dataset
           dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  #10, 60, 5
           # vs_set = np.linspace(dis_min,dis_max, vs_num).astype(float)
           vs_set = [i for i in range(dis_min, dis_max, gap)] # should be the same as the way in synthesized data     
        for vs in vs_set:
            vs_priorset.append((vs,1/len(vs_set)))           
    elif dis_type == 'Categorical':
        vs_set = []
        if not syn_tag: # real dataset 
           extend_rangeScalar, vs_num = dis_pars['extend_rangeScalar'], dis_pars['vs_num'] #extend_rangeScalar, vs_num = 2, 30 # extend_k used for extending the range of prior range
           dis_min, dis_max = prior_i[0] * 1/extend_rangeScalar, prior_i[1] * extend_rangeScalar
           if dis_min == 0:
               dis_min = 1
           vs_set = np.linspace(dis_min,dis_max, vs_num).astype(float)
        else: # synthesized dataset
           dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
           vs_set = [i for i in range(dis_min, dis_max, gap)] # should be the same as the way in synthesized data     
           
           prob_list = read_v_distribution(dis_pars['vs_disPath'] + '/vs_categorical_distribution.txt')
        
        for i, vs in enumerate(vs_set):
            vs_priorset.append((vs, prob_list[i]))          

    return vs_priorset

def Vs_inference_with_one_row(xi, yi, prior_i, syn_tag, dis_type, dis_pars, model):
    # should return the possible of vs and its possibility
    vs_priorSet = Vs_prior_distr_compute(prior_i, dis_type, syn_tag, dis_pars)

    denom_sum, pv_poster = 0, []
    for vs, p_vs in vs_priorSet:
        y_true_idx = np.where(np.array(yi) == 1)[0][0]
        xi_extend = np.insert(np.array(xi), 0, vs)
        xi_extend = xi_extend.reshape(-1, 1, len(xi_extend))
        # logit = model(xi_extend).numpy().tolist()[0]
        logit = model(xi_extend)
        py_on_vs = logit.numpy().tolist()[0][y_true_idx]       

        denom_sum = denom_sum + p_vs * py_on_vs
        pv_poster.append((vs, p_vs * py_on_vs))
    
    vs_posterSet, vs_i, vs_i_p = [], 0, 0
    for vs, p in pv_poster:
        vs_posterSet.append((vs, p / denom_sum))
        if (p / denom_sum) > vs_i_p:
            vs_i_p = p / denom_sum
            vs_i = vs
    
    return vs_priorSet, vs_posterSet, vs_i, vs_i_p

def  Vs_inference_for_batch(x, y, x_prior, syn_tag, vs_priorDistr_type, dis_pars, model):
     vs_batch, priorBatch, postBatch = [], [], []
     for i in range(x.shape[0]):
         x_i, y_i, prior_i = np.array(x)[i], np.array(y)[i], np.array(x_prior)[i]
         vs_i_priorSet, vs_i_postSet, vs_iMax, _ = Vs_inference_with_one_row(x_i, y_i, prior_i, syn_tag, vs_priorDistr_type, dis_pars, model)
         vs_batch.append(vs_iMax)
         priorBatch.append(vs_i_priorSet)
         postBatch.append(vs_i_postSet)
     return vs_batch, priorBatch, postBatch
 
    
def Vs_prior_distr_computeArray(x_prior, dis_type, syn_tag, dis_pars):
    # generate the prior distribution of vs based on its distribution type and distribution pars
    # dis_pars = {'extend_rangeScalar':2, 'vs_num': 30} # for real data
    # dis_pars = {'price_min':10, 'price_max':60, 'gap': 5} # for this synthesized data
    vs_priorSet_Array = []
    if dis_type == 'Uniform' and syn_tag == True:
        dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  #10, 60, 5
        vs_set = [i for i in range(dis_min, dis_max, gap)] # should be the same as the way in synthesized data     
        
        vs_priorset = list(map(lambda x: (x, 1/len(vs_set)), vs_set))       
        vs_priorSet_Array = [vs_priorset] * len(x_prior)
    
    if dis_type == 'Categorical' and syn_tag == True:
        dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  #10, 60, 5
        vs_set = [i for i in range(dis_min, dis_max, gap)] # should be the same as the way in synthesized data            
        
        prob_list = read_v_distribution(dis_pars['vs_disPath'] + '/vs_categorical_distribution.txt')
        
        vs_priorset = list(map(lambda x, y: (x, y), vs_set, prob_list)) 
        vs_priorSet_Array = [vs_priorset] * len(x_prior)
    
    if dis_type == 'Uniform' and syn_tag == False:    
        extend_rangeScalar, vs_num = dis_pars['extend_rangeScalar'], dis_pars['vs_num'] #extend_rangeScalar, vs_num = 2, 30 # extend_k used for extending the range of prior range
        
        dis_minArray, dis_maxArray = np.array(x_prior)[:, 0] * 1./extend_rangeScalar, np.array(x_prior)[:, 1] * extend_rangeScalar
        dis_minArray[np.where(dis_minArray == 0)] = 1

        vs_priorsetArray = list(map(lambda x, y: list(np.linspace(x, y, vs_num)), list(dis_minArray), list(dis_maxArray)))        
        vs_priorSet_Array = [list(map(lambda x: (x, 1/len(vs_set)), vs_set)) for vs_set in vs_priorsetArray]
    
    
    return vs_priorSet_Array


def Vs_inference_with_rowsNumpy(x, y, x_prior, syn_tag, dis_type, dis_pars, model):
    # should return the possible of vs and its possibility
    # x datframe; y: dataframe
    vs_priorSet_Array = Vs_prior_distr_computeArray(x_prior, dis_type, syn_tag, dis_pars)
    y_trueIdx_batch = list(np.argmax(np.array(y), axis = 1))  

    ### how to accelerate it
    def p_vsPoster_compute(xi, yid, zSet):
        # print(zSet)
        vs_Set = [ zSet[i][0] for i in range(len(zSet))]
        p_vs_Set = [ zSet[i][1] for i in range(len(zSet))]

        xi_extendSet = [ np.insert(xi, 0, vs) for vs in vs_Set]
        xi_extendSet = [xi_extend.reshape(-1, 1, len(xi_extend)) for xi_extend in xi_extendSet]

        logitSet = [ model(xi_extend) for xi_extend in xi_extendSet]
        py_onVs_set = list(map(lambda x, y: x.numpy().tolist()[0][y], logitSet, [yid]*len(logitSet)))

        pv_posterSet = list(map(lambda x, y: x*y, p_vs_Set, py_onVs_set))        
        pv_posterArray = list(map(lambda x, y: (x, y/np.sum(pv_posterSet)), vs_Set, pv_posterSet))

        return pv_posterArray

    vs_posterSet_Array = list(map(lambda x, y, zSet: p_vsPoster_compute(x, y, zSet), list(np.array(x)), y_trueIdx_batch, vs_priorSet_Array))
     
    vs_batch_Array = list(map(lambda x: np.array(x)[np.argmax(np.array(x)[:,1]), 0], vs_posterSet_Array))
    vsP_batch_Array = list(map(lambda x: np.array(x)[np.argmax(np.array(x)[:,1]), 1], vs_posterSet_Array))    
    
    return vs_batch_Array, vs_priorSet_Array, vs_posterSet_Array

def rewardBatch_A_computeTensor(r_cons, vs_batch, x_prior):    
    
    rBatch_A = tf.zeros([x_prior.get_shape()[0],], dtype = float)
    rBatch_A = tf.add(rBatch_A, r_cons).numpy()
    
    differ_1 = tf.subtract(vs_batch, x_prior[:, 0])
    differ_2 = tf.subtract(x_prior[:, 1], vs_batch)

    for i in range(x_prior.get_shape()[0]):
        if differ_1[i] < 0:
            rBatch_A[i] = differ_1[i]
        elif differ_2[i] < 0:
            rBatch_A[i] = differ_2[i]
    scalar = 5
    rBatch_A = tf.convert_to_tensor(rBatch_A / scalar)
   
    return rBatch_A

def Vs_prior_distr_computeTensor(x_prior, dis_type, syn_tag, dis_pars):
    # generate the prior distribution of vs based on its distribution type and distribution pars
    # dis_pars = {'extend_rangeScalar':2, 'vs_num': 30} # for real data
    # dis_pars = {'price_min':10, 'price_max':60, 'gap': 5} # for this synthesized data
    vs_priorSet_TensorBatch, vs_priorset_Tensor = None, None
    if dis_type == 'Uniform' and syn_tag == True:
        dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  
        vs_setTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)    
        prob_listTensor = tf.map_fn(lambda t: 1/vs_setTensor.get_shape()[0], vs_setTensor)
        if 'vs_disPath' in dis_pars:
            prob_listTensor = tf.convert_to_tensor(read_v_distribution(dis_pars['vs_disPath'] + '/vs_uniform_distribution.txt'))
            
        vs_priorset_Tensor = tf.stack([vs_setTensor, prob_listTensor], axis = 1) #      
        vs_priorSet_TensorBatch = tf.tile(tf.expand_dims(vs_priorset_Tensor, 0), [x_prior.get_shape()[0], 1, 1])
    
    if dis_type == 'Categorical' and syn_tag == True:
        dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']  #10, 60, 5
        vs_setTensor = tf.convert_to_tensor([i for i in range(dis_min, dis_max, gap)], dtype = float)         
        prob_listTensor = tf.convert_to_tensor(read_v_distribution(dis_pars['vs_disPath'] + '/vs_categorical_distribution.txt'))
        
        vs_priorset_Tensor = tf.stack([vs_setTensor, prob_listTensor], axis = 1) #       
        vs_priorSet_TensorBatch = tf.tile(tf.expand_dims(vs_priorset_Tensor, 0), [x_prior.get_shape()[0], 1, 1])
        # print(vs_priorSet_TensorBatch)
        # print(tf.shape(vs_priorSet_TensorBatch)) # batch_size * vs_num * 2(10*13*2)
    
    if dis_type == 'Uniform' and syn_tag == False:    
        vs_num =  dis_pars['vs_num'] 
        # vs_setTensor = tf.convert_to_tensor(np.linspace(10, 100, vs_num), dtype = float)
        Vs = [i for i in range(10, 100, 2)]
        del Vs[np.where(np.array(Vs) == 10)[0].tolist()[0]]
        Vs.append(100)
        vs_setTensor = tf.convert_to_tensor(Vs, dtype = float)
        
        prob_listTensor = tf.map_fn(lambda t: 1 / vs_setTensor.get_shape()[0], vs_setTensor)
        vs_priorset_Tensor = tf.stack([vs_setTensor, prob_listTensor], axis = 1) #      
        vs_priorSet_TensorBatch = tf.tile(tf.expand_dims(vs_priorset_Tensor, 0), [x_prior.get_shape()[0], 1, 1])
    
    return vs_priorSet_TensorBatch, vs_priorset_Tensor   
 
def Vs_inference_with_rowsTensor(x, y, x_prior, syn_tag, dis_type, dis_pars, model):
    # x tensor; y: tensor
    vs_priorSet_TensorBatch, vs_priorset_Tensor = Vs_prior_distr_computeTensor(x_prior, dis_type, syn_tag, dis_pars)
    y_trueIdx_batchTensor = tf.math.argmax(y, axis = 1)
    x_extends = tf.tile(x, tf.constant([1, vs_priorset_Tensor.get_shape()[0]]))
    # print(x_extends)
    x_extends = tf.reshape(x_extends, [x.get_shape()[0], vs_priorset_Tensor.get_shape()[0] , -1])

    # x_extends_Tensor = tf.keras.layers.concatenate([vs_priorSet_TensorBatch[:,:,0:1], x_extends], axis = -1)    
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
    # print(x_extends_Tensor)
    # print(x_extends_Tensor.get_shape()) # 10*13*7
    vs_probs = tf.gather(vs_priorset_Tensor, [1], axis =1)
    vs_setTensor = tf.gather(vs_priorset_Tensor, [0], axis =1)
    
    vs_TensorBatch, vs_postSet_TensorBatch = None, None
    for i in range(x_extends_Tensor.get_shape()[0]):
        xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], 3, -1]))
        yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)

        yi_probs_onVs = tf.divide(tf.math.multiply(yi_probs, vs_probs), tf.reduce_sum(tf.math.multiply(yi_probs, vs_probs)))
        
        vs_postSet_Tensor = tf.stack([vs_setTensor, yi_probs_onVs], axis = 1)
        vs_postSet_Tensor = tf.reshape(vs_postSet_Tensor, [vs_priorset_Tensor.get_shape()[0], -1])
        
        vs_i = vs_postSet_Tensor[K.eval(tf.math.argmax(yi_probs_onVs, axis = 0))[0], 0]
        
        # print(vs_postSet_Tensor) 
        # print(vs_i)
        vs_i_temp = tf.expand_dims(vs_i, 0)
        vs_postSet_TensorTemp = tf.expand_dims(vs_postSet_Tensor, 0) # (1, 13,2)
        if i == 0:
            vs_postSet_TensorBatch = vs_postSet_TensorTemp
            vs_TensorBatch = vs_i_temp
        else:
            vs_postSet_TensorBatch = tf.concat([vs_postSet_TensorBatch, vs_postSet_TensorTemp], axis = 0)
            vs_TensorBatch = tf.concat([vs_TensorBatch, vs_i_temp], axis = 0)
    
    return vs_TensorBatch, vs_priorSet_TensorBatch, vs_postSet_TensorBatch
        
def Vs_inference_with_MultiRows_Tensor(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model):
    
    vs_priorSet_TensorBatch, vs_priorset_Tensor = Vs_prior_distr_computeTensor(xPrior_tensorBatch, dis_type, syn_tag, dis_pars)

    y_trueIdx_batchTensor = tf.math.argmax(y_tensorBatch, axis = 1)

    x_extends = tf.tile(x_tensorBatch, tf.constant([1, vs_priorset_Tensor.get_shape()[0]]))
    # print(x_extends)
    x_extends = tf.reshape(x_extends, [x_tensorBatch.get_shape()[0], vs_priorset_Tensor.get_shape()[0] , -1])
    # print(x_extends.get_shape()) # 10*13*6

    # x_extends_Tensor = tf.keras.layers.concatenate([vs_priorSet_TensorBatch[:,:,0:1], x_extends], axis = -1)    
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
    
    # print(x_extends_Tensor.get_shape()) # 10*13*7
    vs_probs = tf.gather(vs_priorset_Tensor, [1], axis =1)
    vs_setTensor = tf.gather(vs_priorset_Tensor, [0], axis =1)
    
    visited_vsID = []   
    resultsPost_dict, Vs_inferred_dict = {}, {}
    for i in range(x_extends_Tensor.get_shape()[0]):
        vs_ID = x_vsID_Batch[i]
        if vs_ID.numpy() not in visited_vsID:
            visited_vsID.append(vs_ID.numpy())                     
            multi_res = tf.where(tf.equal(vs_ID, x_vsID_Batch))
            yi_probs = None
            if multi_res.get_shape()[0] == 1: # only-one row           
                xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], 3, -1]))
                yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)
                
            elif multi_res.get_shape()[0] > 1: # more than one row exist!
                data_multi = tf.gather(x_extends_Tensor, tf.reshape(multi_res, [-1]), axis = 0)
                y_trueIdx_new = tf.gather(y_trueIdx_batchTensor, tf.reshape(multi_res, [-1]), axis = 0)
                yi_probs_temp = None
                for d in range(data_multi.get_shape()[0]):
                    xi_d_logits = model(tf.reshape(data_multi[d,:,:], [data_multi.get_shape()[1], 3, -1])) #(13,4)
                    yi_d_true = tf.gather(y_trueIdx_new, d, axis = 0) 
                    yi_d_probs = tf.gather(xi_d_logits, yi_d_true, axis = 1) #(13,)   
                    yi_d_probs = tf.reshape(yi_d_probs, [-1, 1])
                    if d == 0:
                        yi_probs_temp = yi_d_probs
                    else:
                        yi_probs_temp = tf.concat([yi_probs_temp, yi_d_probs], axis = 1) # (13,9)
                # print(yi_probs_temp.get_shape())  #(13,3)                
                yi_probs = tf.math.reduce_prod(yi_probs_temp, axis = 1) # the product of each row   
                yi_probs = tf.reshape(yi_probs, [-1, 1])            
                assert(yi_probs_temp.get_shape()[1] == multi_res.get_shape()[0])

            yi_probs_onVs = tf.divide(tf.math.multiply(yi_probs, vs_probs), tf.reduce_sum(tf.math.multiply(yi_probs, vs_probs)))                   
            vs_postSet_Tensor = tf.stack([vs_setTensor, yi_probs_onVs], axis = 1)
            vs_postSet_Tensor = tf.reshape(vs_postSet_Tensor, [vs_priorset_Tensor.get_shape()[0], -1])            
            vs_i = vs_postSet_Tensor[K.eval(tf.math.argmax(yi_probs_onVs, axis = 0))[0], 0]
           
            for d in range(multi_res.get_shape()[0]):
                multi_resReshape = tf.reshape(multi_res, [-1])
                resultsPost_dict[str(multi_resReshape[d].numpy())] = vs_postSet_Tensor
                Vs_inferred_dict[str(multi_resReshape[d].numpy())] = vs_i

    assert(len(resultsPost_dict) == x_extends_Tensor.get_shape()[0])
    # concatenate it to the result and avoid to compute repeatedly!
    vs_TensorBatch, vs_postSet_TensorBatch = None, None
    for i in range(x_extends_Tensor.get_shape()[0]):
        vs_i = Vs_inferred_dict[str(i)]         
        vs_postSet_Tensor = resultsPost_dict[str(i)]
        
        vs_i = tf.expand_dims(vs_i, 0)
        vs_postSet_Tensor = tf.expand_dims(vs_postSet_Tensor, 0) # (1, 13,2)
        if i == 0:
            vs_postSet_TensorBatch = vs_postSet_Tensor
            vs_TensorBatch = vs_i
        else:
            vs_postSet_TensorBatch = tf.concat([vs_postSet_TensorBatch, vs_postSet_Tensor], axis = 0)
            vs_TensorBatch = tf.concat([vs_TensorBatch, vs_i], axis = 0)
    
    assert(vs_TensorBatch.get_shape()[0] == x_extends_Tensor.get_shape()[0])
    
    return vs_TensorBatch, vs_priorSet_TensorBatch, vs_postSet_TensorBatch

def insert_VS_into_X(vs_tensorBatch, x_tensorBatch):
    
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
   
    return x_exBatch_Tensor

def Vs_inference_with_MultiRows_Tensor_transformer(x_tensorBatch, y_tensorBatch, xPrior_tensorBatch, x_vsID_Batch, dis_type, syn_tag, dis_pars, model):
    
    vs_priorSet_TensorBatch, vs_priorset_Tensor = Vs_prior_distr_computeTensor(xPrior_tensorBatch, dis_type, syn_tag, dis_pars)

    y_trueIdx_batchTensor = tf.math.argmax(y_tensorBatch, axis = 1)

    x_extends = tf.tile(x_tensorBatch, tf.constant([1, vs_priorset_Tensor.get_shape()[0]]))
    # print(x_extends)
    x_extends = tf.reshape(x_extends, [x_tensorBatch.get_shape()[0], vs_priorset_Tensor.get_shape()[0] , -1])
    # print(x_extends.get_shape()) # 10*13*6

    # x_extends_Tensor = tf.keras.layers.concatenate([vs_priorSet_TensorBatch[:,:,0:1], x_extends], axis = -1)    
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
    
    # print(x_extends_Tensor.get_shape()) # 10*13*7
    vs_probs = tf.gather(vs_priorset_Tensor, [1], axis =1)
    vs_setTensor = tf.gather(vs_priorset_Tensor, [0], axis =1)
    
    visited_vsID = []   
    resultsPost_dict, Vs_inferred_dict = {}, {}
    for i in range(x_extends_Tensor.get_shape()[0]):
        vs_ID = x_vsID_Batch[i]
        if vs_ID.numpy() not in visited_vsID:
            visited_vsID.append(vs_ID.numpy())                     
            multi_res = tf.where(tf.equal(vs_ID, x_vsID_Batch))
            yi_probs = None
            if multi_res.get_shape()[0] == 1: # only-one row           
                xi_logits = model(tf.reshape(x_extends_Tensor[i,:,:], [x_extends_Tensor.get_shape()[1], -1, 1]))
                yi_probs = tf.gather(xi_logits, [y_trueIdx_batchTensor[i]], axis = 1)
                
            elif multi_res.get_shape()[0] > 1: # more than one row exist!
                data_multi = tf.gather(x_extends_Tensor, tf.reshape(multi_res, [-1]), axis = 0)
                y_trueIdx_new = tf.gather(y_trueIdx_batchTensor, tf.reshape(multi_res, [-1]), axis = 0)
                yi_probs_temp = None
                for d in range(data_multi.get_shape()[0]):
                    xi_d_logits = model(tf.reshape(data_multi[d,:,:], [data_multi.get_shape()[1], -1, 1])) #(13,4)
                    yi_d_true = tf.gather(y_trueIdx_new, d, axis = 0) 
                    yi_d_probs = tf.gather(xi_d_logits, yi_d_true, axis = 1) #(13,)   
                    yi_d_probs = tf.reshape(yi_d_probs, [-1, 1])
                    if d == 0:
                        yi_probs_temp = yi_d_probs
                    else:
                        yi_probs_temp = tf.concat([yi_probs_temp, yi_d_probs], axis = 1) # (13,9)
                # print(yi_probs_temp.get_shape())  #(13,3)                
                yi_probs = tf.math.reduce_prod(yi_probs_temp, axis = 1) # the product of each row   
                yi_probs = tf.reshape(yi_probs, [-1, 1])            
                assert(yi_probs_temp.get_shape()[1] == multi_res.get_shape()[0])

            yi_probs_onVs = tf.divide(tf.math.multiply(yi_probs, vs_probs), tf.reduce_sum(tf.math.multiply(yi_probs, vs_probs)))                   
            vs_postSet_Tensor = tf.stack([vs_setTensor, yi_probs_onVs], axis = 1)
            vs_postSet_Tensor = tf.reshape(vs_postSet_Tensor, [vs_priorset_Tensor.get_shape()[0], -1])            
            vs_i = vs_postSet_Tensor[K.eval(tf.math.argmax(yi_probs_onVs, axis = 0))[0], 0]
           
            for d in range(multi_res.get_shape()[0]):
                multi_resReshape = tf.reshape(multi_res, [-1])
                resultsPost_dict[str(multi_resReshape[d].numpy())] = vs_postSet_Tensor
                Vs_inferred_dict[str(multi_resReshape[d].numpy())] = vs_i

    assert(len(resultsPost_dict) == x_extends_Tensor.get_shape()[0])
    # concatenate it to the result and avoid to compute repeatedly!
    vs_TensorBatch, vs_postSet_TensorBatch = None, None
    for i in range(x_extends_Tensor.get_shape()[0]):
        vs_i = Vs_inferred_dict[str(i)]         
        vs_postSet_Tensor = resultsPost_dict[str(i)]
        
        vs_i = tf.expand_dims(vs_i, 0)
        vs_postSet_Tensor = tf.expand_dims(vs_postSet_Tensor, 0) # (1, 13,2)
        if i == 0:
            vs_postSet_TensorBatch = vs_postSet_Tensor
            vs_TensorBatch = vs_i
        else:
            vs_postSet_TensorBatch = tf.concat([vs_postSet_TensorBatch, vs_postSet_Tensor], axis = 0)
            vs_TensorBatch = tf.concat([vs_TensorBatch, vs_i], axis = 0)
    
    assert(vs_TensorBatch.get_shape()[0] == x_extends_Tensor.get_shape()[0])
    
    return vs_TensorBatch, vs_priorSet_TensorBatch, vs_postSet_TensorBatch

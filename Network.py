import os
import cv2
import numpy as np
import pandas as pd
import pingouin as pg
from functools import reduce
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,Conv1D,Conv2D,BatchNormalization,Reshape,Activation,ReLU,Input,SimpleRNN,LSTM
from tensorflow.keras.metrics import MeanSquaredError,MeanAbsolutePercentageError,MeanAbsoluteError
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def FC_BN_ReLU(*args, **kwargs):
    """
       Dense & Batch Normalization & relu
    """
    bias_kwargs = {'use_bias': False}
    #no_bias_kwargs['kernel_regularizer'] = l2(5e-4)   #正则化
    bias_kwargs.update(kwargs)
    return compose(
        Dense(*args, **bias_kwargs),
        BatchNormalization(),
        ReLU())

def ROI_weight_network(input_shape):
    """
       信号自适应加权网络
       通过ROI本身的特性，获取加权权重值
    """
    model = Sequential([
        Reshape((input_shape[0],input_shape[1],1),input_shape=input_shape),
        Conv2D(16,(1,3),(1,2),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32,(1,3),(1,2),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(16,(1,3),(1,2),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(8,(1,3),(1,2),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(4,(1,3),(1,2),padding='same',activation='relu'),
        Reshape((input_shape[0],28)),
        Dense(1),
        Reshape((input_shape[0],)),
        Activation('softmax')
    ])
    return model

def HR_map_network(input_shape,mode='RNN'):
    """
       心率信号映射网络
       把加权后的信号映射到心率值上
    """
    if mode == 'RNN':
        inputs = Input(input_shape)
        h = LSTM(units=32,return_sequences=True)(inputs)
        h = LSTM(units=64,return_sequences=True)(h)
        h = LSTM(units=64)(h)
        h = Dense(units=32)(h)
        outputs = Dense(1)(h)

        model = Model(inputs=inputs,outputs=outputs)
    elif mode == 'CNN':
        model = Sequential([
            Conv1D(16,3,strides=2,padding='same',input_shape=input_shape),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(32,3,strides=2,padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(64,3,strides=2,padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(32,3,strides=2,padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(16,3,strides=2,padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(8,3,strides=2,padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(4,3,strides=2,padding='same',activation='relu'),
            Flatten(),
            Dense(1)
        ])
    elif mode == 'DNN':
        inputs = Input(input_shape)
        h = Flatten()(inputs)
        h = FC_BN_ReLU(units=150)(h)
        h = FC_BN_ReLU(units=100)(h)
        h = FC_BN_ReLU(units=20)(h)
        outputs = Dense(1)(h)

        model = Model(inputs=inputs,outputs=outputs)
    
    return model

def HR_network(input_shape,mode='RNN'):
    """
       级联网络，心率预测完整网络
    """
    #  获取加权权重模型
    weight_model = ROI_weight_network(input_shape)
    
    weights = weight_model.output  #  权重值，shape:batch*num_ROI
    weights_reshape = Reshape((weights.shape[1],1))(weights)  #  shape:batch*num_ROI*1
    weighted_ROI_signal = K.sum(weight_model.input * weights_reshape,axis=1)  #  加权得到的信号，shape:batch*250
    weighted_ROI_signal = Reshape((weighted_ROI_signal.shape[1],1))(weighted_ROI_signal) #  shape:batch*250*1
    
    #  获取心率值映射模型
    map_model = HR_map_network(input_shape=(weighted_ROI_signal.shape[1],1),mode=mode)
    
    outputs = map_model(weighted_ROI_signal)  #  shape:batch*1
    
    model = Model(inputs=weight_model.input,outputs=outputs)
    
    return model
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 00:22:25 2020

@author: USUARIO
"""

#%% Libraries and functions
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
from pathlib import Path  
import cv2  
import seaborn as sn  
import pandas as pd   
import keras
from math import ceil

from keras import layers  
from keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, TimeDistributed, AveragePooling1D  
from keras.models import Sequential, Model, load_model  
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint  
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model  
from tensorflow.keras import backend as K
import tensorflow as tf  
import h5py as h5

def ls(ruta = Path.cwd()):
    return [arch.name for arch in Path(ruta).iterdir() if arch.is_file()]


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#%% Organize dataset training
X_train=[]
Y_personality_train=[]
Y_eth_train=[]
Y_gender_train=[]


#Personality
with open('First impressions data set/Train/annotation_training.pkl','rb') as file:
    data=pickle.load(file,encoding='latin1')
    
BigFive=[]
for i in data:
    if i!="interview":
        BigFive.append(i)
    
    
#Ethnicity and gender
eth_gender=""
with open('First impressions data set/eth_gender_annotations_dev.csv', newline='') as File:  
    reader = csv.reader(File)
    for row in reader:
        eth_gender+=row[0]
        eth_gender+=";"  
eth_gender_list=eth_gender.split(";")


#Name of videos (Inputs) and organize Y
temp_bf=[]
for i in range(75):
    if i<9:
        path="First impressions data set/Train/training80_0{}".format(i+1)
    else:
        path="First impressions data set/Train/training80_{}".format(i+1)
    for video in ls(path):
        X_train.append(path+'/'+video)
        index_gender=eth_gender_list.index(video)+3
        index_eth=eth_gender_list.index(video)+2
        for j in BigFive:
            temp_bf.append(data[j][video])
        Y_eth_train.append(int(eth_gender_list[index_eth]))
        Y_gender_train.append(int(eth_gender_list[index_gender]))
        Y_personality_train.append(temp_bf)
        temp_bf=[]


#Make everything a numpy array    
X_train=np.asarray(X_train)
X_train=np.reshape(X_train,(X_train.shape[0],1))
Y_personality_train=np.asarray(Y_personality_train)
Y_eth_train=np.asarray(Y_eth_train)
Y_eth_train=np.reshape(Y_eth_train,(Y_eth_train.shape[0],1))
Y_gender_train=np.asarray(Y_gender_train)
Y_gender_train=np.reshape(Y_gender_train,(Y_gender_train.shape[0],1))

#%% Write dataset training
#12000, 6000 videos, 2 arrays of 100 images for video

#
"""
with h5.File('First impressions data set/Train/training_data.h5', 'w') as hf:
    
    hf.create_dataset("X_train", dtype='uint8', shape=(12000, 100, 248, 248, 3))
    hf.create_dataset("Y_personality", dtype='float64', shape=(12000, 6))
    hf.create_dataset("Y_gender", dtype='int8', shape=(12000, 1))
    hf.create_dataset("Y_eth", dtype='int8', shape=(12000, 1))

    for num, video in enumerate(X_train):
        frame_number=0
        frames_temp=[]
        X_temp=[]
        cap = cv2.VideoCapture(str(video)[2:-2])
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        else: 
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            #leemos un frame y lo guardamos
            for i in range(202):
                try:
                    ret, frame = cap.read()
                    frame_number+=1 
                except:
                    print('Error en el frame ',frame_number+1)
                if ret==True:
                    dim = (248, 248)
                    #resize image
                    frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                    if frame_number<101:
                        frames_temp.append(frame_resized)
                    else:
                        print(frame_number)
                        hf["X_train"][num] = np.asarray(frames_temp)
                        hf["Y_personality"][num] = Y_personality_train[num]
                        hf["Y_gender"][num] = Y_gender_train[num]
                        hf["Y_val"][num] = Y_eth_train[num]
                        frames_temp=[]
                        frame_number=0
            cap.release()
"""

#%% Organize dataset validation
X_val=[]
Y_personality_val=[]
Y_eth_val=[]
Y_gender_val=[]


#Personality
with open('First impressions data set/Validation/annotation_validation.pkl','rb') as file:
    data=pickle.load(file,encoding='latin1')
    
BigFive=[]
for i in data:
    if i!="interview":
        BigFive.append(i)
    
    
#Ethnicity and gender
eth_gender=""
with open('First impressions data set/eth_gender_annotations_dev.csv', newline='') as File:  
    reader = csv.reader(File)
    for row in reader:
        eth_gender+=row[0]
        eth_gender+=";"  
eth_gender_list=eth_gender.split(";")


#Name of videos (Inputs) and organize Y
temp_bf=[]
for i in range(25):
    if i<9:
        path="First impressions data set/Validation/validation80_0{}".format(i+1)
    else:
        path="First impressions data set/Validation/validation80_{}".format(i+1)
    for video in ls(path):
        X_val.append(path+'/'+video)
        index_gender=eth_gender_list.index(video)+3
        index_eth=eth_gender_list.index(video)+2
        for j in BigFive:
            temp_bf.append(data[j][video])
        Y_eth_val.append(int(eth_gender_list[index_eth]))
        Y_gender_val.append(int(eth_gender_list[index_gender]))
        Y_personality_val.append(temp_bf)
        temp_bf=[]


#Make everything a numpy array    
X_val=np.asarray(X_val)
X_val=np.reshape(X_val,(X_val.shape[0],1))
Y_personality_val=np.asarray(Y_personality_val)
Y_eth_val=np.asarray(Y_eth_val)
Y_eth_val=np.reshape(Y_eth_val,(Y_eth_val.shape[0],1))
Y_gender_val=np.asarray(Y_gender_val)
Y_gender_val=np.reshape(Y_gender_val,(Y_gender_val.shape[0],1))
    

#%% Write dataset validation

#4000, 26000 videos, 2 arrays of 100 images for video
"""
with h5.File('First impressions data set/Train/val_data.h5', 'w') as hf:
    
    hf.create_dataset("X_train", dtype='uint8', shape=(4000, 100, 248, 248, 3))
    hf.create_dataset("Y_personality", dtype='float64', shape=(4000, 6))
    hf.create_dataset("Y_gender", dtype='int32', shape=(4000, 1))
    hf.create_dataset("Y_eth", dtype='int32', shape=(4000, 1))

    for num, video in enumerate(X_val):
        print(num)
        frame_number=0
        frames_temp=[]
        X_temp=[]
        cap = cv2.VideoCapture(str(video)[2:-2])
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        else: 
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            #leemos un frame y lo guardamos
            for i in range(202):
                try:
                    ret, frame = cap.read()
                    frame_number+=1 
                except:
                    print('Error en el frame ',frame_number+1)
                if ret==True:
                    dim = (248, 248)
                    #resize image
                    frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                    if frame_number<101:
                        frames_temp.append(frame_resized)
                    else:
                        print(num)
                        print(frame_number)
                        hf["X_train"][num] = np.asarray(frames_temp)
                        hf["Y_personality"][num] = Y_personality_val[num]
                        hf["Y_gender"][num] = Y_gender_val[num]
                        hf["Y_val"][num] = Y_eth_val[num]
                        frames_temp=[]
                        frame_number=0
            cap.release()
"""
#%%  Program functions
            
            
def generate_video_secuences(X, Y, batch_size, size_frame, ventana_analisis): #Batch_size videos a entrar en la red
    while True:    
        number_batches=ceil(X.size/batch_size)
        for batch in range(number_batches):
            X_batch=X[(batch*batch_size):(batch*batch_size)+batch_size]
            #epoch+=1
            Y_temp=[]
            X_temp=[]
            for num, video in enumerate(X_batch):
                frame_number=0
                frames_temp=[]
                
                cap = cv2.VideoCapture(str(video)[2:-2])
                if (cap.isOpened()== False): 
                    print("Error opening video stream or file")
                else: 
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    #leemos un frame y lo guardamos
                    for i in range((ventana_analisis+1)*2):
                        try:
                            ret, frame = cap.read()
                            frame_number+=1 
                        except:
                            print('Error en el frame ',frame_number+1)
                        if ret==True:
                            dim = (size_frame, size_frame)
                            #resize image
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                            frame_resized=(frame_resized/127.5)-1
                            frame_resized=np.reshape(frame_resized,(size_frame,size_frame,1))
                            if frame_number<(ventana_analisis+1):
                                frames_temp.append(frame_resized)
                            else:
                                X_temp.append(frames_temp)
                                Y_temp.append(Y[(batch*batch_size)+num])
                                frames_temp=[]
                                frame_number=0
                    cap.release()
            yield(np.array(X_temp),np.array(Y_temp))
    
    
    
def create_cnn():  
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, kernel_size=(3, 3),activation='relu'),input_shape=(30, 144, 144, 1)))
    #model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    
    #model.add(TimeDistributed(Conv2D(16, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu')))
    
    model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    #model.add(TimeDistributed(Conv2D(2048, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    
    model.add(LSTM(64,activation='relu', return_sequences=False, dropout=0.4))
    #model.add(Dense(200,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(5,activation='sigmoid'))
    
   # model.add(Flatten())
    return model

#%% program
K.clear_session()

sgd=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.1, nesterov=False, name="SGD")
cnn_model=create_cnn()
cnn_model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd, metrics=['acc', 'mse'])  
#cnn_model.build(input_shape=(100, 248, 248, 3))
cnn_model.summary()  


#%% Train and validate generators
batch_size=4
epochs=100
size_frame=144
ventana_analisis=30
X_train=X_train[:320]
Y_personality_train=Y_personality_train[:320]


X_val=X_val[:40]
Y_personality_val=Y_personality_val[:40]




Y_temp=[]
X_temp=[]
for num, video in enumerate(X_val):
    frames_temp=[]
    frame_number=0
    cap = cv2.VideoCapture(str(video)[2:-2])
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    else: 
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #leemos un frame y lo guardamos
        for i in range((ventana_analisis+1)*5):   
            try:
                ret, frame = cap.read()
                frame_number+=1 
            except:
                print('Error en el frame ',frame_number+1)
            if ret==True:
                dim = (size_frame, size_frame)
                #resize image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                frame_resized=(frame_resized/127.5)-1
                frame_resized=np.reshape(frame_resized,(size_frame,size_frame,1))
                if frame_number<(ventana_analisis+1):
                    frames_temp.append(frame_resized)
                else:
                    X_temp.append(frames_temp)
                    Y_temp.append(Y_personality_val[num])
                    frames_temp=[]
                    frame_number=0
        cap.release()
        
X_temp_val=np.array(X_temp);
Y_personality_temp_val=np.array(Y_temp);

train_generator=generate_video_secuences(X_train, Y_personality_train, batch_size, size_frame, ventana_analisis)
#val_generator=generate_video_secuences(X_val, Y_personality_val, batch_size, size_frame, ventana_analisis)
#%% Fit the model
mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
#early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=1)
#callbacks=[early_stopping,mc]
callbacks=[mc]
cnn_model.fit_generator(generator=train_generator,
                             validation_data=(X_temp_val,Y_personality_temp_val),
                             #validation_data=val_generator,
                             #validation_steps=ceil(X_val.zize/batch_size),
                             steps_per_epoch=ceil(X_train.size/batch_size),
                             epochs=epochs,
                             verbose=1,
                             callbacks=callbacks)

#%% PLot the history
plt.figure(0)  
plt.plot(cnn_model.history['acc'],'r')  
plt.plot(cnn_model.history['val_acc'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(cnn_model.history['loss'],'r')  
plt.plot(cnn_model.history['val_loss'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.show() 
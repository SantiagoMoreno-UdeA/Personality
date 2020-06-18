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
from keras.utils import layer_utils, np_utils
from keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, TimeDistributed, AveragePooling1D  
from keras.layers import Conv3D, AveragePooling3D, MaxPooling3D, GlobalMaxPooling3D, GlobalAveragePooling3D 
from keras.models import Sequential, Model, load_model  
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint  
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import load_model  
from tensorflow.keras import backend as K
from scipy import interp  
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import tensorflow as tf  
import h5py as h5

def ls(ruta = Path.cwd()):
    return [arch.name for arch in Path(ruta).iterdir() if arch.is_file()]


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

divide=True;
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
            if divide:
                if data[j][video]<=0.3:
                    temp_bf.append(0)
                elif data[j][video]<=0.6:
                    temp_bf.append(1)
                else:
                    temp_bf.append(2)
            else:
                temp_bf.append(round(data[j][video],2))
        Y_eth_train.append(int(eth_gender_list[index_eth]))
        Y_gender_train.append(int(eth_gender_list[index_gender]))
        if divide:
            #temp_bf= np_utils.to_categorical(temp_bf, 3)  
            Y_personality_train.append(temp_bf)
        else:
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
            if divide:
                if data[j][video]<=0.3:
                    temp_bf.append(0)
                elif data[j][video]<=0.6:
                    temp_bf.append(1)
                else:
                    temp_bf.append(2)
            else:
                temp_bf.append(round(data[j][video],2))
        Y_eth_val.append(int(eth_gender_list[index_eth]))
        Y_gender_val.append(int(eth_gender_list[index_gender]))
        if divide:
            #temp_bf= np_utils.to_categorical(temp_bf, 3)  
            Y_personality_val.append(temp_bf)
        else:
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
#%% Organize dataset test
X_test=[]
Y_personality_test=[]
Y_eth_test=[]
Y_gender_test=[]


#Personality
with open('First impressions data set/Test/annotation_test.pkl','rb') as file:
    data=pickle.load(file,encoding='latin1')
    
BigFive=[]
for i in data:
    if i!="interview":
        BigFive.append(i)
    
    
#Ethnicity and gender
eth_gender=""
with open('First impressions data set/eth_gender_annotations_test.csv', newline='') as File:  
    reader = csv.reader(File)
    for row in reader:
        eth_gender+=row[0]
        eth_gender+=";"  
eth_gender_list=eth_gender.split(";")


#Name of videos (Inputs) and organize Y
temp_bf=[]
for i in range(25):
    if i<9:
        path="First impressions data set/Test/test80_0{}".format(i+1)
    else:
        path="First impressions data set/Test/test80_{}".format(i+1)
    for video in ls(path):
        X_test.append(path+'/'+video)
        index_gender=eth_gender_list.index(video)+3
        index_eth=eth_gender_list.index(video)+2
        for j in BigFive:
            if divide:
                if data[j][video]<=0.3:
                    temp_bf.append(0)
                elif data[j][video]<=0.6:
                    temp_bf.append(1)
                else:
                    temp_bf.append(2)
            else:
                temp_bf.append(round(data[j][video],2))
        Y_eth_test.append(int(eth_gender_list[index_eth]))
        Y_gender_test.append(int(eth_gender_list[index_gender]))
        if divide:
            #temp_bf= np_utils.to_categorical(temp_bf, 3)  
            Y_personality_test.append(temp_bf)
        else:
            Y_personality_test.append(temp_bf)
        temp_bf=[]


#Make everything a numpy array    
X_test=np.asarray(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],1))
Y_personality_test=np.asarray(Y_personality_test)
Y_eth_test=np.asarray(Y_eth_test)
Y_eth_test=np.reshape(Y_eth_test,(Y_eth_test.shape[0],1))
Y_gender_test=np.asarray(Y_gender_test)
Y_gender_test=np.reshape(Y_gender_test,(Y_gender_test.shape[0],1))


#%%  Program functions
            
            
def generate_video_secuences(X, Y, batch_size, size_frame, ventana_analisis): #Batch_size videos a entrar en la red
    while True:    
        number_batches=ceil(X.size/batch_size)
        idx=np.random.permutation(X.size)
        for batch in range(number_batches):
            idx_batch=idx[(batch*batch_size):(batch*batch_size)+batch_size]
            X_batch=X[idx_batch]
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
                            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                            frame_resized=(frame_resized/127.5)-1
                            #frame_resized=np.reshape(frame_resized,(size_frame,size_frame,3))
                            if frame_number<(ventana_analisis+1):
                                frames_temp.append(frame_resized)
                            else:
                                X_temp.append(frames_temp)
                                Y_temp.append(Y[idx_batch[num]])
                                frames_temp=[]
                                frame_number=0
                    cap.release()
            yield(np.array(X_temp),np.array(Y_temp))
            
def generate_video_secuences_test(X, Y, batch_size, size_frame, ventana_analisis): #Batch_size videos a entrar en la red
    while True:    
        number_batches=ceil(X.size/batch_size)
        for batch in range(number_batches):
            X_batch=X[(batch*batch_size):(batch*batch_size)+batch_size]
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
                    for i in range((ventana_analisis+1)):
                        try:
                            ret, frame = cap.read()
                            frame_number+=1 
                        except:
                            print('Error en el frame ',frame_number+1)
                        if ret==True:
                            dim = (size_frame, size_frame)
                            #resize image
                            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                            frame_resized=(frame_resized/127.5)-1
                            #frame_resized=np.reshape(frame_resized,(size_frame,size_frame,3))
                            if frame_number<(ventana_analisis+1):
                                frames_temp.append(frame_resized)
                            else:
                                X_temp.append(frames_temp)
                                Y_temp.append(Y[(batch*batch_size)+num])
                                frames_temp=[]
                                frame_number=0
                    cap.release()
            yield(np.array(X_temp),np.array(Y_temp))
            
def get_videos(X,Y,ventana_analisis):
    X_temp=[]
    Y_temp=[]
    for num, video in enumerate(X):
        frames_temp=[]
        frame_number=0
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
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                    frame_resized=(frame_resized/127.5)-1
                    #frame_resized=np.reshape(frame_resized,(size_frame,size_frame,3))
                    if frame_number<(ventana_analisis+1):
                        frames_temp.append(frame_resized)
                    else:
                        X_temp.append(frames_temp)
                        Y_temp.append(Y[num])
                        frames_temp=[]
                        frame_number=0
            cap.release()
    return(np.array(X_temp),np.array(Y_temp))
    
def create_cnn(ventana_analisis,size_frame):  
    model = Sequential()
    model.add(TimeDistributed(Conv2D(8, kernel_size=(3, 3),activation='relu'),input_shape=(ventana_analisis, size_frame, size_frame, 3)))
    model.add(TimeDistributed(Conv2D(16, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    
    model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3), activation='relu')))
    
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(512, kernel_size=(3, 3), activation='relu')))
    #model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3), activation='relu')))
   
    #model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    
    model.add(LSTM(256,activation='relu', return_sequences=False, dropout=0.4))
    #model.add(Dense(200,activation='relu'))
    #model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    #model.add(Dense(1,activation='linear'))
    
   #model.add(Flatten())
    return model

def create_cnn3d(ventana_analisis,size_frame):  
    model = Sequential()
    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(ventana_analisis,size_frame,size_frame,3)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))

    #model.add(TimeDistributed(GlobalAveragePooling2D()))
    #model.add(LSTM(64,activation='relu', return_sequences=False, dropout=0.4))
    #model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(1,activation='linear'))
    
   # model.add(Flatten())
    return model

def ROC_curve(n_classes,Y_test,vgg16_pred, model,num=0, early=False, manual=False):
# Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()  
    tpr = dict()  
    roc_auc = dict()  
    for i in range(n_classes):  
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], vgg16_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), vgg16_pred.ravel())  
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)  
    for i in range(n_classes):  
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr  
    tpr["macro"] = mean_tpr  
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)  
    plt.plot(fpr["micro"], tpr["micro"],  
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],  
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'violet'])  
    for i, color in zip(range(n_classes), colors):  
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Some extension of Receiver operating characteristic to multi-class')  
    plt.legend(loc="lower right")  
    if early:
        plt.savefig("ROC_{}_early_{}.png".format(model,num),bbox_inches='tight')
    elif manual:
        plt.savefig("ROC_{}_manual_{}.png".format(model,num),bbox_inches='tight')
    else:
        plt.savefig("ROC_ventana{}_optimizer{}_Loss{}_features{}.png".format(ventana_analisis,optimizer, loss_fun,features),bbox_inches='tight')
    plt.show()




    # # Zoom in view of the upper left corner.
    # plt.figure(2)  
    # plt.xlim(0, 0.2)  
    # plt.ylim(0.8, 1)  
    # plt.plot(fpr["micro"], tpr["micro"],  
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],  
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'violet'])  
    # for i, color in zip(range(n_classes), colors):  
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)  
    # plt.xlabel('False Positive Rate')  
    # plt.ylabel('True Positive Rate')  
    # plt.title('Some extension of Receiver operating characteristic to multi-class')  
    # plt.legend(loc="lower right")  
    # plt.show()  
#%% program
loss_fun='categorical_crossentropy'
optimizer='sgd'
features=1
batch_size=2   
epochs=300
size_frame=144
ventana_analisis=30 
K.clear_session()
ceros=0
unos=0
dos=0

Y=[]
X=[]
if features==1:
    Y_personality_train=Y_personality_train[:,0]
    Y_personality_val=Y_personality_val[:,0]
    Y_personality_test=Y_personality_test[:,0]
    if divide==1:
        for index,feat in enumerate(Y_personality_train):
            if feat==0 and ceros<500:
                Y.append(feat)
                X.append(X_train[index])
                ceros+=1
            elif feat==1 and unos<500:
                Y.append(feat)
                X.append(X_train[index])
                unos+=1
            elif feat==2 and dos<500:
                Y.append(feat)
                X.append(X_train[index])
                dos+=1
        Y_personality_train=np_utils.to_categorical(Y, 3)
        X_train=np.asarray(X)
        Y=[]
        X=[]
        ceros=0
        unos=0
        dos=0
        
        for index,feat in enumerate(Y_personality_val):
            if feat==0 and ceros<50:
                Y.append(feat)
                X.append(X_val[index])
                ceros+=1
            elif feat==1 and unos<50:
                Y.append(feat)
                X.append(X_val[index])
                unos+=1
            elif feat==2 and dos<50:
                Y.append(feat)
                X.append(X_val[index])
                dos+=1
        Y_personality_val=np_utils.to_categorical(Y, 3)
        X_val=np.asarray(X)
        Y=[]
        X=[]

        
        for index,feat in enumerate(Y_personality_test):
            if feat==0:
                Y.append(feat)
                X.append(X_test[index])
            elif feat==1:
                Y.append(feat)
                X.append(X_test[index])
            elif feat==2:
                Y.append(feat)
                X.append(X_test[index])
        Y_personality_test=np_utils.to_categorical(Y, 3)
        X_test=np.asarray(X)
        
        Y_personality_train=np.reshape(Y_personality_train,(Y_personality_train.shape[0],3))
        Y_personality_val=np.reshape(Y_personality_val,(Y_personality_val.shape[0],3))
        Y_personality_test=np.reshape(Y_personality_test,(Y_personality_test.shape[0],3))
    else:
        Y_personality_train=np.reshape(Y_personality_train,(Y_personality_train.shape[0],1))
        Y_personality_val=np.reshape(Y_personality_val,(Y_personality_val.shape[0],1))
        Y_personality_test=np.reshape(Y_personality_test,(Y_personality_test.shape[0],1))



sgd=tf.keras.optimizers.SGD(learning_rate=1, momentum=0.5, nesterov=False, name="SGD")
cnn_model=create_cnn(ventana_analisis,size_frame)
cnn_model.compile(loss=loss_fun, optimizer=optimizer, metrics=['acc', 'mse'])  
#cnn_model.build(input_shape=(100, 248, 248, 3))
cnn_model.summary()  


#%% Train and validate generators

X_train_recort=X_train[10:]
Y_personality_train_recort=Y_personality_train[10:,:]


X_val=X_val
Y_personality_val=Y_personality_val

        
#X_temp_val, Y_personality_temp_val=get_videos(X_val,Y_personality_val,ventana_analisis);
#X_temp_test, Y_personality_temp_test=get_videos(X_train[:10],Y_personality_train[:10,:],ventana_analisis);


train_generator=generate_video_secuences(X_train_recort, Y_personality_train_recort, batch_size, size_frame, ventana_analisis)
val_generator=generate_video_secuences(X_val, Y_personality_val, batch_size, size_frame, ventana_analisis)
test_generator=generate_video_secuences_test(X_test, Y_personality_test, batch_size, size_frame, ventana_analisis)

checkpoint_filepath='best_model_ventana{}_optimizer{}_Loss{}_features{}.h5'.format(ventana_analisis,optimizer, loss_fun,features)
#%% Fit the model

mc = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=25, verbose=1)
callbacks=[early_stopping,mc]
#callbacks=[mc]
cnn_model.fit_generator(generator=train_generator,
                             #validation_data=(X_temp_val,Y_personality_temp_val),
                             validation_data=val_generator,
                             validation_steps=ceil(X_val.size/batch_size),
                             steps_per_epoch=ceil(X_train_recort.size/batch_size),
                             epochs=epochs,
                             verbose=1,
                             callbacks=callbacks)


   #%% Load the best model
cnn_model.load_weights(checkpoint_filepath)
Y_pred=cnn_model.predict(test_generator, steps=ceil(X_test.size/batch_size))

  #%% Metrics
Y_pred_arg=np.reshape(np.argmax(Y_pred, axis=1),(Y_pred.shape[0],1))
Y_personality_test_arg=np.reshape(np.argmax(Y_personality_test, axis=1),(Y_personality_test.shape[0],1))

#%%
# Visualiamos la matriz de confusión
matrix=confusion_matrix(Y_personality_test_arg, Y_pred_arg)

matrix_cm = pd.DataFrame(matrix, range(3), range(3))  
plt.figure(2,figsize = (20,14))  
sn.set(font_scale=1.4) #for label size  
sn.heatmap(matrix_cm , annot=True, annot_kws={"size": 12}) # font size  
plt.savefig("Confusion_matrix_ventana{}_optimizer{}_Loss{}_features{}.png".format(ventana_analisis,optimizer, loss_fun,features),bbox_inches='tight')
plt.show()
    
ROC_curve(3,Y_personality_test, Y_pred, 'sgd')
  #%% PLot the history

plt.figure(0)  
plt.plot(cnn_model.history.history['acc'],'r')  
plt.plot(cnn_model.history.history['val_acc'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])
plt.savefig("Accuracy_ventana{}_optimizer{}_Loss{}_features{}.png".format(ventana_analisis,optimizer, loss_fun,features))

plt.figure(1)  
plt.plot(cnn_model.history.history['loss'],'r')  
plt.plot(cnn_model.history.history['val_loss'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])
plt.savefig("Loss_ventana{}_optimizer{}_Loss{}_features{}.png".format(ventana_analisis,optimizer, loss_fun,features))

plt.show() 

#plt.hist(Y_personality_train)
#Run CNN-LSTM

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) #상위 디렉토리를 sys.path에 추가

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import seaborn as sns
import datetime as datetime
import seaborn as sns
import argparse
import yaml
import numpy as np
import matplotlib

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from codes.models.model_CNN_LSTM import Model_CNN_LSTM

with open('../config/config_CNN_LSTM.yaml') as config_file:
    doc = yaml.load(config_file, Loader=yaml.FullLoader)
    config = doc['CNN_LSTM']

if config['DIR_Gait'] == 'H':
    DIR_Gait = ['Acc_H', 'Ang_H']
elif config['DIR_Gait'] == 'B':
    DIR_Gait = ['Acc_B', 'Ang_B']
elif config['DIR_Gait'] == 'W':
    DIR_Gait = ['Acc_W', 'Ang_W']
elif config['DIR_Gait'] == 'all':
    DIR_Gait = ['Acc_H', 'Ang_H', 'Acc_B', 'Ang_B', 'Acc_W', 'Ang_W']
else:
    print('Wrong gait setting')

DIR_Sampling = config['DIR_Sampling']

DIR_GAF= config['DIR_GAF']
DIR_Subject = config['DIR_Subject']
DIR_Result = config['DIR_Result']
epoch = config['epoch']
lr = config['lr']
batch_size = config['batch_size']
class_num = 6

time_now = datetime.datetime.now()
formatted_time = time_now.strftime('%Y%m%d_%H%M%S')
Name = formatted_time
print(Name)

## Load Data
Files = os.listdir(DIR_GAF)

FileNameSet_Acc = []
FileNameSet_Ang = []

for Subject in DIR_Subject:
    for File in Files:
            if Subject+"_" in File:

                for gait in DIR_Gait:
                    if (gait in File)&('Acc' in File):
                        FileNameSet_Acc.append(File)

                    elif (gait in File)&('Ang' in File):
                        FileNameSet_Ang.append(File)
            else:
                pass

print("***********************")
print("FileName_Acc:", FileNameSet_Acc)
print("FileName_Ang:", FileNameSet_Ang)

DataSet_Acc = []
DataSet_Ang = []
LabelSet = []

if config['DIR_Gait'] == 'all':
    for i in range(0,len(DIR_Subject)):
        Data_Acc_1 = np.array(pd.read_csv(DIR_GAF + FileNameSet_Acc[i])).reshape(306, 10, 3, 30, 30)
        Data_Ang_1 = np.array(pd.read_csv(DIR_GAF + FileNameSet_Ang[i])).reshape(306, 10, 3, 30, 30)
        Data_Acc_2 = np.array(pd.read_csv(DIR_GAF + FileNameSet_Acc[i+1])).reshape(306, 10, 3, 30, 30)
        Data_Ang_2 = np.array(pd.read_csv(DIR_GAF + FileNameSet_Ang[i+1])).reshape(306, 10, 3, 30, 30)
        Data_Acc_3 = np.array(pd.read_csv(DIR_GAF + FileNameSet_Acc[i+2])).reshape(306, 10, 3, 30, 30)
        Data_Ang_3 = np.array(pd.read_csv(DIR_GAF + FileNameSet_Ang[i+2])).reshape(306, 10, 3, 30, 30)
        Data_Acc = np.concatenate((Data_Acc_1, Data_Acc_2, Data_Acc_3), axis=1)
        Data_Ang = np.concatenate((Data_Ang_1, Data_Ang_2, Data_Ang_3), axis=1)

        #del Data_Acc_1, Data_Ang_1, Data_Acc_2, Data_Ang_2, Data_Acc_3, Data_Ang_3

        Label = pd.read_csv(DIR_Sampling + DIR_Subject[i] + '_Label.csv', header = None) 
            
        DataSet_Acc.append(Data_Acc)
        DataSet_Ang.append(Data_Ang)    
        LabelSet.append(Label)
        
        #del Data_Acc, Data_Ang, Label

else:
    for i in range(0,len(DIR_Subject)):
        Data_Acc = np.array(pd.read_csv(DIR_GAF + FileNameSet_Acc[i])).reshape(306, 10, 3, 30, 30) #time(51 window*16 activity), feature, rgb, image size #
        Data_Ang = np.array(pd.read_csv(DIR_GAF + FileNameSet_Ang[i])).reshape(306, 10, 3, 30, 30) 
            
        Label = pd.read_csv(DIR_Sampling + DIR_Subject[i] + '_Label.csv', header = None) 

        DataSet_Acc.append(Data_Acc)
        DataSet_Ang.append(Data_Ang)    
        LabelSet.append(Label)
            
        #del Data_Acc, Data_Ang, Label

print("***********************")
print("Shape_Acc:", np.array(DataSet_Acc).shape)
print("Shape_Ang:", np.array(DataSet_Ang).shape)
print("Shape_Label:", np.array(LabelSet).shape)

## Split Data
DataSet_Acc_ = np.vstack(DataSet_Acc)
DataSet_Ang_ = np.vstack(DataSet_Ang)
LabelSet_ = np.vstack(LabelSet)

Data_Tr_Acc, Data_Te_Acc, Data_Tr_Ang, Data_Te_Ang, Label_Tr, Label_Te = train_test_split(DataSet_Acc_, DataSet_Ang_, LabelSet_, test_size = 0.3, random_state=7, shuffle=True)
Data_Tr_Acc, Data_Va_Acc, Data_Tr_Ang, Data_Va_Ang, Label_Tr, Label_Va = train_test_split(Data_Tr_Acc, Data_Tr_Ang, Label_Tr, test_size = 0.1, random_state=7, shuffle=True)

#del DataSet_Acc, DataSet_Ang, LabelSet

print("***********************")
print('+ Training Shape')
print(Data_Tr_Acc.shape)
print(Data_Tr_Ang.shape)
print(Label_Tr.shape)

print('+ Testing Shape')
print(Data_Te_Acc.shape)
print(Data_Te_Ang.shape)
print(Label_Te.shape)

print('+ Validation Shape')
print(Data_Va_Acc.shape)
print(Data_Va_Ang.shape)
print(Label_Va.shape)

## Training
W = len(Data_Tr_Acc[0])
X = len(Data_Tr_Acc[0][0])
Y = len(Data_Tr_Acc[0][0][0])
Z = len(Data_Tr_Acc[0][0][0])

Class = ModelCheckpoint(DIR_Result + Name + '_Weight.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
His = CSVLogger(DIR_Result + Name + '_History.csv', append=True, separator=';')

Model_Final = Model_CNN_LSTM((W, X, Y, Z), class_num)
History = Model_Final.fit([Data_Tr_Acc, Data_Tr_Ang], [Label_Tr], validation_data=([Data_Va_Acc, Data_Va_Ang], [Label_Va]),
                           batch_size=batch_size, epochs=epoch, callbacks = [Class, His], verbose=1)

plot_model(Model_Final, to_file= DIR_Result + Name +'_Model.png', show_shapes=True, show_layer_names=True)

## Font
matplotlib.font_manager.fontManager.addfont('./times-new-roman.ttf')

LabelFont = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 35}

UnitFont = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 25}

LegendFont = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 30}

plt.rc('axes', unicode_minus=False)

## ACCURACY
plt.figure(figsize=(6,4))
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xlim(0,epoch)
plt.ylim(0,1)
plt.xticks(np.linspace(0,epoch,class_num))
plt.yticks(np.linspace(0,1,class_num))
plt.plot(History.history['accuracy'], 'black')
plt.plot(History.history['val_accuracy'], 'gray')
plt.legend(['Train', 'Val', 'Test'], loc='lower right', prop={'family' : 'Times New Roman', 'size': 15})
plt.savefig(DIR_Result + Name + '_Result' + '_Accuracy.png', bbox_inches='tight', dpi=300, transparent = True)
plt.close()

## LOSS
plt.figure(figsize=(class_num,4))
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0,epoch)
plt.ylim(0,1.5)
plt.xticks(np.linspace(0,epoch,class_num))
plt.yticks(np.linspace(0,1.5,class_num))
plt.plot(History.history['loss'], 'black')
plt.plot(History.history['val_loss'], 'gray')
plt.legend(['Train', 'Val', 'Test'], loc='upper right', prop={'family' : 'Times New Roman', 'size': 15})
plt.savefig(DIR_Result + Name + '_Result' + '_Loss.png', bbox_inches='tight', dpi=300, transparent = True)
plt.close()

## Test
Model = load_model(DIR_Result + Name + '_Weight.hdf5')

print("-- Evaluate --")
scores = Model.evaluate([Data_Te_Acc, Data_Te_Ang], [Label_Te], batch_size=4)
print("%s: %.4f%%" %(Model.metrics_names[0], scores[0]))
print("%s: %.4f%%" %(Model.metrics_names[1], scores[1]*100))

predictions = Model_Final.predict([Data_Te_Acc, Data_Te_Ang])
print(len(predictions))
pre = []
for i in range(len(predictions)):
  pre.append(np.argmax(predictions[i]))
ann = []
for i in range(len(Label_Te)):
  ann.append(list(Label_Te[i]).index(1))
df= pd.DataFrame({"pre": pre, "ann": ann})
df.to_csv(DIR_Result + Name + "_test.csv")

predictions = Model_Final.predict([Data_Va_Acc, Data_Va_Ang])
print(len(predictions))
pre = []
for i in range(len(predictions)):
  pre.append(np.argmax(predictions[i]))
ann = []
for i in range(len(Label_Va)):
  ann.append(list(Label_Va[i]).index(1))
df= pd.DataFrame({"pre": pre, "ann": ann})
df.to_csv(DIR_Result + Name + "_val.csv")

df = pd.read_csv(DIR_Result + '%s_test.csv'%(Name))

list_ann = sorted(list(set(df["ann"])))
list_pre = sorted(list(set(df["pre"])))
list_big = []

for ann in list_ann:
    list_small = []

    for pre in list_pre:
        condition1 = (df['ann'] == ann) 
        condition2 = (df['pre'] == pre)
        count = df.loc[condition1 & condition2, 'ann'].count()
        list_small.append(count)
    list_big.append(list_small)

df_2 = pd.DataFrame(list_big)
df_2.index = list_ann
df_2.columns = list_pre

plt.rc('font', family='times-new-roman') 

sns.heatmap(df_2, annot=True, fmt='d')
plt.title(Name, fontsize=20)

plt.savefig(DIR_Result +'./%s_heatmap.png'%(Name), transparent = True)
plt.close()

## Log
'''
end_time = time.time()

log = [Name, TimeStamp, str(end_time-start_time), epoch, lr, test_acc, test_loss]
df_log = pd.read_csv(DIR_Result + 'Log.csv', index_col = 'Index')
df_log.loc[len(df_log)+1] = log
df_log.to_csv(DIR_Result + 'Log.csv')'''
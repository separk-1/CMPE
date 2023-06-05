#dataset을 CNN-LSTM의 input에 맞도록 전처리(sampling)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) #상위 디렉토리를 sys.path에 추가

import pandas as pd
import yaml
from tensorflow.keras.utils import to_categorical

from codes.functions.functions_set_IMU import Set_IMU
from codes.functions.functions_set_tool import Tool_File_Sort
from codes.functions.functions_set_sampling import Set_Sampling
from codes.functions.functions_set_utility import Set_Utility_Save_Label

with open('../../config/config_preprocessing.yaml') as config_file:
    doc = yaml.load(config_file, Loader=yaml.FullLoader)
    config = doc['PREPROCESSING']['Sampling']

DIR_Subject = config['DIR_Subject']
DIR_Raw = config['DIR_Raw']
DIR_Gait = config['DIR_Gait']
DIR_Sampling = config['DIR_Sampling']

LabelSet = []
IMUData = pd.DataFrame()

for Subject in DIR_Subject:
    ann = pd.read_csv(DIR_Raw + Subject + '/%s_ann.csv'%(Subject))
    ann.values.astype('int')

    line_start = []
    line_end = []
    ls = ann.at[0, "line_start"]
    line_start.append(ann.at[0, "line_start"])
    for i in range(len(ann)):
        le = ls + 128*ann["duration"][i]+1
        line_end.append(le)
        if i != len(ann)-1:
            ls = ls +128*(ann["start"][i+1] - ann["start"][i])
            line_start.append(ls)

    point= []
    for i in range(len(line_start)):
        temp = [line_start[i], line_end[i]]
        point.append(temp)

    Label_Sub = list(ann["activity"][1:])
    Range_Sub = point[1:]

    IMUData = pd.read_csv(DIR_Raw + Subject + '/%s.csv'%(Subject), encoding='cp949')

    IMU_Set_Sub_Temp = Set_IMU(IMUData).Set_IMU()

    for i in range(0,6):

        IMU_Set_Sub = []

        for j in range(0,3):

            IMU_Set_Sub.append(IMU_Set_Sub_Temp[i*3+j]) 

        Sample_Set, Label_Set = Set_Sampling(Range_Sub, Label_Sub, IMU_Set_Sub)

        Label_Set = to_categorical(Label_Set, num_classes = 6) # 0 제외, 1~5

        # Save #
        Sample_Name = DIR_Sampling + Subject + '_' + DIR_Gait[i] + '.csv'
        Label_Name = DIR_Sampling + Subject + '_Label.csv'

        Set_Utility_Save_Label(Sample_Set, Sample_Name)
        Set_Utility_Save_Label(Label_Set, Label_Name)


    
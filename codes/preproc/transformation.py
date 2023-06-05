#dataset을 CNN-LSTM의 input에 맞도록 전처리(GAF)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) #상위 디렉토리를 sys.path에 추가

import yaml
import pandas as pd
import numpy as np

from codes.functions.functions_set_tool import Tool_File_Sort
from codes.functions.functions_set_data import Set_Data_Segmentation, Set_Data_GAF
from codes.functions.functions_set_utility import Set_Utility_Save_GAF

with open('../../config/config_preprocessing.yaml') as config_file:
    doc = yaml.load(config_file, Loader=yaml.FullLoader)
    config = doc['PREPROCESSING']['Transformation']

DIR_GAF = config['DIR_GAF']
DIR_Sampling = config['DIR_Sampling']

Files = os.listdir(DIR_Sampling)
Files = Tool_File_Sort(Files)
for File in Files:
    if 'Label' in File:
        pass
    else:
        Sample = pd.read_csv(DIR_Sampling + '/' + File, header = None)
        Sample = np.array(Sample)

        Segment_Set = Set_Data_Segmentation(Sample)
        
        # Transformation # 
        GAF_Set = Set_Data_GAF(Segment_Set) # Gramian Anugular Field

        # Save #
        GAF_Name = DIR_GAF + File.replace('.csv', '_GAF.csv')
        Set_Utility_Save_GAF(GAF_Set, GAF_Name)
        print(GAF_Name)
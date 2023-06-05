import pandas as pd
import numpy as np
import csv

def Set_Utility_Save_Label(Label, Name):

  df = pd.DataFrame(Label)
  df.to_csv(Name, index = False, header = False)

def Set_Utility_Save_GAF(GAF, Name):
    
   # Feature #
    with open(Name,"w+",newline="") as CSV:

        CSVWriter = csv.writer(CSV,delimiter=',')

        Header = np.zeros((1,len(GAF[0][0][0][0])))
        CSVWriter.writerows(Header)

        for i in range(0,len(GAF)):

            for j in range(0,len(GAF[0])):

                for k in range(0,len(GAF[0][0])):

                    CSVWriter.writerows(GAF[i][j][k])
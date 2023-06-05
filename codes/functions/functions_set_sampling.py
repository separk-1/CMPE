import numpy as np
def Set_Sampling(Range_Sub, Label_Sub, IMU_Set_Sub):

    Sample_Set = []
    Label_Set = []

    for i in range(0,len(Range_Sub)): # 16

        Label = Label_Sub[i]

        for j in range(0,51): # 55 sec - window size. window 51개

            Window = j * 100
            Sample_Set_Temp = []

            for k in range(0,len(IMU_Set_Sub)): # 3 axis
                Sample = IMU_Set_Sub[k][int(Range_Sub[i][0]+Window):int(Range_Sub[i][0]+Window+500)] # 5 seconds 
                Sample_Set.append(Sample)
                #print('******************')
                #print(np.array(Sample_Set).shape)
                #print('activity:', i)
                #print('window:', j)
                #print('axis:', k)
                #print('******************')
                #print("activity:%s, window:%s, axis: %s, range: %s:%s"%(i, j, k, Range_Sub[i][0]+Window, Range_Sub[i][0]+Window+500))
                
            Sample_Set_Temp.append(Sample)

            Label_Set_Temp = np.zeros(np.array(Sample_Set_Temp).shape[0])
            Label_Set_Temp[:] = Label
            Label_Set = np.append(Label_Set, Label_Set_Temp)
            
    
    return (Sample_Set, Label_Set)
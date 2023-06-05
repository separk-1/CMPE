import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.preprocessing import minmax_scale
from codes.functions.functions_set_filter import Set_Filter_BL


class Set_IMU:
    def __init__(self, IMUData):
        self.IMUdata = IMUData
        self.AccX_H = minmax_scale(Set_Filter_BL(IMUData['Acceleration X (m/s^2)'])) #본인 센서에 맞게 바꿔줘야함
        self.AccY_H = minmax_scale(Set_Filter_BL(IMUData['Acceleration Y (m/s^2)']))
        self.AccZ_H = minmax_scale(Set_Filter_BL(IMUData['Acceleration Z (m/s^2)']))

        self.AngX_H = minmax_scale(Set_Filter_BL(IMUData['Angular Velocity X (rad/s)']))
        self.AngY_H = minmax_scale(Set_Filter_BL(IMUData['Angular Velocity Y (rad/s)']))
        self.AngZ_H = minmax_scale(Set_Filter_BL(IMUData['Angular Velocity Z (rad/s)']))

        self.AccX_B = minmax_scale(Set_Filter_BL(IMUData['Acceleration X (m/s^2).1']))
        self.AccY_B = minmax_scale(Set_Filter_BL(IMUData['Acceleration Y (m/s^2).1']))
        self.AccZ_B = minmax_scale(Set_Filter_BL(IMUData['Acceleration Z (m/s^2).1']))

        self.AngX_B = minmax_scale(Set_Filter_BL(IMUData['Angular Velocity X (rad/s).1']))
        self.AngY_B = minmax_scale(Set_Filter_BL(IMUData['Angular Velocity Y (rad/s).1']))
        self.AngZ_B = minmax_scale(Set_Filter_BL(IMUData['Angular Velocity Z (rad/s).1']))

        self.AccX_W = minmax_scale(Set_Filter_BL(IMUData['Acceleration X (m/s^2).2']))
        self.AccY_W = minmax_scale(Set_Filter_BL(IMUData['Acceleration Y (m/s^2).2']))
        self.AccZ_W = minmax_scale(Set_Filter_BL(IMUData['Acceleration Z (m/s^2).2']))
        
        self.AngX_W = minmax_scale(Set_Filter_BL(IMUData['Angular Velocity X (rad/s).2']))
        self.AngY_W = minmax_scale(Set_Filter_BL(IMUData['Angular Velocity Y (rad/s).2']))
        self.AngZ_W = minmax_scale(Set_Filter_BL(IMUData['Angular Velocity Z (rad/s).2']))

    def Set_IMU(self): #for CNN-LSTM model
        IMU_Set = []

        # Set Data #
        IMU_Set.append(self.AccX_H)
        IMU_Set.append(self.AccY_H)
        IMU_Set.append(self.AccZ_H)

        IMU_Set.append(self.AngX_H)
        IMU_Set.append(self.AngY_H)
        IMU_Set.append(self.AngZ_H)

        IMU_Set.append(self.AccX_B)
        IMU_Set.append(self.AccY_B)  
        IMU_Set.append(self.AccZ_B)
                
        IMU_Set.append(self.AngX_B)    
        IMU_Set.append(self.AngY_B)
        IMU_Set.append(self.AngZ_B)

        IMU_Set.append(self.AccX_W)
        IMU_Set.append(self.AccY_W)  
        IMU_Set.append(self.AccZ_W)

        IMU_Set.append(self.AngX_W)    
        IMU_Set.append(self.AngY_W)
        IMU_Set.append(self.AngZ_W)


        return (IMU_Set)